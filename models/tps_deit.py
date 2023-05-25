#!/usr/bin/env python3
# Hacked together by / Copyright 2021 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""Vision Transformer (ViT)
ViT: `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
<https://arxiv.org/abs/2010.11929>`_
DeiT: `"Training data-efficient image transformers & distillation through attention"
<https://arxiv.org/abs/2012.12877>`_
References:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import reduce
from collections import OrderedDict
from typing import Callable, Optional, Union

import os
import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
from basecls.layers import (DropPath, activation, init_vit_weights, norm2d,
                            trunc_normal_)
from basecls.utils import recursive_update, registers
from loguru import logger
from megengine.random import uniform
from megengine.utils.tuple_function import _pair as to_2tuple
from .tps import TPS
from functools import partial


PROJECT_DIR = os.path.split(os.path.abspath(__file__))[
    0].replace("models", 'pretrained')


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-6, dim=-1):
    def _gen_gumbels():
        gumbels = -F.log(-F.log(uniform(0, 1, size=logits.shape) + eps)+eps)
        # while F.isnan(gumbels).sum() or F.isinf(gumbels).sum():
        #     # to avoid zero in exp output
        #     gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = F.softmax(gumbels, dim)

    if hard:
        # Straight through.
        index = F.argmax(y_soft, dim, keepdims=True)
        y_hard = F.scatter(F.zeros_like(logits), dim,
                           index, F.ones_like(index))
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def batch_index_select(x, idx):
    B, N, C = x.shape[0], x.shape[1], x.shape[2]
    N_new = idx.shape[1]
    offset = F.arange(0, B, 1, dtype=np.int32).reshape(B, 1) * N
    idx = idx + offset
    out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
    return out


class PatchEmbed(M.Module):
    """Image to Patch Embedding
    Args:
        img_size: Image size.  Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        flatten: Flatten embedding. Default: ``True``
        norm_name: Normalization layer. Default: ``None``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        norm_name: str = None,
        **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = M.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)
        self.norm = norm2d(norm_name, embed_dim) if norm_name else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = F.flatten(x, 2).transpose(0, 2, 1)
        if self.norm:
            x = self.norm(x)
        return x


class Attention(M.Module):
    """Self-Attention block.
    Args:
        dim: input Number of input channels.
        num_heads: Number of attention heads. Default: ``8``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``False``
        qk_scale: Override default qk scale of ``head_dim ** -0.5`` if set.
        attn_drop: Dropout ratio of attention weight. Default: ``0.0``
        proj_drop: Dropout ratio of output. Default: ``0.0``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = M.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = M.Softmax(axis=-1)
        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.shape[0], policy.shape[1]
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = F.eye(N, dtype=attn_policy.dtype,
                    device=attn_policy.device).reshape(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = attn.max(axis=-1, keepdims=True)
        attn = attn - max_att
        # for stable training
        attn = F.exp(attn.astype(attn_policy.dtype)) * \
            attn_policy.astype(attn_policy.dtype)
        attn = (attn + eps/N) / (attn.sum(axis=-1, keepdims=True) + eps)
        return attn.astype(max_att.dtype)

    def forward(self, x, policy=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if policy is None:
            attn = self.softmax(attn)
        else:
            attn = self.softmax_with_policy(attn, policy)

        attn = self.attn_drop(attn)
        x = F.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(M.Module):
    """FFN for ViT
    Args:
        in_features: Number of input features.
        hidden_features: Number of input features. Default: ``None``
        out_features: Number of output features. Default: ``None``
        drop: Dropout ratio. Default: ``0.0``
        act_name: activation function. Default: ``"gelu"``
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
        act_name: str = "gelu",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Linear(in_features, hidden_features)
        self.act = activation(act_name)
        self.fc2 = M.Linear(hidden_features, out_features)
        self.drop = M.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(M.Module):
    """Transformer Encoder block.
    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``False``
        qk_scale: Override default qk scale of ``head_dim ** -0.5`` if set.
        drop: Dropout ratio of non-attention weight. Default: ``0.0``
        attn_drop: Dropout ratio of attention weight. Default: ``0.0``
        drop_path: Stochastic depth rate. Default: ``0.0``
        norm_name: Normalization layer. Default: ``"LN"``
        act_name: Activation layer. Default: ``"gelu"``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_name: str = "LN",
        act_name: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm2d(norm_name, dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = norm2d(norm_name, dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.mlp = FFN(
            in_features=dim, hidden_features=ffn_hidden_dim, drop=drop, act_name=act_name
        )

    def forward(self, x, policy=None):
        if self.drop_path:
            x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x), policy=policy)
            x = x + self.mlp(self.norm2(x))
        return x


class PredictorLG(M.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = M.Sequential(
            M.LayerNorm(embed_dim),
            M.Linear(embed_dim, embed_dim),
            M.GELU()
        )

        self.out_conv = M.Sequential(
            M.Linear(embed_dim, embed_dim // 2),
            M.GELU(),
            M.Linear(embed_dim // 2, embed_dim // 4),
            M.GELU(),
            M.Linear(embed_dim // 4, 2),
        )

    def forward(self, x, policy=None):
        x = self.in_conv(x)
        B, N, C = x.shape[0], x.shape[1], x.shape[2]
        local_x = x[:, :, :C//2]
        if policy is not None:
            global_x = (x[:, :, C//2:] * policy).sum(axis=1,
                                                     keepdims=True) / F.sum(policy, axis=1, keepdims=True)
        else:
            global_x = x[:, :, C//2:].mean(axis=1, keepdims=True)
        x = F.concat([local_x, F.broadcast_to(
            global_x, (B, N, C//2))], axis=-1)
        x = F.logsoftmax(self.out_conv(x), axis=-1)
        return x


@registers.models.register()
class TPSViT(M.Module):
    """ViT model.
    Args:
        img_size: Input image size. Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        depth: Depth of Transformer Encoder layer. Default: ``12``
        num_heads: Number of attention heads. Default: ``12``
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
        qk_scale: Override default qk scale of head_dim ** -0.5 if set. Default: ``None``
        representation_size: Size of representation layer (pre-logits). Default: ``None``
        distilled: Includes a distillation token and head. Default: ``False``
        drop_rate: Dropout rate. Default: ``0.0``
        attn_drop_rate: Attention dropout rate. Default: ``0.0``
        drop_path_rate: Stochastic depth rate. Default: ``0.0``
        embed_layer: Patch embedding layer. Default: :py:class:`PatchEmbed`
        norm_name: Normalization layer. Default: ``"LN"``
        act_name: Activation function. Default: ``"gelu"``
        num_classes: Number of classes. Default: ``1000``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        representation_size: int = None,
        distilled: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: M.Module = PatchEmbed,
        norm_name: str = "LN",
        act_name: str = "gelu",
        num_classes: int = 1000,
        tps_type=None,
        prune_loc=None,
        keep_ratio=None,
        **kwargs,
    ):
        super().__init__()
        assert not tps_type or tps_type.lower() in ('dtps', 'etps')

        # hyperparams for compression
        self.tps_type = tps_type
        self.use_tps = True if self.tps_type else False
        if self.tps_type:
            self.prune_loc = set() if not prune_loc else set(prune_loc)
            self.keep_ratio = keep_ratio
            self.keep_ratio_list = keep_ratio if isinstance(keep_ratio, list) else [
                keep_ratio ** (k+1) for k in range(len(self.prune_loc))
            ]
            self.n_prune = len(self.prune_loc)
            if self.tps_type.lower() == 'dtps':
                # use learnable token scoring from dynamicViT
                self.score_predictor = [PredictorLG(
                    embed_dim) for _ in range(self.n_prune)]

                self.tps = TPS('dTPS')
        self.drop_rate = drop_rate
        # Patch Embedding
        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        # CLS & DST Tokens
        self.cls_token = mge.Parameter(F.zeros([1, 1, embed_dim]))
        self.dist_token = mge.Parameter(
            F.zeros([1, 1, embed_dim])) if distilled else None
        self.num_tokens = 2 if distilled else 1
        # Pos Embedding
        self.pos_embed = mge.Parameter(
            F.zeros([1, num_patches + self.num_tokens, embed_dim]))
        self.pos_drop = M.Dropout(drop_rate)
        # Blocks
        dpr = [
            x.item() for x in F.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = M.Sequential(
            *[
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_name=norm_name,
                    act_name=act_name,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm2d(norm_name, embed_dim)
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = M.Sequential(
                OrderedDict(
                    [("fc", M.Linear(embed_dim, representation_size)),
                     ("act", activation("tanh"))]
                )
            )
        else:
            self.pre_logits = None
        # Classifier head(s)
        self.head = M.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else None
        self.head_dist = None
        if distilled:
            self.head_dist = M.Linear(
                self.embed_dim, num_classes) if num_classes > 0 else None
        # Init
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

    def forward(self, x):

        x = self.patch_embed(x)
        cls_token = F.broadcast_to(
            self.cls_token, (x.shape[0], 1, self.cls_token.shape[-1]))
        if self.dist_token is None:
            x = F.concat((cls_token, x), axis=1)
        else:
            dist_token = F.broadcast_to(
                self.dist_token, (x.shape[0], 1, self.dist_token.shape[-1]))
            x = F.concat((cls_token, dist_token, x), axis=1)
        x = self.pos_drop(x + self.pos_embed)

        # vars for TPS training
        p_count = 0
        B = x.shape[0]
        init_n = x.shape[1] - 1
        hard_decision_list = []
        prev_decision = F.ones(
            (B, init_n, 1), dtype=x.dtype, device=x.device) if self.training else None
        policy = F.ones((B, init_n + 1, 1), dtype=x.dtype, device=x.device)

        for i, blk in enumerate(self.blocks):
            if not self.use_tps:
                x = blk(x)
            elif i in self.prune_loc:
                spatial_x = x[:, 1:]
                pred_score = self.score_predictor[p_count](
                    spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    # use gumbel-softmax and mask-attention with policy
                    hard_keep_decision = gumbel_softmax(pred_score, hard=True)[
                        :, :, 0:1] * prev_decision
                    # TODO: dTPS and eTPS
                    current_pruned_decision = (
                        1-hard_keep_decision) * prev_decision
                    spatial_x = self.tps(
                        spatial_x, None, hard_keep_decision, current_pruned_decision)
                    x = F.concat([x[:, :1, :], spatial_x], axis=1)
                    hard_decision_list.append(
                        hard_keep_decision.reshape(B, init_n))
                    cls_policy = F.ones(
                        (B, 1, 1), dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = F.concat([cls_policy, hard_keep_decision], axis=1)
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:, :, 0]
                    num_keep_node = int(init_n * self.keep_ratio_list[p_count])
                    sort_idxs = F.argsort(score, descending=True)
                    keep_idxs = sort_idxs[:, :num_keep_node]
                    drop_idxs = sort_idxs[:, num_keep_node:]
                    spatial_x = self.tps(batch_index_select(
                        spatial_x, keep_idxs), batch_index_select(spatial_x, drop_idxs), None, None)
                    x = F.concat([x[:, :1, :], spatial_x], axis=1)
                    x = blk(x)
                p_count += 1
            else:
                x = blk(x, policy if self.training else None)

        x = self.norm(x)
        if self.dist_token is None:
            # for now, we only support non-distilled deit
            cls_token, spatial_tokens = x[:, 0], x[:, 1:]
            logit = self.head(cls_token)
            return logit, spatial_tokens, hard_decision_list
        else:
            raise NotImplementedError

    def load_state_dict(
        self,
        state_dict: Union[dict, Callable[[str, mge.Tensor], Optional[np.ndarray]]],
        strict=False,
    ):
        for k, newv in state_dict.items():
            if k in self.state_dict():
                new_shape = newv.shape
                old_shape = self.state_dict()[k].shape
                new_psize = reduce(lambda a, b: a*b, new_shape)
                old_psize = reduce(lambda a, b: a*b, old_shape)
                if new_psize == old_psize and new_shape != old_shape:
                    state_dict[k] = newv.reshape(*old_shape)
                    print(f'reshape {k} shape to {old_shape}')
        super().load_state_dict(state_dict, strict)


@registers.models.register()
@hub.pretrained(
    os.path.join(
        PROJECT_DIR, "torch_deit_tiny_patch16_224.pkl"
    )
)
def deit_tiny_patch16_224(**kwargs):

    model = TPSViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, ffn_ratio=4,
                   qkv_bias=True, norm_name=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


@registers.models.register()
@hub.pretrained(
    os.path.join(
        PROJECT_DIR, "torch_deit_tiny_patch16_224.pkl"
    )
)
def dtps_deit_tiny_patch16_224(**kwargs):

    prune_loc = kwargs.pop('prune_loc', [3, 6, 9])
    keep_ratio = kwargs.pop('keep_ratio', 0.7)
    model = TPSViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, ffn_ratio=4, qkv_bias=True, norm_name=partial(M.LayerNorm, eps=1e-6),
                   tps_type='dtps', keep_ratio=keep_ratio, prune_loc=prune_loc, **kwargs)
    return model


@registers.models.register()
@hub.pretrained(
    os.path.join(
        PROJECT_DIR, "deit_small_patch16_224-torch.pkl"
    )
)
def deit_small_patch16_224(**kwargs):

    model = TPSViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, ffn_ratio=4,
                   qkv_bias=True, norm_name=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


@registers.models.register()
@hub.pretrained(
    os.path.join(
        PROJECT_DIR, "deit_small_patch16_224-torch.pkl"
    )
)
def dtps_deit_small_patch16_224(**kwargs):

    prune_loc = kwargs.pop('prune_loc', [3, 6, 9])
    keep_ratio = kwargs.pop('keep_ratio', 0.7)
    model = TPSViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, qkv_bias=True, norm_name=partial(M.LayerNorm, eps=1e-6),
                   tps_type='dtps', keep_ratio=keep_ratio, prune_loc=prune_loc, **kwargs)
    return model


if __name__ == "__main__":

    # model = deit_small_patch16_224(pretrained=True)
    model = deit_small_patch16_224(pretrained=True)
    inp = uniform(-1, 1, size=(2, 3, 224, 224)).astype(np.float32)
    out = model(inp)
