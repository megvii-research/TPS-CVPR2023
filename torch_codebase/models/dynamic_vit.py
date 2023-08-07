# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
import math
from sklearn.utils import shuffle
import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.utils.checkpoint import checkpoint

from .functions import gumbel_softmax


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        # * policy.reshape(B, 1, N, 1)
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype,
                        device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn if return_attn else None


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None, return_attn=False):
        x_ori = x
        x, attn = self.attn(self.norm1(x), policy=policy,
                            return_attn=return_attn)
        x = x_ori + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, policy=None, return_attn=False):
        x, attn = self.attn(self.norm1(x), policy=policy,
                            return_attn=return_attn)
        x = x + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn


class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),

            # NOTE: the head should return the logits for gumbel softmax
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :C//2]
        global_x = (x[:, :, C//2:] * policy).sum(dim=1,
                                                 keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def get_sim(x, y, eps=1e-6, mask_eye=-100, l2_norm=True):

    if y is None:
        y = x
    if l2_norm:
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        y = y / (y.norm(dim=-1, keepdim=True) + eps)

    sim = torch.bmm(x, y.permute(0, 2, 1))
    if mask_eye is not None:
        sim.masked_fill_(
            torch.eye(x.size(1), device=x.device).unsqueeze(0).bool(), mask_eye)
    return sim


class TPSModule(nn.Module):

    # from pruned tokens to keep tokens
    def __init__(self, l2_norm=True, temperature=1) -> None:
        super().__init__()
        self.l2_norm = l2_norm
        self.temperature = temperature

    def forward(self, x, y, current_keep_decision, current_pruned_decision, relative_dist=None):
        B, N, C = x.size(0), x.size(1), x.size(2)
        if self.training:

            cos_sim = get_sim(
                x, None, mask_eye=-100, l2_norm=self.l2_norm)

            cos_sim = cos_sim/self.temperature
            cos_sim = cos_sim.masked_fill(
                ~current_keep_decision.bool().reshape(B, 1, N), -100)

            sim_th = cos_sim.amax(
                dim=2, keepdims=True)

            # N, pruned token dim, keep token dim
            mask = (cos_sim == sim_th).float() * current_pruned_decision
            cos_sim = (mask * cos_sim)
            # N,keep token dim, pruned_token dim
            mask = mask.permute(0, 2, 1)
            cos_sim = cos_sim.permute(0, 2, 1)
            numerator = torch.exp(cos_sim) * mask
            denominator = math.e + numerator.sum(dim=-1, keepdims=True)
            x = x * (math.e / denominator) + \
                torch.bmm(numerator / denominator, x)

        else:

            # given k =  prune num
            cos_sim = get_sim(
                y, x, mask_eye=None, l2_norm=self.l2_norm)
            cos_sim = cos_sim/self.temperature
            sim_th = cos_sim.amax(dim=2, keepdims=True)
            mask = (cos_sim == sim_th).float()
            # N, pruned token dim, keep token dim
            cos_sim = mask * cos_sim
            # N,keep token dim, pruned_token dim
            mask = mask.permute(0, 2, 1)
            cos_sim = cos_sim.permute(0, 2, 1)
            numerator = torch.exp(cos_sim) * mask
            denominator = math.e + numerator.sum(dim=-1, keepdims=True)
            x = x * (math.e / denominator) + \
                torch.bmm(numerator / denominator, y)

        return x


def get_module_cls(identifier):
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret name : {str(identifier)}")


class DynamicVit(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp,
                 dpr_constant=True, init_scale=1e-4,
                 no_embed_class=True,
                 # pruned hps
                 prune_loc=[3, 6, 9], keep_ratio=0.7, momentum=0.9, prune_th_init=1.0,
                 token_merge_module=None,
                 grad_checkpoint_layers=[0, 1, 2],
                 token_score_head_cls="PredictorLG",
                 reuse_last_attn=False,
                 shuffle_score=False,  # for random rebust exps
                 ):
        super().__init__()

        # NOTE: for grad checkpointing to save GPU memory
        self.grad_checkpointing = False
        self.grad_checkpoint_layers = grad_checkpoint_layers
        self.reuse_last_attn = False
        self.reuse_attn_layers = [] if not self.reuse_last_attn else [l-1 for l in prune_loc]
        self.shuffle_score = shuffle_score

        print(f"The Token Scoring Head Module: {token_score_head_cls}")
        token_score_head_cls = get_module_cls(token_score_head_cls)

        # NOTE: code for pruing
        self.token_merge_module = token_merge_module if token_merge_module is not None else None
        self.prune_loc = set() if not prune_loc else set(prune_loc)
        # the keep ratio stands for the token keeping ratio among the current layer tokens
        self.keep_ratio_list = keep_ratio if isinstance(keep_ratio, list) else [
            keep_ratio ** (k+1) for k in range(len(self.prune_loc))
        ]
        self.momentum = momentum
        predictor_list = [token_score_head_cls(embed_dim)
                          for _ in range(len(prune_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)

        self.dropout_rate = drop_rate

        self.no_embed_class = no_embed_class
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        embed_cls_token = 0 if self.no_embed_class else 1
        embed_len = num_patches + embed_cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [
            dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(
                    x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(
                    x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed

        k = 1
        p_count = 0
        hard_decision_list = []
        init_n = x.size(1) - k
        prev_decision = torch.ones(
            B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n + k, 1, dtype=x.dtype, device=x.device)
        token_prune_logit_list = []

        last_attn = None
        for i, blk in enumerate(self.blocks):
            return_attn = i in self.reuse_attn_layers
            if i in self.prune_loc:
                spatial_x = x[:, k:]
                pred_score = self.score_predictor[p_count](
                    spatial_x, prev_decision).reshape(B, -1, 2)
                if self.shuffle_score:
                    pred_score = random_shuffle_dim1(pred_score)
                # TODO: maintain a EMA topk thresold
                if self.training:
                    hard_keep_decision = gumbel_softmax(
                        pred_score, hard=True)[:, :, 0:1] * prev_decision

                    if self.token_merge_module is not None:
                        current_pruned_decision = (
                            1-hard_keep_decision) * prev_decision
                        spatial_x = self.token_merge_module(
                            spatial_x, None, hard_keep_decision, current_pruned_decision, None)
                        x = torch.cat([x[:, :1, :], spatial_x], dim=1)

                    hard_decision_list.append(
                        hard_keep_decision.reshape(B, init_n))
                    cls_policy = torch.ones(
                        B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([cls_policy, hard_keep_decision], dim=1)

                    if self.grad_checkpointing and not torch.jit.is_scripting() and i in self.grad_checkpoint_layers:
                        x, last_attn = checkpoint(blk, x, policy, return_attn)
                    else:
                        x, last_attn = blk(
                            x, policy=policy, return_attn=return_attn)
                    prev_decision = hard_keep_decision

                    token_prune_logit_list.append(pred_score)

                else:
                    score = pred_score[:, :, 0]
                    num_keep_node = int(init_n * self.keep_ratio_list[p_count])

                    sort_idxs = torch.argsort(score, dim=1, descending=True)
                    keep_policy = sort_idxs[:, :num_keep_node]

                    cls_policy = torch.zeros(
                        B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat(
                        [cls_policy, keep_policy + 1], dim=1)

                    if self.token_merge_module is not None:
                        drop_policy = sort_idxs[:, num_keep_node:]

                        spatial_x = self.token_merge_module(
                            batch_index_select(spatial_x, keep_policy), batch_index_select(spatial_x, drop_policy), None, None, None)
                        x = torch.cat([x[:, :1, :], spatial_x], dim=1)

                    else:
                        x = batch_index_select(x, now_policy)

                    # only change token size
                    prev_decision = batch_index_select(
                        prev_decision, keep_policy)
                    x, last_attn = blk(x, return_attn=return_attn)
                p_count += 1
            else:
                if self.training:
                    if self.grad_checkpointing and not torch.jit.is_scripting() and i in self.grad_checkpoint_layers:
                        x, _ = checkpoint(blk, x, policy, return_attn)
                    else:
                        x, _ = blk(x, policy, return_attn=return_attn)

                else:
                    x, _ = blk(x, return_attn=return_attn)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], prev_decision.detach(), hard_decision_list, token_prune_logit_list

    def forward(self, x):

        x, spatial_features, last_decision, hard_decision_list, token_prune_logit_list = self.forward_features(
            x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate),
                          training=self.training)
        x = self.head(x)

        return x, spatial_features, last_decision, hard_decision_list, token_prune_logit_list


"""
 dynamic DeiT without distillation
"""


@register_model
def dynamic_deit_tiny_patch16_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    model = DynamicVit(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )

        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def dynamic_deit_tiny_patch16_224_token_merge_softmax_reweight(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    dist_decay_alpha = kwargs.pop(
        'dist_decay_alpha') if "dist_decay_alpha" in kwargs else 0
    temperature = kwargs.pop(
        'temperature') if "temperature" in kwargs else 1
    model = DynamicVit(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False,
        token_merge_module=TPSModule(
            temperature=temperature),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def dynamic_deit_small_patch16_224(pretrained=False, **kwargs):
    print(kwargs)
    kwargs.pop('pretrained_cfg')
    model = DynamicVit(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def dynamic_deit_small_patch16_224_token_merge_softmax_reweight(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')

    temperature = kwargs.pop(
        'temperature') if "temperature" in kwargs else 1
    model = DynamicVit(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False, token_merge_module=TPSModule(temperature=temperature),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )

        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def dynamic_deit_base_patch16_224(pretrained=False, **kwargs):
    print(kwargs)
    kwargs.pop('pretrained_cfg')
    model = DynamicVit(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def dynamic_deit_base_patch16_224_token_merge_softmax_reweight(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    temperature = kwargs.pop(
        'temperature') if "temperature" in kwargs else 1
    model = DynamicVit(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False, token_merge_module=TPSModule(temperature=temperature),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )

        model.load_state_dict(checkpoint["model"], strict=False)
    return model
