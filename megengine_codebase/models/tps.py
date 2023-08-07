
import megengine.functional as F
import math
import megengine.module as M
import megengine as mge
import numpy as np


def cal_cosine_similarity(x, y,eps = 1e-6, mask_eye=-100):
    """
    cacluating the cosine similarity among tokens of each sample in the batch

    Args:
        x (mge.Tensor): tokens, (batch,n_token,channels)
        y (mge.Tensor): tokens, (batch,n_token,channels)
        mask_eye (int, optional): the value used to fill the self-interaction. Defaults to -100.

    Returns:
        sim: the cosine similarity used as the cost matrix
    """

    x = x/(F.norm(x,2,axis = -1,keepdims = True) + eps)
    if y is None:
        y = x
    else:
        y = y/(F.norm(y,2,axis = -1,keepdims = True) + eps)
    
    sim = F.matmul(x, y.transpose(0, 2, 1))
    if mask_eye is not None:

        # sim.masked_fill_(
        #     mge.eye(x.shape[1], device=x.device).unsqueeze(0).bool(), mask_eye)
        mask = F.eye(x.shape[1], device=x.device)[None].astype('bool')
        mask = F.broadcast_to(mask, sim.shape)
        sim[mask] = mask_eye
    return sim


class TPS(M.Module):

    def __init__(self, variant='dTPS'):
        """
            TPS: joint Token Pruning & Squeezing
            Here, the TPS module contains the operation of squeezing only.
            You can follow pruning operations of dynamicViT, EViT or others.
        Args:
            variant (str, optional): the variant type, the inter-block variant dTPS and intra-block variant eTPS. Defaults to 'dTPS'.
        """
        super().__init__()
        assert variant in ('dTPS', 'eTPS')
        self.variant = variant

    def forward(self, reserved, pruned, now_reserved_policy, now_pruned_policy):
        """

        The reserved tokens and pruned tokens are split from the input token of current stage based on the token scoring result.

        Args:
            reserved (mge.Tensor): the reserved tokens, (batch,n_token,channels)
            pruned (mge.Tensor): the reserved tokens, (batch,n_token,channels)
            now_reserved_policy (mge.Tensor): the policy which denotes the token to reserve in the current stage, it's only needed under training stage of dTPS
            now_pruned_policy (mge.Tensor): the policy which denotes the token to prune in the current stage, it's only needed under training stage of dTPS

        Returns:
            updated_reserved: the updated reserved tokens
        """
        B, N, _ = reserved.shape[0], reserved.shape[1], reserved.shape[2]
        if self.variant == 'dTPS' and self.training:

            # during training, the tokens maintain a fixed shape
            # following dynamicViT, the pruned tokens' interaction with the class token will be removed in the multi-head attention layer
            cost_matrix = cal_cosine_similarity(
                reserved, None, mask_eye=-100)
            cost_matrix[F.broadcast_to(~now_reserved_policy.astype(
                'bool').reshape(B, 1, N), cost_matrix.shape)] = -100
            # the mask only keeps the interactions between pruned tokens and nearest reserved tokens in the current stage
            sim_th = cost_matrix.max(axis=2, keepdims=True)
            mask = (cost_matrix == sim_th).astype(
                'float32') * now_pruned_policy
            cost_matrix = (mask * cost_matrix)

            # transpose the dimension for batch matrix-multiplying
            mask = mask.transpose(0, 2, 1)
            cost_matrix = cost_matrix.transpose(0, 2, 1)
            numerator = F.exp(cost_matrix) * mask
            denominator = math.e + numerator.sum(axis=-1, keepdims=True)
            # fuse the host tokens with all matched pruned tokens
            reserved = reserved * (math.e / denominator) + \
                F.matmul(numerator / denominator, reserved)

        else:

            # during inference or training & infernce of the eTPS,
            # the pruned tokens and reserved tokens are splitted from the input tokens
            # and the pruned subset will be aggreagted into the matched reserved tokens dubbed as host tokens
            cost_matrix = cal_cosine_similarity(
                pruned, reserved, mask_eye=None)
            sim_th = cost_matrix.max(axis=2, keepdims=True)
            mask = (cost_matrix == sim_th).astype('float32')
            cost_matrix = mask * cost_matrix
            mask = mask.transpose(0, 2, 1)
            cost_matrix = cost_matrix.transpose(0, 2, 1)
            numerator = F.exp(cost_matrix) * mask
            denominator = math.e + numerator.sum(axis=-1, keepdims=True)
            reserved = reserved * (math.e / denominator) + \
                F.matmul(numerator / denominator, pruned)

        return reserved


if __name__ == "__main__":

    mge.config.async_level = 0
    B, N, C = 16, 196, 32

    reserved = np.random.randn(B, N, C) * 1
    pruned = np.random.randn(B, N, C)*1

    score = np.random.randn(B, N, 1)
    now_reserved_policy = score > 0.
    now_pruned_policy = score > 0.5
    tps_m = TPS()

    out = tps_m(mge.Tensor(reserved), mge.Tensor(pruned),
                mge.Tensor(now_reserved_policy), mge.Tensor(now_pruned_policy))
