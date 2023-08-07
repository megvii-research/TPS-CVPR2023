 
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


class SoftmaxXentLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        log_p = F.log_softmax(logits)
        nll = -(log_p*labels).sum(dim=-1)
        return nll.mean()


class PruneSoftTargetKDLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()
        self.keep_ratio_list = net.keep_ratio_list
        self.alpha = args.axu_w

    def forward(self, token_prune_logit_list,
                token_attn_sim_list):

        loss = 0
        n = len(token_prune_logit_list)
        for logit, sim, ratio in zip(token_prune_logit_list, token_attn_sim_list, self.keep_ratio_list):
            # k = int((1-ratio)*sim.size(1))
            # sim = sim[:, :, 1]
            # topk_value = torch.topk(
            #     sim, k=k, dim=-1)[0][:,  k-1].reshape(-1, 1)
            # keep_target = (sim <= topk_value).float().unsqueeze(-1)
            # drop_target = torch.cat([keep_target,1-keep_target], dim=-1).detach()
            # loss += -(logit*drop_target).mean()

            min_ = torch.amin(sim, dim=1, keepdim=True)
            max_ = torch.amax(sim, dim=1, keepdim=True)
            sim = (sim-min_)/(max_-min_ + 1e-6).detach()
            loss += -(logit*sim).mean()

        return self.alpha * loss.mean() / n


class PruneHardTargetKDLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()
        self.keep_ratio_list = net.keep_ratio_list
        self.alpha = args.axu_w

    def forward(self, token_prune_logit_list,
                token_attn_sim_list):

        loss = 0
        n = len(token_prune_logit_list)
        for logit, sim, ratio in zip(token_prune_logit_list, token_attn_sim_list, self.keep_ratio_list):
            # k = int((1-ratio)*sim.size(1))
            # sim = sim[:, :, 1]
            # topk_value = torch.topk(
            #     sim, k=k, dim=-1)[0][:,  k-1].reshape(-1, 1)
            # keep_target = (sim <= topk_value).float().unsqueeze(-1)
            # drop_target = torch.cat([keep_target,1-keep_target], dim=-1).detach()
            # loss += -(logit*drop_target).mean()
            loss += -(logit*sim).mean()

        return self.alpha * loss.mean() / n


class TopkmaskLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()
        self.keep_ratio_list = net.keep_ratio_list
        self.alpha = args.axu_w

    def forward(self, pred_mask_list,
                token_attn_sim_list):
        """"
        pred mask means hard keep decision mask

        """

        loss = 0
        n = len(pred_mask_list)
        for mask, sim, ratio in zip(pred_mask_list, token_attn_sim_list, self.keep_ratio_list):
            k = int((1-ratio)*sim.size(1))
            sim = sim[:, :, 1]
            topk_value = torch.topk(
                sim, k=k, dim=-1)[0][:,  k-1].reshape(-1, 1)
            target = (sim < topk_value).float()
            loss += ((target - mask)**2).mean()
        return self.alpha * loss.mean() / n


class AttnDistillKLLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()
        self.keep_ratio_list = net.keep_ratio_list
        self.attn_distill_w = args.attn_distill_w
        self.distill_w = args.distill_w
        self.mse_token_loss = args.mse_token_loss
        self.attn_distill_layers = args.attn_distill_layers

    def forward(self, pred, pred_t, spatial_features, last_decision, spatial_features_t, hard_keep_decision_list, token_attn_sim_list):
        """"
        pred mask means hard keep decision mask

        """

        loss = 0
        n = self.attn_distill_layers
        if self.attn_distill_layers == -1:
            n = len(self.keep_ratio_list)
        attn_layer_count = 0
        for mask, sim, ratio in zip(hard_keep_decision_list, token_attn_sim_list, self.keep_ratio_list):
            if attn_layer_count == n:
                break
            k = int((1-ratio)*sim.size(1))
            sim = sim[:, :, 1]
            topk_value = torch.topk(
                sim, k=k, dim=-1)[0][:,  k-1].reshape(-1, 1)
            target = (sim < topk_value).float()
            loss += ((target - mask)**2).mean()
            attn_layer_count += 1
        attn_distill_loss = self.attn_distill_w * loss.mean() / n

        cls_kl_loss = F.kl_div(
            F.log_softmax(pred, dim=-1),
            F.log_softmax(pred_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        B, N, C = spatial_features.size()
        assert last_decision.numel() == B * N

        bool_mask = last_decision.reshape(B*N) > 0.5

        spatial_features = spatial_features.reshape(B*N, C)
        spatial_features_t = spatial_features_t.reshape(B*N, C)

        if last_decision.sum() < 0.1:
            token_kl_loss = spatial_features.new(1,).fill_(0.0)
        else:
            spatial_features = spatial_features[bool_mask]
            spatial_features_t = spatial_features_t[bool_mask]
            if self.mse_token_loss:
                token_kl_loss = torch.pow(
                    spatial_features - spatial_features_t, 2).mean()
            else:
                token_kl_loss = F.kl_div(
                    F.log_softmax(spatial_features, dim=-1),
                    F.log_softmax(spatial_features_t, dim=-1),
                    reduction='batchmean',
                    log_target=True
                )

        token_distill_loss = self.distill_w * \
            cls_kl_loss + self.distill_w * token_kl_loss

        return attn_distill_loss + token_distill_loss


class DistillKLLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(args, Namespace)
        super().__init__()
        self.alpha = args.axu_w
        self.mse_token_loss = args.mse_token_loss

        self.only_distill_logits = False
        if 'dynamic' not in args.arch.lower():
            self.only_distill_logits = True

    def forward(self, pred, pred_t, spatial_features=None, last_decision=None, spatial_features_t=None):
        """"
        dynamicViT distill loss

        """
        cls_kl_loss = F.kl_div(
            F.log_softmax(pred, dim=-1),
            F.log_softmax(pred_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )
        if self.only_distill_logits:
            return self.alpha * cls_kl_loss

        B, N, C = spatial_features.size()
        assert last_decision.numel() == B * N

        bool_mask = last_decision.reshape(B*N) > 0.5

        spatial_features = spatial_features.reshape(B*N, C)
        spatial_features_t = spatial_features_t.reshape(B*N, C)

        if last_decision.sum() < 0.1:
            token_kl_loss = spatial_features.new(1,).fill_(0.0)
        else:
            spatial_features = spatial_features[bool_mask]
            spatial_features_t = spatial_features_t[bool_mask]
            if self.mse_token_loss:
                token_kl_loss = torch.pow(
                    spatial_features - spatial_features_t, 2).mean()
            else:
                token_kl_loss = F.kl_div(
                    F.log_softmax(spatial_features, dim=-1),
                    F.log_softmax(spatial_features_t, dim=-1),
                    reduction='batchmean',
                    log_target=True
                )

        return self.alpha * cls_kl_loss + self.alpha * token_kl_loss


class KeepRatioLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()
        self.keep_ratio_list = net.keep_ratio_list
        self.alpha = args.axu_w

    def forward(self, keep_pred):

        loss = 0
        for pred, target in zip(keep_pred, self.keep_ratio_list):
            pred = pred.mean(dim=1)
            loss += ((pred - target)**2).mean()
        return self.alpha * loss / len(self.keep_ratio_list)


class GlobalKeepRatioLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()

        self.keep_ratio = sum(
            net.keep_ratio_list) / len(net.keep_ratio_list)

        self.alpha = args.axu_w

    def forward(self, keep_decision_list):

        loss = 0
        avg_global_keep_ratio = (sum(keep_decision_list) /
                                 len(keep_decision_list)).mean()
        return self.alpha * (avg_global_keep_ratio - self.keep_ratio) ** 2


class GlobalKeepRatioLoss(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()

        self.keep_ratio = sum(
            net.keep_ratio_list) / len(net.keep_ratio_list)

        self.alpha = args.axu_w

    def forward(self, keep_decision_list):

        loss = 0
        avg_global_keep_ratio = (sum(keep_decision_list) /
                                 len(keep_decision_list)).mean()
        return self.alpha * (avg_global_keep_ratio - self.keep_ratio) ** 2


class TokenActLayerRatio(nn.Module):

    def __init__(self, net, args):
        assert isinstance(net, nn.Module) and isinstance(args, Namespace)
        super().__init__()

        self.layer_ratio = sum(
            net.keep_ratio_list) / len(net.keep_ratio_list)

        self.alpha = args.axu_w

    def forward(self, keep_decision_list):

        loss = 0
        layer_keep_ratio = torch.cat([r.unsqueeze(-1)
                                      for r in keep_decision_list], dim=-1)
        layer_keep_ratio = layer_keep_ratio.mean(dim=-1)
        loss = (layer_keep_ratio-self.layer_ratio) ** 2
        return self.alpha * loss.mean()


def get_loss_cls(identifier):
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret name : {str(identifier)}")
