import megengine as mge
import megengine.module as M
import megengine.functional as F
from argparse import Namespace
import numpy as np
from basecls.utils import  registers
# NOTE: 
# The loss functions follow the implements in dynamicVit [https://github.com/raoyongming/DynamicViT]
# and EViT for a fair comparison,
# DynamicViT employs a keep_ratio_loss to help model training adapt the given token keep ratio
# and the distillation loss helps the model learn the teacher model from both token and logit levels.
# Notice that the EViT only employs the distillation loss of logit.

def kl_div(log_pred,log_target):

    loss = F.exp(log_target) *(log_target -log_pred)
    b = loss.shape[0]
    return loss.sum()/b

@registers.models.register()
class SoftTargetCrossEntropy(M.Module):

    def __init__(self, net,args):
        super().__init__()

    def forward(self, x, target):
        loss = F.sum(-target * F.logsoftmax(x, axis=-1), axis=-1)
        return loss.mean()

@registers.models.register()
class PruningLoss(M.Module):

    def __init__(self, net, args):
        assert isinstance(net, M.Module) and isinstance(args, Namespace)
        super().__init__()
        self.keep_ratio_list = net.keep_ratio_list

    def forward(self, decision_list):

        loss = 0
        for pred, target in zip(decision_list, self.keep_ratio_list):
            pred = pred.mean(axis=1)
            loss += ((pred - target)**2).mean()
        return loss / len(self.keep_ratio_list)


@registers.models.register()
class DistillKLLoss(M.Module):

    def __init__(self, net, args):
        assert isinstance(net, M.Module) and isinstance(args, Namespace)
        super().__init__()

        self.only_distill_logits = getattr(args,"only_distill_logits",False)
        self.mse_token_loss = getattr(args,"mse_token_loss",False)

    def forward(self, pred, pred_t, spatial_features=None, spatial_features_t=None,last_decision=None):

        cls_kl_loss = kl_div(
            F.logsoftmax(pred, axis=-1),
            F.logsoftmax(pred_t, axis=-1),
        )
        if self.only_distill_logits:
            return  cls_kl_loss

        B, N, C = tuple(spatial_features.shape)
        last_decision = last_decision.detach()
        bool_mask = last_decision.reshape(B*N) > 0.5

        spatial_features = spatial_features.reshape(B*N, C)
        spatial_features_t = spatial_features_t.reshape(B*N, C)


        spatial_features = spatial_features[bool_mask]
        spatial_features_t = spatial_features_t[bool_mask]
        if self.mse_token_loss:
            token_kl_loss = F.pow(
                spatial_features - spatial_features_t, 2).mean()
        else:
            token_kl_loss = kl_div(
                F.logsoftmax(spatial_features, axis=-1),
                F.logsoftmax(spatial_features_t, axis=-1),
            )
        return cls_kl_loss + token_kl_loss


if __name__ == "__main__":
    
    from tps_deit import dtps_deit_small_patch16_224
    from megengine.random import uniform
    model = dtps_deit_small_patch16_224(pretrained=True)
    inp = uniform(-1,1,size =(2,3,224,224)).astype(np.float32)
    cls_token, spatial_tokens, hard_decision_list = model(inp)

    for o in [cls_token, spatial_tokens ]:
        print(o.shape)
    cls_token_t  = uniform(-1,1,size =(2,384)).astype(np.float32)
    token_t  = uniform(-1,1,size =(2,196,384)).astype(np.float32)
    last_decision = hard_decision_list[-1]
    args = Namespace(axu_w = 2)

    loss_fn = DistillKLLoss(model,args,mse_token_loss = True)
    
    print(
     loss_fn(cls_token,cls_token_t,spatial_tokens,token_t,last_decision)   
    )