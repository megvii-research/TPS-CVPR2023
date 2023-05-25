import torch
import megengine.functional as F
import megengine as mge
def batch_index_select(x, idx):
    B, C = x.shape[0], x.shape[2]
    N_new = idx.shape[1]
    idx = F.broadcast_to(idx.reshape(B, N_new, 1), (B, N_new, C))
    return F.gather(x, 1, idx)

def mge_batch_index_select(x,idx):
    B, N, C = x.shape[0],x.shape[1],x.shape[2]
    N_new = idx.shape[1]
    offset = F.arange(0,B,1, dtype=np.int32).reshape(B, 1) * N
    idx = idx + offset
    out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
    return out


def torch_batch_index_select(x, idx):
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


import numpy as np
x = np.random.randn(128,196,384).astype('float32')
idx = np.random.randint(0,196,(128,196//2)).astype('int32')
print(idx.max())

torch_y = torch_batch_index_select(torch.Tensor(x),torch.Tensor(idx).long())
mge_y = batch_index_select(mge.Tensor(x),mge.Tensor(idx))

diff = np.abs(torch_y.numpy() - mge_y.numpy()).mean()
print(diff)