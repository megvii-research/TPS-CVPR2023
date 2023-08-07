import inspect
import torch


def get_fn_name():

    return inspect.currentframe().f_code.co_name


def random_shuffle_dim1(inp):

    batch = inp.shape[0]
    dim_size = inp.shape[1]
    idx = torch.rand(batch, dim_size).argsort(
        dim=1).unsqueeze(-1).expand_as(inp).to(device=inp.device)
    return torch.gather(inp, 1, idx)
