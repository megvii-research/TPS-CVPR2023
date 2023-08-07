from torch.utils.data._utils.collate import default_collate
import torch
from typing import Tuple
import numpy as np
import torch.nn.functional as F


def partial_mixup(input: torch.Tensor,
                  alphas,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input.index_select(0, indices)
    return input * alphas + perm_input * (1-alphas)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          alphas,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(
        0), device=input.device, dtype=torch.long)

    return partial_mixup(input, alphas.view(-1, 1, 1, 1), indices), partial_mixup(target, alphas.view(-1, 1), indices)


class Mixup():

    def __init__(self, alpha=0.2, seed=123):

        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(self, input, target):
        alphas = torch.from_numpy(self.rng.beta(
            self.alpha, self.alpha, size=input.size(0))).float()
        alphas = alphas.to(input.device)
        return mixup(input, target, alphas)


class MixupCollater():

    def __init__(self, alpha=0.2, ncls=1000, seed=123):

        self.mixup_caller = Mixup(alpha, seed)
        self.ncls = ncls

    def __call__(self, data):

        input, target = default_collate(data)

        target = F.one_hot(target, self.ncls).float()
        return self.mixup_caller(input, target)


if __name__ == "__main__":

    a, b = torch.randn(8, 1, 4, 4), torch.randn(8, 10)
    c, d = Mixup()(a, b)
    print(c.shape, d.shape, c.dtype, d.dtype)
