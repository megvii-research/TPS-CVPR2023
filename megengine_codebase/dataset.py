import os
import numpy as np
import math
import cv2
import torch
import megengine.data as data
import megengine.data.transform as T
from megengine.data.transform import VisionTransform, Compose
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.nn import Module as TModule
from megengine.data.sampler import MapSampler, Sampler
import megengine.distributed as dist
import megengine.functional as F
from PIL import Image

cv2.setNumThreads(1)
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['VECLIB_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
torch.set_num_threads(1)


class RASampler(MapSampler):

    def __init__(
        self,
        dataset,
        batch_size=1,
        drop_last=False,
        world_size=None,
        rank=None,
        seed=None,
        num_repeats=3,
    ):
        super().__init__(dataset, batch_size, drop_last, None, world_size, rank, seed)

        self.num_repeats = num_repeats
        self.num_samples = int(
            math.ceil(len(self.dataset) * self.num_repeats / self.world_size))
        self.total_size = self.num_samples * self.world_size

        self.num_selected_samples = int(math.floor(
            len(self.dataset) // 256 * 256 / self.world_size))

    def sample(self):
        r"""Return a list contains all sample indices."""
        indices = self.rng.permutation(len(self.dataset))
        return indices

    def batch(self):
        indices = self.sample()
        indices = np.repeat(indices, self.num_repeats, axis=0).tolist()

        total_size = len(indices)
        if self.world_size > 1:
            indices = indices[self.rank: total_size: self.world_size]
        indices = indices[:self.num_selected_samples]

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch


class BGR2RGB(VisionTransform):

    def __init__(self,):
        super().__init__()

    def _apply_image(self, image):
        return image[:, :, ::-1]


class ToNumpyArr(VisionTransform):

    def __init__(self,):
        super().__init__()

    def _apply_image(self, image):
        return image.numpy()


class ValScale(VisionTransform):

    def __init__(self, div=255):
        super().__init__()
        self.div = div

    def _apply_image(self, image):
        return image.astype('float32')/255


class ToPILImage(VisionTransform):

    def __init__(self,):
        super().__init__()

    def _apply_image(self, ndarr):
        return Image.fromarray(ndarr)


def torch2mge_transform(t: TModule):

    if not isinstance(t, VisionTransform):
        t.order = ("image",)
        if hasattr(t, 'forward'):
            setattr(t, '_apply_image', t.forward.__get__(t, t.__class__))
        elif hasattr(t, '__call__'):
            setattr(t, '_apply_image', t.__call__.__get__(t, t.__class__))
        else:
            raise TypeError
        setattr(t, '_get_apply', VisionTransform._get_apply.__get__(t, t.__class__))
        setattr(t, 'apply', VisionTransform.apply.__get__(t, t.__class__))
        setattr(t, 'apply_batch',
                VisionTransform.apply_batch.__get__(t, t.__class__))
    else:
        pass


def get_dataset(data_name, data_dir):
    train_dataset, test_dataset, num_classes = None, None, 100

    if data_name.lower() == 'imagenet':
        train_dataset = data.dataset.ImageNet(data_dir, train=True)
        test_dataset = data.dataset.ImageNet(data_dir, train=False)
        num_classes = 1000
    else:
        raise NotImplementedError
    return train_dataset, test_dataset, num_classes


def build_transform(is_train, args):
    # the transform on Imagenet1K is borrowed from https://github.com/facebookresearch/deit
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        t = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        for trans in t.transforms:
            torch2mge_transform(trans)
        t.transforms.insert(0, BGR2RGB())
        t.transforms.insert(1, ToPILImage())
        t.transforms.append(ToNumpyArr())
        t = Compose(t.transforms)
        return t

    else:
        t = []
        size = int((256 / 224) * args.input_size)
        t.append(BGR2RGB())
        t.append(
            # to maintain same ratio w.r.t. 224 images
            T.Resize(size, interpolation=3),
        )
        t.append(T.CenterCrop(args.input_size))
        t.append(ValScale(255))
        t.append(T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        t.append(T.ToMode("CHW"))
    return T.Compose(t)


def load_dataset(name, data_dir, batch_size, workers, args):
    print('load dataset:', name)
    train_dataset, test_dataset, num_classes = get_dataset(name, data_dir)
    if name.lower() == 'imagenet':
        train_transform = build_transform(True, args)
        test_transform = build_transform(False, args)
    else:
        raise NotImplementedError('Not Defined this Dataset')

    if not args.repeated_aug:
        train_sampler = data.RandomSampler(
            train_dataset, batch_size=batch_size, drop_last=True)
    else:
        train_sampler = RASampler(
            train_dataset, batch_size=batch_size, drop_last=True)
    train_sampler = data.Infinite(train_sampler)
    test_sampler = data.SequentialSampler(
        test_dataset, batch_size=batch_size//2)
    train_dataloader = data.DataLoader(
        train_dataset, train_sampler, train_transform, num_workers=workers, preload=True)
    test_dataloader = data.DataLoader(
        test_dataset, test_sampler, test_transform, num_workers=workers, preload=True)
    return train_dataloader, test_dataloader, num_classes, len(train_dataset)


if __name__ == "__main__":

    train_dataset, test_dataset, num_classes = get_dataset(
        'imagenet', '/data/datasets/imagenet1k')
