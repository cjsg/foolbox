import abc
import random
import sys
import numpy as np
from .improved_boundary_utils import get_label

abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})

# TODO: sample images without pytorch
# TODO: rewrite extract_img_patch function without pytorch
# TODO: check image format


class Sampler(ABC):
    def __init__(self, *args, **kwargs):
        return

    @abstractmethod
    def __call__(self):
        """Samples a new estimate of boundary weights W

        Typicially starts by sampling a perturbation X_ of current image X and
        then uses direction (X_ - X) as update for W.
        Returns the update W_ and preturbed sample X_.
        If the sampler does not need to compute a perturbed X_ to update W_
        (f.ex. OptimalSampler), then the returned X_ is a 0. array.

        """
        raise NotImplementedError

    def __repr__(self):
        string = 'Weight perturbation sampler with parameters:'
        for key, val in self.__dict__.items():
            string += '\n' + key + ': ' + str(val)
        return string


class PixelSampler(Sampler):
    def __init__(self, perturb_size=.5, *args, **xargs):
        super(PixelSampler, self).__init__()
        self.perturb_size = perturb_size

    def __call__(self, a, X, y, *args, **xargs):
        min_, max_ = a.bounds()
        eps = self.perturb_size

        sgn = -1 if y == a.original_class else 1
        ix = self.sample_random_ix(X)
        dX = np.zeros_like(X)
        dX[ix] = max_ - min_  # make perturb. size proportional to bounds

        X_ = np.clip(X + eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = (X_ - X) / eps  # accounting for the clamping
            return sgn * dX / eps, X_

        X_ = np.clip(X - eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = - (X_ - X) / eps
            return -sgn * dX / eps, X_

        return np.zeros_like(X), X_

    def sample_random_ix(self, X):
        ixs = []
        sizes = X.shape
        for size in sizes:
            ixs.append(random.randint(0, size-1))
        return tuple(ixs)


class GaussianSampler(Sampler):
    def __init__(self, perturb_size=.5, avg_pool_k=1, *args, **xargs):
        """Sample Gaussian perturbation and check if sign can be flipped

        preturb_size : float
            Perturbation size. Perturbation gets sampled from a Gaussian,
            then normalized by its l2-norm, then multiplied by the pixel
            bounds, then multiplied by perturb_size.
        avg_pool_k : int
            How many pixels to group when generating the perturbation. Defaults
            to 1. If > 1, then Gaussian perturbation is generated with
            dimensions height / avg_pool_k and width / avg_pool_k and
            supsampled to fir the original image size. Then follows the upper
            normalization.

        """

        super(GaussianSampler, self).__init__()
        self.perturb_size = perturb_size
        self.avg_pool_k = avg_pool_k

    def __call__(self, a, X, y, *args, **xargs):
        min_, max_ = a.bounds()
        eps = self.perturb_size
        X_o = a.original_image

        shape = list(X.shape)
        channel_axis = a.channel_axis(batch=False)
        assert shape[1] % self.avg_pool_k == 0

        if X.ndim == 3 and channel_axis == 0:
            shape[1] //= self.avg_pool_k
            shape[2] //= self.avg_pool_k
        elif X.ndim == 3 and channel_axis == 2:
            shape[0] //= self.avg_pool_k
            shape[1] //= self.avg_pool_k
        elif X.ndim == 4 and channel_axis == 0:
            shape[2] //= self.avg_pool_k
            shape[3] //= self.avg_pool_k
        elif X.ndim == 4 and channel_axis == 2:
            shape[1] //= self.avg_pool_k
            shape[2] //= self.avg_pool_k

        dX = np.random.randn(*shape).astype(X.dtype)
        dX = self.upsample(dX, self.avg_pool_k, channel_axis)

        dX = dX / np.linalg.norm(dX) * (max_ - min_)
        sgn = -1 if y == a.original_class else 1

        # if np.sign((dX * W).sum()) < 0.:
        if np.sign((dX * (X_o - X)).sum()) < 0.:
            dX = -dX

        X_ = np.clip(X + sgn * eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = (X_ - X) / (sgn*eps)
            return dX / eps, X_

        X_ = np.clip(X - sgn * eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = - (X_ - X) / (sgn*eps)
            return - dX / eps, X_

        return np.zeros_like(X), X_

    def upsample(self, X, k, channel_axis):
        assert channel_axis in {0, 2}, ('Channel axis must be 0 or 2')

        if X.ndim == 3 and channel_axis == 0:
            return X.repeat(k, axis=1).repeat(k, axis=2)
        elif X.ndim == 3 and channel_axis == 2:
            return X.repeat(k, axis=0).repeat(k, axis=1)
        elif X.ndim == 4 and channel_axis == 0:
            return X.repeat(k, axis=2).repeat(k, axis=3)
        elif X.ndim == 4 and channel_axis == 2:
            return X.repeat(k, axis=1).repeat(k, axis=2)


class OptimalSampler(Sampler):
    def __init__(self, *args, **xargs):
        super(OptimalSampler, self).__init__()
        return

    def __call__(self, a, X, *args, **xargs):
        y_o = a.original_class
        y_a = a.adversarial_class

        in_gradient = np.zeros(a.num_classes()).astype(X.dtype)
        in_gradient[y_o] = 1
        in_gradient[y_a] = -1

        W_ = a.backward(in_gradient, X)  # grad(net(X)[y_o] - net(X)[y])
        return W_, np.zeros_like(a.original_image)


# ######################################################################## #
# ######## WARNING: the following classes/functions use pytorch ########## #
# ######################################################################## #

class ImageSampler(Sampler):
    def __init__(self,
                 perturb_size=.5,
                 crop_size=None,
                 loader=None,
                 datafolder='~/datasets/cifar10/',
                 *args, **xargs):

        import torch
        import torchvision
        import torchvision.transforms as transforms

        super(ImageSampler, self).__init__()
        self.perturb_size = perturb_size
        self.crop_size = crop_size

        if loader is None:
            transform = transforms.ToTensor()

            trainset = torchvision.datasets.CIFAR10(
                            root=datafolder, train=True,
                            download=False, transform=transform)

            self.loader = torch.utils.data.DataLoader(
                            trainset, batch_size=1,
                            shuffle=True, num_workers=2)
        else:
            self.loader = loader

        self.dataiter = iter(self.loader)

    def __call__(self, a, X, y, *args, **xargs):
        min_, max_ = a.bounds()
        eps = self.perturb_size
        X_o = a.original_image

        try:
            dX, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            dX, _ = next(self.dataiter)

        dX = dX.detach().cpu().numpy().reshape(*X.shape)
        dX = (dX - min_) / (max_ - min_)  # normalizing data
        dX = dX / np.linalg.norm(dX) * (max_ - min_)
        sgn = -1 if y == a.original_class else 1

        if self.crop_size is not None:
            # note that ||dX||_2 \neq 1 anymore if cropping
            dX = extract_img_patch(dX, size=self.crop_size)

        # if np.sign((dX * W).sum()) < 0.:
        if np.sign((dX * (X_o - X)).sum()) < 0.:
            dX = -dX

        X_ = np.clip(X + sgn * eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = (X_ - X) / (sgn*eps)
            return dX / eps, X_

        X_ = np.clip(X - sgn * eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = - (X_ - X) / (sgn*eps)
            return - dX / eps, X_

        return np.zeros_like(X), X_


class PlugPatchSampler(Sampler):
    def __init__(self,
                 crop_size=5,
                 loader=None,
                 datafolder='~/datasets/cifar10/',
                 *args, **kwargs):

        import torch
        import torchvision
        import torchvision.transforms as transforms

        super(PlugPatchSampler, self).__init__()
        self.crop_size = crop_size

        if loader is None:
            transform = transforms.ToTensor()

            trainset = torchvision.datasets.CIFAR10(
                            root=datafolder, train=True,
                            download=False, transform=transform)

            self.loader = torch.utils.data.DataLoader(
                            trainset, batch_size=1,
                            shuffle=True, num_workers=2)
        else:
            self.loader = loader

        self.dataiter = iter(self.loader)

    def __call__(self, a, X, y, *args, **xargs):
        min_, max_ = a.bounds()

        try:
            dX, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            dX, _ = next(self.dataiter)

        dX = dX.squeeze(0).detach().cpu().numpy()
        dX = (dX - min_) / (max_ - min_)  # normalize input
        # Note that ||dX||_2 \neq 1 anymore if cropping
        dX, mask = extract_img_patch(dX, size=self.crop_size, with_mask=True)
        X_ = X * (np.ones_like(X) - mask) + dX
        X_ = np.clip(X_, min_, max_)
        dX = X_ - X

        sgn = -1 if y == a.original_class else 1
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            return sgn * dX, X_

        return np.zeros_like(X), X_


def extract_img_patch(X, size=5, with_mask=False):
    # hw = height_width
    hw = X.shape[1]  # holds with both HWC and CWH format
    ix1 = random.randint(0, hw-size)
    ixs1 = np.arange(ix1, ix1+size)
    mask1 = np.zeros_like(X)
    mask1[:, ixs1, :] = 1
    ix2 = random.randint(0, hw-size)
    ixs2 = np.arange(ix2, ix2+size)
    mask2 = np.zeros_like(X)
    mask2[:, :, ixs2] = 1
    mask = mask1 * mask2
    if with_mask:
        return X * mask, mask
    else:
        return X * mask
