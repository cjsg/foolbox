import random
import numpy as np
from .improved_boundary_utils import get_label

# TODO: sample images without pytorch
# TODO: rewrite extract_img_patch function without pytorch
# TODO: check image format


class PixelSampler(object):
    def __init__(self, perturb_size=.5, *args, **xargs):
        self.eps = perturb_size

    def __call__(self, X, y, a, W, *args, **xargs):
        min_, max_ = a.bounds()

        sgn = -1 if y == a.original_class else 1
        ix = self.sample_random_ix(X)
        dX = np.zeros_like(X)
        dX[ix] = 1.

        X_ = np.clip(X + self.eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = (X_ - X) / self.eps  # accounting for the clamping
            return sgn * dX / self.eps

        X_ = np.clip(X - self.eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = - (X_ - X) / self.eps
            return -sgn * dX / self.eps

        return np.zeros_like(X)

    def sample_random_ix(self, X):
        ixs = []
        sizes = X.shape
        for size in sizes:
            ixs.append(random.randint(0, size-1))
        return tuple(ixs)


class GaussianSampler(object):
    def __init__(self, perturb_size=.5, *args, **xargs):
        self.eps = perturb_size

    def __call__(self, X, y, a, W, *args, **xargs):
        min_, max_ = a.bounds()
        X_o = a.original_image

        dX = np.random.randn(*X.shape).astype('float32')
        dX = dX / np.linalg.norm(dX)
        sgn = -1 if y == a.original_class else 1

        # if np.sign((dX * W).sum()) < 0.:
        if np.sign((dX * (X_o - X)).sum()) < 0.:
            dX = -dX

        X_ = np.clip(X + sgn * self.eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = (X_ - X) / (sgn*self.eps)
            return dX / self.eps

        X_ = np.clip(X - sgn * self.eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = - (X_ - X) / (sgn*self.eps)
            return - dX / self.eps

        return np.zeros_like(X)


class OptimalSampler(object):
    def __init__(self, *args, **xargs):
        return

    def __call__(self, X, y, a, *args, **xargs):
        y_o = a.original_class
        y_a = a.adversarial_class

        in_gradient = np.zeros(a.num_classes()).astype('float32')
        in_gradient[y_o] = 1
        in_gradient[y_a] = -1

        W_ = a.backward(in_gradient, X)  # grad(net(X)[y_o] - net(X)[y])
        return W_


# ######################################################################## #
# ######## WARNING: the following classes/functions use pytorch ########## #
# ######################################################################## #

class ImageSampler(object):
    def __init__(self,
                 perturb_size=.5,
                 crop_size=None,
                 loader=None,
                 datafolder='~/datasets/cifar10/',
                 *args, **xargs):

        import torch
        import torchvision
        import torchvision.transforms as transforms

        self.eps = perturb_size
        self.crop_size = crop_size

        if loader is None:
            transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

            trainset = torchvision.datasets.CIFAR10(
                            root=datafolder, train=True,
                            download=False, transform=transform)

            self.loader = torch.utils.data.DataLoader(
                            trainset, batch_size=1,
                            shuffle=True, num_workers=2)
        else:
            self.loader = loader

        self.dataiter = iter(self.loader)

    def __call__(self, X, y, a, W, *args, **xargs):
        min_, max_ = a.bounds()
        X_o = a.original_image

        try:
            dX, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            dX, _ = next(self.dataiter)

        dX = dX.detach().cpu().numpy().reshape(*X.shape)
        dX = dX / np.linalg.norm(dX)
        sgn = -1 if y == a.original_class else 1

        if self.crop_size is not None:
            # TODO: check here
            # note that ||dX||_2 \neq 1 anymore if cropping
            dX = extract_img_patch(dX, size=self.crop_size)

        # if np.sign((dX * W).sum()) < 0.:
        if np.sign((dX * (X_o - X)).sum()) < 0.:
            dX = -dX

        X_ = np.clip(X + sgn * self.eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = (X_ - X) / (sgn*self.eps)
            return dX / self.eps  # boundary normal direction estimate

        X_ = np.clip(X - sgn * self.eps * dX, min_, max_)
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            dX = - (X_ - X) / (sgn*self.eps)
            return - dX / self.eps  # boundary normal direction estimate

        return np.zeros_like(X)


class PlugPatchSampler(object):
    def __init__(self,
                 crop_size=5,
                 loader=None,
                 datafolder='~/datasets/cifar10/',
                 *args, **xargs):

        import torch
        import torchvision
        import torchvision.transforms as transforms
        # TODO: beware, check the image format (HWC or CHW)

        self.crop_size = crop_size

        if loader is None:
            transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

            trainset = torchvision.datasets.CIFAR10(
                            root=datafolder, train=True,
                            download=False, transform=transform)

            self.loader = torch.utils.data.DataLoader(
                            trainset, batch_size=1,
                            shuffle=True, num_workers=2)
        else:
            self.loader = loader

        self.dataiter = iter(self.loader)

    def __call__(self, X, y, a, W, *args, **xargs):
        min_, max_ = a.bounds()

        try:
            dX, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            dX, _ = next(self.dataiter)

        dX = dX.detach.cpu().numpy()
        # Note that ||dX||_2 \neq 1 anymore if cropping
        dX, mask = extract_img_patch(dX, size=self.crop_size, with_mask=True)
        X_ = X * (np.ones_like(X) - mask) + dX
        X_ = np.clip(X_, min_, max_)
        dX = X_ - X

        sgn = -1 if y == a.original_class else 1
        logits, _ = a.predictions(X_)
        y_ = get_label(logits)
        if y_ != y:
            return sgn * dX  # boundary normal direction estimate

        return np.zeros_like(X)


# TODO: check here
# TODO: check for image format (HWC or CHW)
def extract_img_patch(X, size=5, with_mask=False):
    import torch

    height = X.size(3)
    ix1 = random.randint(0, height-size)
    ixs1 = torch.arange(ix1, ix1+size)
    mask1 = torch.zeros_like(X)
    mask1[:, :, ixs1, :] = 1
    ix2 = random.randint(0, height-size)
    ixs2 = torch.arange(ix2, ix2+size)
    mask2 = torch.zeros_like(X)
    mask2[:, :, :, ixs2] = 1
    mask = mask1 * mask2
    if with_mask:
        return X * mask, mask
    else:
        return X * mask
