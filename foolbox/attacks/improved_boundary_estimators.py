import random
import torch
from .improved_boundary_utils import get_label

# TODO: rewrite these functions without pytorch


class PixelWeightEstimator(object):
    def __init__(self, perturbation_size=.5, min_pix=-2.5, max_pix=2.5):
        self.eps = perturbation_size
        self.min_pix = min_pix
        self.max_pix = max_pix

    def __call__(self, X, y, a, W, *args, **xargs):
        net.eval()

        sgn = -1 if y == a.original_class else 1
        ix = self.sample_random_ix(X)
        dX = np.zeros_like(X)
        dX[ix] = 1.

        X_ = np.clip(X + self.eps * dX, self.min_pix, self.max_pix)
        # y_, _ = get_label(X_, net)
        y_ = a.predictions(X_)
        if y_ != y:
            dX = (X_ - X) / self.eps  # accounting for the clamping
            return sgn * dX / self.eps, X_

        X_ = np.clip(X - self.eps * dX, self.min_pix, self.max_pix)
        # y_, _ = get_label(X_, net)
        y_ = a.predictions(X_)
        if y_ != y:
            dX = - (X_ - X) / self.eps
            return -sgn * dX / self.eps, X_

        return np.zeros_like(X), X_

    def sample_random_ix(self, X):
        ixs = []
        sizes = X.shape
        for size in sizes:
            ixs.append(random.randint(0, size-1))
        return tuple(ixs)


class GaussianWeightEstimator(object):
    def __init__(self,
                 perturbation_size=.5,
                 min_pix=-2.5,
                 max_pix=2.5,
                 *args, **xargs):

        self.eps = perturbation_size
        self.min_pix = min_pix
        self.max_pix = max_pix

    def __call__(self, X, y, a, W, X_t, *args, **xargs):
        net.eval()

        dX = torch.randn_like(X)
        dX = dX / dX.norm(p=2).item()
        sgn = -1 if y == a.original_class else 1

        # if torch.sign((dX * W).sum()) < 0.:
        if torch.sign((dX * (X_t - X)).sum()) < 0.:
            dX = -dX

        X_ = np.clip(X + sgn * self.eps * dX, self.min_pix, self.max_pix)
        # y_, _ = get_label(X_, net)
        y_ = a.predictions(X_)
        if y_ != y:
            dX = (X_ - X) / (sgn*self.eps)
            return dX / self.eps, X_  # boundary normal direction estimate

        X_ = np.clip(X - sgn * self.eps * dX, self.min_pix, self.max_pix)
        # y_, _ = get_label(X_, net)
        y_ = a.predictions(X_)
        if y_ != y:
            dX = - (X_ - X) / (sgn*self.eps)
            return - dX / self.eps, X_  # boundary normal direction estimate

        return np.zeros_like(X), X_


class ImageWeightEstimator(object):
    def __init__(self,
                 perturbation_size=.5,
                 min_pix=-2.5,
                 max_pix=2.5,
                 crop_size=None,
                 loader=None,
                 datafolder='~/datasets/cifar10/',
                 *args, **xargs):

        import torch
        import torchvision
        import torchvision.transforms as transforms

        self.eps = perturbation_size
        self.min_pix = min_pix
        self.max_pix = max_pix
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

    def __call__(self, X, y, a, W, X_t, *args, **xargs):
        import torch

        try:
            dX, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            dX, _ = next(self.dataiter)

        dX = (dX / dX.norm(p=2).item()).detach().cpu().numpy()
        sgn = -1 if y == a.original_class else 1

        if self.crop_size is not None:
            # note that ||dX||_2 \neq 1 anymore if cropping
            dX = extract_img_patch(dX, size=self.crop_size)

        # if np.sign((dX * W).sum()) < 0.:
        if np.sign((dX * (X_t - X)).sum()) < 0.:
            dX = -dX

        X_ = (X + sgn * self.eps * dX).clamp(self.min_pix, self.max_pix)
        y_, _ = get_label(X_, net)
        if y_ != y:
            dX = (X_ - X) / (sgn*self.eps)
            return dX / self.eps, X_  # boundary normal direction estimate

        X_ = (X - sgn * self.eps * dX).clamp(self.min_pix, self.max_pix)
        y_, _ = get_label(X_, net)
        if y_ != y:
            dX = - (X_ - X) / (sgn*self.eps)
            return - dX / self.eps, X_  # boundary normal direction estimate

        return torch.zeros_like(X), X_


class PlugPatchWeightEstimator(object):
    def __init__(self,
                 min_pix=-2.5,
                 max_pix=2.5,
                 crop_size=5,
                 loader=None,
                 datafolder='~/datasets/cifar10/',
                 *args, **xargs):

        import torchvision
        import torchvision.transforms as transforms

        self.min_pix = min_pix
        self.max_pix = max_pix
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

    def __call__(self, X, y, y_t, net, W, X_t, *args, **xargs):
        net.eval()

        try:
            dX, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            dX, _ = next(self.dataiter)

        # Note that ||dX||_2 \neq 1 anymore if cropping
        dX, mask = extract_img_patch(dX, size=self.crop_size, with_mask=True)
        X_ = X * (torch.ones_like(X) - mask) + dX
        X_ = torch.clamp(X_, self.min_pix, self.max_pix)
        dX = X_ - X

        sgn = -1 if y == y_t else 1
        y_, _ = get_label(X_, net)
        if y_ != y:
            return sgn * dX, X_  # boundary normal direction estimate

        return torch.zeros_like(X), X_


class OptimalWeightEstimator(object):
    def __init__(self, *args, **xargs):
        return

    def __call__(self, X, y, y_t, net, *args, **xargs):
        from torch.autograd import grad

        X.requires_grad = True
        out = net(X)
        get_label.count += 1

        if y_t != y:
            f = out[0, y_t] - out[0, y]
        else:
            _, ixs = torch.topk(out, 2)
            f = out[0, y_t] - out[0, ixs[0, 1]]

        W_ = grad(f, X)[0]
        X.requires_grad = False
        return W_, torch.zeros_like(X)


def extract_img_patch(X, size=5, with_mask=False):
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
