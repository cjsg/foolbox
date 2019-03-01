import numpy as np
from .improved_boundary_smoothers import DoubleExponentialSmoother

# only for GPstepSizeHandler
import torch
from collections import deque


# TODO: Implement constant step-size handler


class RLStepSizeHandler(object):
    """An Reinforcement Learning type of step-size updater.

    Each step-size sz gets sampled from a Gaussian with mean sz_mean and
    standard deviation sz_mean / 5. The mean then gets updated by a small
    positive or negative amount in the direction (sz - sz_mean), depending on
    whether the delta-loss (i.e. loss - previous loss) increased or decreased.
    Said differently, sz_mean gets updated in the direction:
    (sz - sz_mean) * sign(second derivative of the loss)

    """
    def __init__(self, initial_sz, eps=.025, lam=.05, use_sign=True):
        self.sz = initial_sz  # step_size
        self.sz_mean = initial_sz
        self.sz_mean_std_ratio = 5.
        self.loss_0 = self.loss_1 = self.loss_2 = None
        self.non_adv_rate = .5  # probability not to be adversarial
        self.eps = eps  # step-size of sz_mean-update
        self.lam = lam  # update strength of not_adv
        self.use_sign = use_sign  # whether to use sign or relative improvement

    def update(self, dist, non_adv):
        self.non_adv_rate = (
            (1-self.lam)*self.non_adv_rate
            + self.lam*int(non_adv))  # update is_adv_rate

        # loss = dist*(1 + max(self.non_adv_rate - .5, 0))
        # loss = dist*(1 + .5*(self.non_adv_rate - .5))
        loss = dist

        return self._update(loss)

    def _update(self, loss):
        self.loss_0 = self.loss_1
        self.loss_1 = self.loss_2
        self.loss_2 = loss

        # check if all losses were initialized
        if None not in {self.loss_0, self.loss_1, self.loss_2}:
            dloss_1 = self.loss_0 - self.loss_1
            dloss_2 = self.loss_1 - self.loss_2

            if self.use_sign:
                sz_update = np.sign(
                    (self.sz - self.sz_mean) * (dloss_2 - dloss_1))
            else:
                sz_update = ((self.sz - self.sz_mean) /
                             (self.sz_std) *
                             (dloss_2 - dloss_1))

            # gradient-descent like update:
            self.sz_mean = (1 + self.eps * sz_update) * self.sz_mean

        sz_std = self.sz_mean / self.sz_mean_std_ratio
        self.sz = np.random.normal(loc=self.sz_mean, scale=sz_std)
        return self.sz


class BallesStepSizeHandler(object):
    """Use a step-size update inspired from equation (22) in [1]_

    .. _[1]: Lukas Balles, Javier Romero, Philipp Hennig, "Coupling Adaptive
             Batch Sizes with Learning Rates",
             https://arxiv.org/abs/1612.05086

    """
    def __init__(self, initial_sz, lam=.1, update_every=20):
        self.sz = initial_sz  # step_size
        self.initial_sz = initial_sz
        self.prev_loss = None
        self.smooth_loss = DoubleExponentialSmoother(lam=lam)
        self.smooth_dloss = DoubleExponentialSmoother(lam=lam)
        self.dloss_var_sum = 0.
        self.update_every = update_every
        self.n = 0
        self.initial_loss = None
        self.initial_var = None

    def update(self, dist, non_adv):

        # update smoothed_loss
        loss = dist  # *(1 + .1*(non_adv - .5))
        self.smooth_loss.update(loss)

        if self.prev_loss is not None:
            dloss = (self.prev_loss - loss) / self.sz

            # self.dloss_var_sum += (loss - self.smooth_loss.get(1.))**2
            if self.smooth_dloss.get(1.) is not None:
                self.dloss_var_sum += (dloss - self.smooth_dloss.get(1.))**2
                self.n += 1

                if self.n % self.update_every == 0:
                    self._change_step_size()
                    self.dloss_var_sum = 0.
                    self.n = 0

            self.smooth_dloss.update(dloss)

        self.prev_loss = loss

        return self.sz

    def _change_step_size(self):
        if self.initial_loss is None:
            self.initial_loss = self.smooth_loss.get()
            self.initial_var = self.dloss_var_sum / self.n
        else:
            self.sz = (
                self.initial_sz *
                (self.smooth_loss.get() / self.initial_loss) /
                ((self.dloss_var_sum / self.n) / self.initial_var))


class BasicStepSizeHandler(object):
    """Basic step size handler.

    If loss increases (resp. decreases) over n_wait_consecutive steps, then
    divide (resp. multiply) current stepsize by 1.2 (resp. 1.1). If current
    image is non-adversarial for n_wait_non_adv consecutive steps, then divide
    current step-size by 2.

    """
    def __init__(self,
                 initial_sz,
                 lam=.1,
                 n_wait_consecutive=3,
                 n_wait_non_adv=30):

        self.sz = initial_sz  # step_size
        self.prev_loss = 0
        self.consecutive_loss_incr = 0
        self.consecutive_loss_decr = 0
        self.consecutive_non_adv = 0
        self.n_wait_consecutive = n_wait_consecutive
        self.n_wait_non_adv = n_wait_non_adv

    def update(self, dist, non_adv):
        loss = dist  # *(1 + .1*(non_adv - .5))
        if loss >= self.prev_loss:
            self.consecutive_loss_incr += 1
            self.consecutive_loss_decr = 0
        else:
            self.consecutive_loss_incr = 0
            self.consecutive_loss_decr += 1

        if non_adv == 1:
            self.consecutive_non_adv += 1
        else:
            self.consecutive_non_adv = 0

        if ((self.consecutive_loss_incr == self.n_wait_consecutive)
                and (self.sz > dist/1000.)):
            self.consecutive_loss_incr = 0
            self.consecutive_loss_decr = 0
            self.consecutive_non_adv = 0
            self.sz /= 1.2
        elif ((self.consecutive_loss_decr == self.n_wait_consecutive)
                and (self.sz < dist/10.)):
            self.consecutive_loss_incr = 0
            self.consecutive_loss_decr = 0
            self.consecutive_non_adv = 0
            self.sz *= 1.1
        elif self.consecutive_non_adv == self.n_wait_non_adv:
            self.consecutive_loss_incr = 0
            self.consecutive_loss_decr = 0

            # don't divide sz twice consecutively because of non_adv
            self.consecutive_non_adv = -np.inf
            self.sz /= 2

        self.prev_loss = loss

        return self.sz


class GPStepSizeHandler(object):
    """A GP-based step-size handler

    TODO: comment and improve. Pragma: No cover.

    """

    def __init__(self, initial_sz, lam=.1):
        self.sz = initial_sz  # step_size
        self.initial_sz = initial_sz
        self.smooth_loss = DoubleExponentialSmoother(lam=lam, fade_in=True)
        self.loss_res = deque()
        self.locations = deque()
        self.n_updates = 0

    def update(self, dist, non_adv):
        loss = dist  # *(1 + .1*(non_adv - .5))
        self.n_updates += 1
        self._add_loss_residual(loss)
        self.smooth_loss.update(loss)
        if self.n_updates > 5:
            slope = self._covariance_of_loss_residuals()
            sz_with_max_impr = self._sz_with_max_expected_impr(slope)
            self.sz = np.clip(sz_with_max_impr, .99*self.sz, 1.01*self.sz)
        return self.sz

    def _add_loss_residual(self, loss):  # compare loss with predicted loss
        # TODO: pop the first value of loss_res if len(loss_res) > 10
        if self.smooth_loss.get() is not None:
            self.loss_res.append(loss - self.smooth_loss.get(self.sz))
            if len(self.locations) == 0:
                self.locations.append(float(self.sz))
            else:
                self.locations.append(float(self.locations[-1] + self.sz))

            if len(self.loss_res) > 100:
                self.loss_res.popleft()
                self.locations.popleft()

    def _covariance_of_loss_residuals(self):
        n = len(self.loss_res)
        vario_cloud = []
        lags = []
        for i in range(n):
            for j in range(i+1, n):
                dLi = self.loss_res[i]
                dLj = self.loss_res[j]
                lag_ij = self.locations[j] - self.locations[i]

                vario_cloud.append(.5*(dLi - dLj)**2)
                lags.append(lag_ij)

        slope = self._linear_fit(lags, vario_cloud)

        self.lags = lags
        self.vario_cloud = vario_cloud
        self.slope = slope

        return slope

    def _linear_fit(self, lags, vario_cloud):  # linear regression
        # TODO: do a smoothing before.
        X = torch.FloatTensor(lags).view(-1, 1)
        X_tr = X.transpose(0, 1)
        y_tr = torch.FloatTensor(vario_cloud).view(1, -1)
        f = y_tr @ (X @ (X_tr @ X + 1e-6).inverse())  # slope
        return f  # f(x) = f @ x

    def _sz_with_max_expected_impr(self, slope):
        step_sizes = 10**torch.arange(-4, 1, step=.25)
        max_expected_impr = -np.inf
        max_sz = -np.inf
        for sz in step_sizes:
            expected_impr = self._expected_impr(sz, slope)
            if expected_impr > max_expected_impr:
                max_expected_impr = expected_impr
                max_sz = sz
        return max_sz

    def _expected_impr(self, sz, slope):
        v_hat = (slope * sz)
        s_hat = v_hat.sqrt()
        sz = sz.view(-1, 1)
        loss_hat = self.smooth_loss.get(sz)
        # loss_min = self.smooth_loss.get(torch.zeros_like(sz))
        # return ((loss_min - loss_hat) *
        #             self._gauss_cdf(loss_min, loss_hat, s_hat) +
        #             v_hat * self._gauss_pdf(loss_min, loss_hat, s_hat))
        return loss_hat - .0000001*s_hat

    def _gauss_pdf(self, s, mu, sigma):
        if type(s) != torch.Tensor:
            raise ValueError('s, mu and sigma must be pytorch tensors')
        return torch.exp(-(s-mu)**2/(2*sigma**2)) / (2.*np.pi*sigma)

    def _gauss_cdf(self, s, mu, sigma):
        if type(s) != torch.Tensor:
            raise ValueError('s, mu and sigma must be pytorch tensors')
        return 1./2. * (1 + torch.erf((s - mu) / (np.sqrt(2)*sigma)))
