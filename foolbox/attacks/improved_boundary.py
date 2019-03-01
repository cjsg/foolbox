import warnings
import torch
import numpy as np

from .improved_boundary_utils import compute_cos

# from .improved_boundary_smoothers import ExponentialSmoother
from .improved_boundary_smoothers import DoubleExponentialSmoother

from .improved_boundary_stepsize import RLStepSizeHandler
# from .improved_boundary_stepsize import BallesStepSizeHandler
# from .improved_boundary_stepsize import BasicStepSizeHandler

# from .improved_boundary_estimators import PixelWeightEstimator
from .improved_boundary_estimators import GaussianWeightEstimator
# from .improved_boundary_estimators import ImageWeightEstimator
# from .improved_boundary_estimators import PlugPatchWeightEstimator
# from .improved_boundary_estimators import OptimalWeightEstimator

from .base import Attack
from .base import call_decorator
from .blended_noise import BlendedUniformNoiseAttack

# TODO: don't use pytorch
# TODO: make a.predictions possibly stochastic
# TODO: add a count a.predictions
# TODO: get min_ and max_ from a.bounds()


class ImprovedBoundaryAttack(Attack):
    """Improved boundary attack.

    First move towards a classification boundary (f.ex. by interpolating
    between the original target image X_t and another image with different
    classfication label). Then move along the boundary towards X_t, using
    estimated boundary weight/direction W (with W pointing towards the original
    image class). At every step, update the boundary weight estimate by
    sampling a random perturbation W_, and updating W with

    .. math::
        W \leftarrow (1 - \lambda) W + \lambda \epsilon  W_

    where :math:`\epsilon` is 1 if the classification sign got switched from
    adversarial to non-adversarial, -1 if it got switched from non-adversarial
    to adversarial, and zero if it did not get switched. Note that this update
    corresponds to an exponential smoothing. Other smoothing procedures are
    possible (e.g. double exponential smoothing).

    """

    @call_decorator
    def __call__(self,
                 input_or_adv,
                 label=None,
                 unpack=True,
                 starting_point=None,
                 initialization_attack=None,
                 n_sampling_steps=5000,
                 step_size=.01,
                 with_parallel_norm=True,
                 print_every=10,
                 bias_towards_adv=.2,
                 stochastic=False,
                 sz_handler=None,
                 new_weight_estimate=None,
                 perturbation_size = .5,
                 W_smoother=None,
                 weight_smoothing_coef=.0005,
                 smoothed_lam_perp=None,
                 lam_perp_smoothing_coef=.08,
                 classes=None):

        """Applies the improved boundary attack.

        Parameters
        ----------
        starting_point : `numpy.ndarray`
            Adversarial input to use as a starting point, in particular
            for targeted attacks.
        initialization_attack : :class:`Attack`
            Attack to use to find a starting point. Defaults to
            BlendedUniformNoiseAttack.
        n_sampling_steps : int
            Maximal number of sampling/optimization steps
        step_size : float
            Initial step-size when doing image update
        with_parallel_norm : bool
            If True, make steps of size step_size * G_parallel_norm. Otherwise,
            normalizes the component parallel to the boundary by its norm.
        print_every : int
            Print logging info every `print_every` steps.
        bias_towards_adv : float
            Bias of the perpendicular component towards preferring adversarial
            examples. Must be between -1 and 1, where 1. means 'by default, move
            along -W, i.e.  perpendicular to estimated boundary, in adversarial
            direction'.
        stochastic : bool
            Use stochastic labeling (i.e. use the network's logits).
        sz_handler :
            A step-size handler. Default: RLStepSizeHandler(initial_sz=step_size)
        new_weight_estimate :
            A perturbation sampler to estimate to get a new boundary-weight
            estimations. Default: GaussianWeightEstimator(...)
        perturbation_size : float
            The norm of the perturbation to be used when estimating a new
            weight with weight_estimate. Must be > 0. Used only if
            new_weight_estimate is not provided.
        W :
            Used to track and smooth the current boundary-weight estimate.
            Default: DoubleExponentialSmoother(lam=weight_smoothing_coef,
                                               fade_in=True)
        weight_smoothing_coef :
            The smoothing coefficient when updating a new weight. Must be
            between 0 and 1 (included) with 1 meaning no smoothing. Used only
            if W is not provided.
        smoothed_lam_perp :
            Used to track and smooth the current perpendicular component size.
            Default: DoubleExponentialSmoother(lam=lam_perp_smoothing_coef,
                                               fade_in=True)
        lam_perp_smoothing_coef : float
            The smoothing coefficient when updating lam_perp. Must be between 0
            and 1 (included) with 1 meaning no smoothing. Used only if
            smoothed_lam_perp is not provided.
        classes : list of strings
            The class labels for each output coordinate of the model.
            len(classes) should be equal to the output size of the model.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        min_, max_ = a.bounds()

        if sz_handler is None:
            # RL-, Balles-, Easy-, GP- StepSizeHandler
            sz_handler = RLStepSizeHandler(initial_sz=step_size)
        if new_weight_estimate is None:
            # Optimal-, Pixel-, Gaussian-, Image-WeightEstimator()
            new_weight_estimate = \
                GaussianWeightEstimator(perturbation_size=perturbation_size,
                                        min_pix=min_, max_pix=max_)
        if W is None:
            # DoubleExponential- or ExponentialSmoother
            W = DoubleExponentialSmoother(lam=weight_smoothing_coef,
                                          fade_in=True)
        if smoothed_lam_perp is None:
            # DoubleExponential- or ExponentialSmoother
            smoothed_lam_perp = \
                DoubleExponentialSmoother(lam=lam_perp_smoothing_coef,
                                          fade_in=True)

        # get_label.count = 0
        # get_label.stochastic = stochastic

        self._starting_point = starting_point
        self._initialization_attack = initialization_attack

        # TODO: maybe check if original_image is adversarial

        X_t = a.original_image  # beware: not batch dimension
        y_t = a.original_class

        # X_a = torch.FloatTensor(starting_point)
        # X, _, _ = find_boundary(net, X_t, X_a)  # move to boundary

        self.initialize_starting_point(a)
        X = a.image  # beware: no batch dimension
        y = a.output  # increment

        # W = initialize_weights(X, y, y_t, net, steps=100)
        W.update(10e-6 * np.random.randn(*X.shape))
        lam_perp = 0.  # initial weight of perpendicular component
        smoothed_lam_perp.update(lam_perp)

        c_t = y_t if classes if None else classes[y_t]
        c = y if classes is None else classes[y]
        print('Origin/target class: %s, Original adversary: %s' % (c_t, c))

        prediction_count = 0
        count_no_grad = 0
        count_non_adv = 0
        step_size_sum = 0.
        cos_sum = 0.
        dist_0 = dist = a.normalized_distance(X)  # distance(X, X_t)
        all_dist = [dist_0]
        all_sz = [step_size]
        all_cos = []

        for i in range(n_sampling_steps):
            y = a.predictions(X)  # increase evaluation count
            c = y if classes is None else classes[y]
            non_adv = int(y == y_t)

            W_, X_ = new_weight_estimate(X, y, a, W, X_t)
            W.update(W_)

            smoothed_lam_perp.update(2*non_adv-1)
            lam_perp = np.clip(
                smoothed_lam_perp.get() + bias_towards_adv, -1., 1.)

            update_img(W, X, y, X_t, y_t, step_size,  # *dist/dist_0
                       min_, max_, lam_perp, with_parallel_norm)

            dist = a.normalized_distance(X)
            step_size = sz_handler.update(dist, non_adv)

            all_dist.append(dist)
            step_size_sum += step_size
            all_sz.append(step_size)

            image = X.detach().cpu().numpy()[0]
            y, is_adversarial = a.predictions(image)
            count_non_adv += int(not is_adversarial)

            count_no_grad += int(np.all(W_ <= 1e-7))
            cos_cur = compute_cos(W, X, y_t, net)
            cos_sum += cos_cur
            all_cos.append(cos_cur)

            if (i+1) % print_every == 0:
                print('ep: %04d neval: %3.1f dist: %5.2f cos:% 3.2f '\
                      'no_g:%3.0f%% not_adv:%3.0f%% lam_pe:% 3.2f sz:% '\
                      '6.5f yt: %-6s y: %s' % (
                          i, prediction_count/i, dist, cos_sum/print_every,
                          count_no_grad*100/print_every,
                          count_non_adv*100/print_every,
                          lam_perp, step_size_sum/print_every, c_t, c))
                count_no_grad = 0
                count_non_adv = 0
                step_size_sum = 0.
                cos_sum = 0.

    def initialize_starting_point(self, a):
        starting_point = self._starting_point
        init_attack = self._initialization_attack

        if a.image is not None:
            print(
                'Attack is applied to a previously found adversarial.'
                ' Continuing search for better adversarials.')
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring starting_point parameter because the attack'
                    ' is applied to a previously found adversarial.')
            if init_attack is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring initialization_attack parameter because the'
                    ' attack is applied to a previously found adversarial.')
            return

        if starting_point is not None:
            a.predictions(starting_point)
            assert a.image is not None, ('Invalid starting point provided.'
                                         ' Please provide a starting point'
                                         ' that is adversarial.')
            return

        if init_attack is None:
            init_attack = BlendedUniformNoiseAttack

        if issubclass(init_attack, Attack):
            # instantiate if necessary
            init_attack = init_attack()

        init_attack(a)


def update_img(W, X, y, X_t, y_t, step_size,
               min_=-np.inf, max_=np.inf,
               lam_perp=.5, with_parallel_norm=False):

    # X = current img (1 x 3 x 32 x 32)
    # y = network label of current img X, i.e. net(X)
    # x_t = target img
    # y_t = true label of x_t

    x = X.reshape(1, -1)
    x_t = X_t.reshape(1, -1)
    w = W.reshape(1, -1)

    # compute direction along boundary towards X_t
    u = w / (np.linalg.norm(w) + 1e-7)  # unitary gradient vector
    g_parallel = x_t - x - ((x_t-x) * u).sum() * u
    g_parallel_norm = np.linalg.norm(g_parallel) + 1e-7
    g_parallel = (g_parallel / g_parallel_norm)  # normalize it

    # compute direction towards boundary
    g_perp = -u

    G_parallel = g_parallel.reshape(*X.shape)
    G_perp = g_perp.reshape(*X.shape)

    if with_parallel_norm:
        X += step_size * (
            (1.-lam_perp) * G_parallel + lam_perp * G_perp) * g_parallel_norm
    else:
        X += step_size * (
            (1.-lam_perp) * G_parallel + lam_perp * G_perp)

    # TODO: use a.bounds()
    X[:] = np.clip(X, min_, max_)
