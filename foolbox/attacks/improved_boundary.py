import warnings
import numpy as np

from .improved_boundary_utils import get_label, compute_cos
from .improved_boundary_smoothers import DoubleExponentialSmoother
from .improved_boundary_stepsize import ConstantStepSizeHandler
from .improved_boundary_samplers import GaussianSampler

from .base import Attack
from .base import call_decorator
from .blended_noise import BlendedUniformNoiseAttack


class ImprovedBoundaryAttack(Attack):
    """Improved boundary attack.

    First move towards a classification boundary (f.ex. by interpolating
    between the original target image X_o and another image with different
    classfication label). Then move along the boundary towards X_o, using
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
                 use_parallel_norm=True,
                 print_every=10,
                 bias_towards_adv=.2,
                 stochastic=False,
                 sz_handler=None,
                 W_sampler=None,
                 perturb_size=.5,
                 W=None,
                 W_actu=.0005,
                 perp=None,
                 perp_actu=.08,
                 classes=None):

        """Applies the improved boundary attack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
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
        use_parallel_norm : bool
            If True, make steps of size step_size * G_parallel_norm. Otherwise,
            normalizes the component parallel to the boundary by its norm.
        print_every : int
            Print logging info every `print_every` steps.
        bias_towards_adv : float
            Bias of the perpendicular component towards preferring adversarial
            examples. Must be between -1 and 1, where 1. means 'by default,
            move along -W, i.e.  perpendicular to estimated boundary, in
            adversarial direction'.
        stochastic : bool
            Use stochastic labeling (i.e. use the network's logits).
        sz_handler :
            A step-size handler.
            Default: RLStepSizeHandler(initial_sz=step_size)
        W_sampler :
            A perturbation sampler to estimate to get a new boundary-weight
            estimations. Default: GaussianSampler(...)
        perturb_size : float
            The norm of the perturbation to be used when estimating a new
            weight with weight_estimate. Must be > 0. Used only if
            W_sampler is not provided.
        W :
            Used to track and smooth the current boundary-weight estimate.
            Default: DoubleExponentialSmoother(lam=W_actu,
                                               fade_in=True)
        W_actu :
            The actualization coefficient when updating W. Must be between 0
            and 1 (included) with 1 meaning no smoothing. Used only if W is not
            provided.
        perp :
            Used to track and smooth the current perpendicular component size.
            Default: DoubleExponentialSmoother(lam=perp_actu,
                                               fade_in=True)
        perp_actu : float
            The smoothing coefficient when updating the perpendicular
            component. Must be between 0 and 1 (included) with 1 meaning no
            smoothing. Used only if perp is not provided.
        classes : list of strings
            The class labels for each output coordinate of the model.
            len(classes) should be equal to the output size of the model.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert ((classes is None) or (len(classes) == a.num_classes()))

        min_, max_ = a.bounds()

        if sz_handler is None:
            # Constant-, RL-, Balles-, Basic-, GP- StepSizeHandler
            sz_handler = ConstantStepSizeHandler(initial_sz=step_size)
        if W_sampler is None:
            # Optimal-, Pixel-, Gaussian-, Image-Sampler()
            W_sampler = \
                GaussianSampler(perturb_size=perturb_size,
                                min_pix=min_, max_pix=max_)
        if W is None:
            # DoubleExponential- or ExponentialSmoother
            W = DoubleExponentialSmoother(lam=W_actu,
                                          fade_in=True)
        if perp is None:
            # DoubleExponential- or ExponentialSmoother
            perp = \
                DoubleExponentialSmoother(lam=perp_actu,
                                          fade_in=True)

        get_label.count = 0
        get_label.stochastic = stochastic

        # Initialize starting point, weight and perpendicular component
        self._starting_point = starting_point
        self._initialization_attack = initialization_attack
        self.initialize_starting_point(a)  # TODO: find boundary using other im
        X = a.image
        y = a.adversarial_class
        y_o = a.original_class
        self.initialize_weights(a, W, W_sampler, steps=0)
        perp.update(bias_towards_adv)

        # Initialize progress monitoring variables
        dist = np.sqrt(
                    (max_ - min_)**2 * X.size *
                    a.normalized_distance(X).value)
        c_o = y_o if classes is None else classes[y_o]
        c = y if classes is None else classes[y]
        count_no_grad = 0
        count_non_adv = 0
        step_size_sum = 0.
        cos_sum = 0.
        all_dist = [dist]
        all_sz = [step_size]
        all_cos = [compute_cos(W, X, a)]

        print('Origin/target class: %s, '
              'Original adversary: %s '
              'Original distance: %.2f' % (c_o, c, dist))

        for i in range(n_sampling_steps):
            logits, is_adv = a.predictions(X)
            y = get_label(logits)
            c = y if classes is None else classes[y]
            non_adv = int(not is_adv)

            W_ = W_sampler(X, y, a, W)
            W.update(W_)
            cur_cos = compute_cos(W, X, a)  # monitor cos btw W and true grad

            perp.update(2*non_adv-1)
            biased_perp = np.clip(perp + bias_towards_adv, -1, 1)

            update_img(W, X, a, step_size,  # *dist/all_dist[0]
                       biased_perp, use_parallel_norm)

            step_size = sz_handler.update(dist, non_adv)

            # Monitor progress
            dist = np.sqrt(
                    (max_ - min_) * X.size *
                    a.normalized_distance(X).value)
            cos_sum += cur_cos
            step_size_sum += step_size
            count_non_adv += non_adv
            count_no_grad += int(np.all(W_ <= 1e-7))
            all_dist.append(dist)
            all_sz.append(step_size)
            all_cos.append(cur_cos)

            if (i+1) % print_every == 0:
                print('ep: %04d neval: %3.1f dist: %5.2f cos:% 3.2f '
                      'no_g:%3.0f%% not_adv:%3.0f%% lam_pe:% 3.2f sz:% '
                      '6.5f yt: %-6s y: %s' % (
                          i, get_label.count/i, dist, cos_sum/print_every,
                          count_no_grad*100/print_every,
                          count_non_adv*100/print_every,
                          biased_perp, step_size_sum/print_every, c_o, c))
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

    def initialize_weights(self, a, W, W_sampler, steps=0):
        X_t = a.original_image
        X = a.image
        y = a.adversarial_class

        W.update(10e-6 * np.random.randn(*X.shape))

        for _ in range(steps):
            W_, _ = W_sampler(X, y, a, W, X_t)
            W.update(W_)


def update_img(W, X, a, step_size,
               perp=.5, use_parallel_norm=False):

    w = W.flatten()
    x = X.flatten()
    x_o = a.original_image.flatten()

    # compute direction along boundary towards original image x_o
    u = w / (np.linalg.norm(w) + 1e-7)  # unitary gradient vector
    g_parallel = x_o - x - ((x_o-x) * u).sum() * u
    g_parallel_norm = np.linalg.norm(g_parallel) + 1e-7
    g_parallel = (g_parallel / g_parallel_norm)  # normalize it

    # compute direction towards boundary
    g_perp = -u

    G_parallel = g_parallel.reshape(*X.shape)
    G_perp = g_perp.reshape(*X.shape)

    if use_parallel_norm:
        X += step_size * (
            (1.-perp) * G_parallel + perp * G_perp) * g_parallel_norm
    else:
        X += step_size * (
            (1.-perp) * G_parallel + perp * G_perp)

    min_, max_ = a.bounds()
    X[:] = np.clip(X, min_, max_)
