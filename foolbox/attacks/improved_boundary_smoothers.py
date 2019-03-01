import sys
import abc
abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})


class Smoother(ABC):
    def __init__(self, *args, **kwargs):
        return

    @abstractmethod
    def update(self, v):
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    def __getattr__(self, item):
        return getattr(self.value, item)

    def __repr__(self):
        return str(self.value)

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __pow__(self, other):
        return self.value ** other

    def __div__(self, other):
        return self.value / other

    def __floordiv__(self, other):
        return self.value // other

    def __mod__(self, other):
        return self.value % other

    def __lshift__(self, other):
        return self.value << other

    def __rshift__(self, other):
        return self.value >> other

    def __and__(self, other):
        return self.value & other

    def __or__(self, other):
        return self.value | other

    def __xor__(self, other):
        return self.value ^ other

    def __not__(self, other):
        return ~self.value

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other


class ExponentialSmoother(Smoother):
    """Implements an exponential smoother.

    Use smoother.update(v) to update the internally smoothed value with a new
    sample point v. Use smoother.get() to get the exponentially smoothed
    estimate of the current value.

    After the initialization period, each update of the smoothed value
    :math:`v_{\mathrm{smooth}}` with a new value :math:`v` amounts to doing

    .. math::
        v_{\mathrm{smooth}} = (1-\lambda) v_{\mathrm{smooth}} +
        \lambda v

    where :math:`\lambda` is the update weight (or inverse momentum).

    Parameters
    ----------
    lam : float
        The smoothing weight. Must be between 0 and 1 (included).
    v : float or `numpy.ndarray`
        The initial value. Defaults to None. Initializing the exponential
        smoother with a value v different from None amounts to initializing it
        with v=None and then updating it with v.
    fade_in : bool
        If true, uses a specific initialization procedure which converges to
        the stationary update formula from above. Otherwise uses the above
        update formula from the second sample on. Note that with fade_in=False,
        the first sample value v could get unnaturally high weight.

    Fade-in Procedure:
    ------------------
    After a large number :math:`n` of updates, the above formula amounts to
    using an exponential smoothing window with weights :math:`\lambda`,
    :math:`(1-\lambda)\lambda`, :math:`(1-\lambda)^2\lambda`, ... over the
    unsmoothed values :math:`v_n`, :math:`v_{n-1}`, :math:`v_{n-2}`, ...
    respectively. When :math:`n` is large enough, these weights essentially sum
    to 1, so that the smoothing window is automatically normalized. With
    fade_in=True, we use the smoothing window implementation, but where, for
    each :math:`n`, we normalize the window's weights to sum to 1.

    """

    def __init__(self, lam=.1, v=None, fade_in=True):
        self.lam = lam
        self.v = v
        self.fade_in = fade_in
        self.total_weight = 1.  # only used with fade_in

    def update(self, v):
        if self.v is None:
            self.v = v
        else:
            if self.fade_in:
                self.v *= self.total_weight
                self.total_weight = (1-self.lam)*self.total_weight + 1.
                self.v = ((1-self.lam)*self.v + v) / self.total_weight
            else:
                self.v = (1-self.lam)*self.v + self.lam*v

    @property
    def value(self):
        return self.v

    def __str__(self):
        if self.value is None:
            string = 'lam: {:.3f}\nsmoothed value: None'.format(
                self.lam)
        else:
            string = 'lam: {:.3f}\nsmoothed value: {}'.format(
                self.lam, self.value)
        return string


class DoubleExponentialSmoother(Smoother):
    """Implements Brown's linear exponential smoother, which a special kind of
    double exponential smoothing with only one smoothing parameter
    :math:`\lamdba'.

    Use smoother.update(v) to update the internally smoothed value with a new
    sample point v. Use smoother.get() to get the exponentially smoothed
    estimate of the current value.

    See wikipedia_ for further information on the smoothing update.

    Parameters
    ----------
    lam : float
        The smoothing weight. Must be between 0 and 1 (included).
    v : float or `numpy.ndarray`
        The initial value. Defaults to None. Initializing the exponential
        smoother with a value v different from None amounts to initializing it
        with v=None and then updating it with v.
    fade_in : bool
        If true, uses a specific initialization procedure which converges to
        the stationary update formula from above. Otherwise uses the above
        update formula from the second sample on. Note that with fade_in=False,
        the first sample value v could get unnaturally high weight.

    Fade-in Procedure:
    ------------------
    The double exponential smoother keeps two internal values s1 and s2 that
    are essentially updated like (simple) exponential smoothing. When
    fade_in=True, we update each of these values using the smoothing-window
    formulation explained for the :class:`ExponentialSmoother`.

    .. _wikipedia:
        https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing

    """

    def __init__(self, lam=.1, v=None, fade_in=True):
        self.lam = lam
        self.s1 = self.s2 = v
        self.fade_in = fade_in
        self.total_weight = 1.

    def update(self, v):
        if self.s1 is None:
            self.s1 = self.s2 = v
        else:
            if self.fade_in:
                self.s1 *= self.total_weight
                self.s2 *= self.total_weight
                self.total_weight = (1-self.lam)*self.total_weight + 1.
                self.s1 = ((1.-self.lam)*self.s1 + v) / self.total_weight
                self.s2 = ((1.-self.lam)*self.s2 + self.s1) / self.total_weight
            else:
                self.s1 = (1.-self.lam)*self.s1 + self.lam*v
                self.s2 = (1.-self.lam)*self.s2 + self.lam*self.s1

    def get(self, dx=0.):
        if self.s1 is None:
            return None
        else:
            a = 2*self.s1 - self.s2
            b = self.lam / (1-self.lam) * (self.s1 - self.s2)
            return a + dx*b  # predict dx steps ahead

    @property
    def value(self):
        return self.get(dx=0.)

    def __str__(self):
        if self.value is None:
            string = 'lam: {:.3f}\nsmoothed value: None'.format(
                self.lam)
        else:
            string = 'lam: {:.3f}\nsmoothed value: {}'.format(
                self.lam, self.value)
        return string
