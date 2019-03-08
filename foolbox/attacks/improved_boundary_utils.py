import numpy as np


def get_label(logits, stochastic=None, count=True):
    """Returns the label from logits (argmax if stochastic is true, sample
    otherwise)

    logits : `numpy.ndarray`
        The logits from which to sample or get the max label.
    stochastic : bool
        If True, then the label is sampled from the logits. If False,
        then the label is the maximal logit value. If None, uses the value of
        get_label.stochastic. Defaults to True.
    count : bool
        If True, increases the counter get_label.count by one. Used to count
        the total number of function calls. Defaults to True.

    """

    if count:
        if hasattr(get_label, 'count'):
            get_label.count += 1
        else:
            get_label.count = 1

    # if provided, use input 'stochastic', otherwise use get_label.stochastic
    if stochastic is not None:
        stochastic = stochastic
    else:
        if not hasattr(get_label, 'stochastic'):
            get_label.stochastic = True
        stochastic = get_label.stochastic

    if stochastic:
        p = softmax(logits)
        y = np.random.choice(np.arange(len(p)), p=p)
    else:
        y = np.argmax(logits)
    return y


def softmax(X, theta=1.0, axis=None):
    """ Compute the softmax of each element along an axis of X.

    From https://nolanbconaway.github.io/blog/2017/softmax-numpy

    Parameters
    ----------
    X: `numpy.ndarray`. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def find_boundary(a, X2):
    """Find image on the classification boundary between X1 and X2.

    Computes get_label(net(X)) for every X on the line from X1 to X2 with
    step-size .01 and stops after the first label switch. find_boundary returns
    an error if net assigns the same label to X1 and X2.

    a : :class:`Adversarial`
        An :class:`Adversarial` instance.
    X2 : `numpy.ndarray`
        Any image that does not get assigned the same label than
        a.original_class.

    """

    X1 = a.original_image
    y1 = a.original_class

    logits = a.predictions(X2)
    y2 = get_label(logits, stochastic=False, count=False)

    if y1 == y2:
        raise Exception(
                'net needs to output different classes for X1 and X2. But\n' +
                'y1 = %r\ny2=%r' % (y1, y2))
    X_prev = X1
    for p in np.arange(0, 1, .01):
        X = (1-p)*X1 + p*X2
        logits = a.predictions(X)
        y = get_label(logits, stochastic=False, count=False)
        if y != y1:
            break
        X_prev = X

    # X = first img along (X1->X2)-line with different label than X1
    # X_prev = last img along (X1->X2)-line with same label than X
    # p = weight s.t. X = (1-p)*X1 + p*X2
    return X, X_prev, p


def compute_cos(W, X, a):
    y_t = a.original_class
    y_a = a.adversarial_class
    # logits = a.output
    # y_a = get_label(logits, stochastic=False, count=False)

    in_gradient = np.zeros(a.num_classes()).astype(X.dtype)
    in_gradient[y_t] = 1
    in_gradient[y_a] = -1

    G = a.backward(in_gradient, X)  # grad(net(X)[y_t] - net(X)[y_a])

    w = W.flatten()
    g = G.flatten()

    return (w * g).sum() / (np.linalg.norm(w) * np.linalg.norm(g))
