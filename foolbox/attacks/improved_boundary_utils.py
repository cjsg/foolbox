import torch
import numpy as np

# TODO: use numpy instead of pytorch


def get_label(X, net, stochastic=None, count=True):
    """Returns the label assigned by net to X.

    X : `torch.tensor`
        The images to be classified with dimensions bs x c x w x h.
    net : `torch.nn.module`
        The classification network. Must return class-logits.
    stochastic : bool
        If True, then the label is sampled from the logits net(X). If False,
        then the label is the maximal logit value. If None, uses the value of
        get_label.stochastic. Defaults to True.
    count : bool
        If True, increases the counter get_label.count by one. Used to count
        the total number of function calls. Defaults to True.

    """

    from torch.distributions.categorical import Categorical

    classes = ('plane', 'car', 'bird', 'cat', 'deer',   # TODO: no classes
               'dog', 'frog', 'horse', 'ship', 'truck')

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
        out = net(X)
        m = Categorical(logits=out)
        y = m.sample()
        return y.item(), classes[y.item()]

    else:
        _, y = torch.max(net(X), 1)
        return y.item(), classes[y.item()]


def find_boundary(net, X1, X2):
    """Find image on the classification boundary between X1 and X2.

    Computes get_label(net(X)) for every X on the line from X1 to X2 with
    step-size .01 and stops after the first label switch. find_boundary returns
    an error if net assigns the same label to X1 and X2.

    net : `torch.nn.module`
        The classification network. Must return logits.
    X1 : `torch.tensor`
        The original image (for which we search an adversary).
    X2 : `torch.tensor`
        Any image that does not get assigned the same label than X1 by net.

    """

    net.eval()

    _, y1 = torch.max(net(X1), 1)
    _, y2 = torch.max(net(X2), 1)

    if y1 == y2:
        raise Exception(
                'net needs to output different classes for X1 and X2. But\n' +
                'y1 = %r\ny2=%r' % (y1, y2))
    X_prev = X1
    for p in np.arange(0, 1, .01):
        X = (1-p)*X1 + p*X2
        _, y = torch.max(net(X), 1)  # TODO: use get_label!!
        if y != y1:
            break
        X_prev = X

    # X = first img along (X1->X2)-line with different label than X1
    # X_prev = last img along (X1->X2)-line with same label than X
    # p = weight s.t. X = (1-p)*X1 + p*X2
    return X, X_prev, p


def distance(X, X_t):
    """Compute the l2-distance between X and X_t

    """
    x = X.view(1, -1)
    x_t = X_t.view(1, -1)
    return (x-x_t).norm(p=2).item()


def compute_cos(W, X, y_t, net):
    from torch.autograd import grad

    X.requires_grad = True
    out = net(X)
    _, y = torch.max(net(X), 1)
    y = y.item()

    if y_t != y:
        f = out[0, y_t] - out[0, y]
    else:
        _, ixs = torch.topk(out, 2)
        f = out[0, y_t] - out[0, ixs[0, 1]]

    G = grad(f, X)[0]  # torch.sign(grad(f, X)[0])
    X.requires_grad = False

    w = W.view(1, -1)
    g = G.view(1, -1)

    # ## Print the cosinus between g and g.sign()  -> \approx .6
    # g_ = torch.sign(g)
    # print(((g_ * g).sum(dim=1) / (g_.norm(p=2) * g.norm(p=2))).item())

    return ((w * g).sum(dim=1) / (w.norm(p=2) * g.norm(p=2))).item()


def initialize_weights(X, y_o, y_t, net, steps=20):
    # updates classification weights W, return 0 if able to switch labels
    # W = current weight matrix to update
    # X = current image (1 x 3 x 32 x 32)
    # y_o = current label (at img X)
    # y_t = label of original target image X_t
    # net = the network
    net.eval()
    cum_W = torch.zeros_like(X)

    for _ in range(steps):

        dX = torch.randn_like(X)
        dX = dX / dX.norm(p=2).item()
        sgn = -1 if y_o == y_t else 1

        # epsilons = 10**np.linspace(-3,0,num=10)
        epsilons = [.5]
        for eps in epsilons:
            X_ = X + eps*dX
            y, _ = get_label(X_, net)
            if y != y_o:
                cum_W.data += sgn * (dX.data / eps)
                break

            X_ = X - eps*dX
            y, _ = get_label(X_, net)
            if y != y_o:
                cum_W.data -= sgn * (dX.data / eps)
                break

    return cum_W / steps
