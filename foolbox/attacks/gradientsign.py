from __future__ import division
import numpy as np
from collections import Iterable
import logging

from .iterators import LinearSearchIterator, BinarySearchIterator
from .base import Attack
from .base import call_decorator


class GradientSignAttack(Attack):
    """Adds the sign of the gradient to the image, gradually increasing
    the magnitude until the image is misclassified.

    Does not do anything if the model does not have a gradient.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000, max_epsilon=1, bin_search=False):

        """Adds the sign of the gradient to the image, gradually increasing
        the magnitude until the image is misclassified.

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
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.
        bin_search : bool
            If true, use binary search to find the smallest epsilon that
            generates an adversatial perturbation. This assumes epsilons
            is an int or an *ordered* iterable.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()
        gradient = a.gradient()
        gradient_sign = np.sign(gradient) * (max_ - min_)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, max_epsilon, num=epsilons + 1)[1:]
            decrease_if_first = True
        else:
            decrease_if_first = False

        for _ in range(2):  # to repeat with decreased epsilons if necessary
            if bin_search:
                iterator = BinarySearchIterator(epsilons)
            else:
                iterator = LinearSearchIterator(epsilons)

            for i, epsilon in iterator:
                perturbed = image + gradient_sign * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                _, is_adversarial = a.predictions(perturbed)

                # update iterator:
                iterator.is_adversarial = is_adversarial

            if is_adversarial and decrease_if_first and iterator.i < 20:
                logging.info('repeating attack with smaller epsilons')
                max_epsilon = epsilons[i]
                epsilons = np.linspace(0, max_epsilon, num=20 + 1)[1:]
            else:
                return


FGSM = GradientSignAttack


class IterativeGradientSignAttack(Attack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=100, steps=10, max_epsilon=1,
                 stop_early=None, bin_search=False):

        """Like GradientSignAttack but with several steps for each epsilon.

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
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        steps : int
            Number of gradient steps of size epsilon to do
        max_epsilon : float
            Largest step size epsilon if epsilons is not an iterable.
        stop_early : bool
            If true, stop incrementing epsilon as soon as an adversarial
            image is found at the end of the 'steps' steps.
            Must be true if bin_search is true. Defaults to bin_search.
        bin_search : bool
            If true, use binary search to find the smallest epsilon that
            generates an adversatial perturbation. This assumes epsilons
            is an int or an *ordered* iterable.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        if stop_early is None:
            stop_early = bin_search
        assert (stop_early, bin_search) != (False, True), "When bin_search "\
            "is true, stop_early must be set to true as well"

        image = a.original_image
        min_, max_ = a.bounds()

        if not isinstance(epsilons, Iterable):
            assert isinstance(epsilons, int)
            decrease_if_first = True
            epsilons = np.linspace(0,
                                   max_epsilon / steps,
                                   num=epsilons + 1)[1:]
        else:
            decrease_if_first = False

        if bin_search:
            iterator = BinarySearchIterator(epsilons)
        else:
            iterator = LinearSearchIterator(epsilons, stop_early)

        for _ in range(2):
            for i, epsilon in iterator:
                perturbed = image

                for _ in range(steps):
                    gradient = a.gradient(perturbed)
                    gradient_sign = np.sign(gradient) * (max_ - min_)

                    perturbed = perturbed + gradient_sign * epsilon
                    perturbed = np.clip(perturbed, min_, max_)

                    _, is_adversarial = a.predictions(perturbed)

                iterator.is_adversarial = is_adversarial
                # Beware: if stop_early is true, the iterator will stop
                # as soon as it found the smallest epsilon giving an
                # adversarial perturbation. But there might be a bigger
                # epsilon that leads to a smaller adversarial perturbation.

            if is_adversarial and decrease_if_first and iterator.i < 20:
                logging.info('repeating attack with smaller epsilons')
                max_epsilon = epsilons[i]
                epsilons = np.linspace(0, max_epsilon, num=20 + 1)[1:]
            else:
                return
