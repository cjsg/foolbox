# Defines iterators to be used in (Iterative)Gradient(Sign)Attack

class BinarySearchIterator(object):
    def __init__(self, epsilons):
        self.epsilons = epsilons
        self.maxi = len(epsilons)-1
        self.mini = 0
        self.i = -1
        self.max_tested = self.i
        self.is_adversarial = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if not self.is_adversarial:
            self.mini = self.i + 1
            if self.max_tested < self.maxi: # initialization phase
                self.i = max(self.i + 1, min(2 * self.i, self.maxi))
                self.max_tested = self.i
            elif self.i < self.maxi:
                self.i = int((self.i + self.maxi + 1) / 2)
            else:
                raise StopIteration()
        else:
            self.maxi = self.i
            if self.i > self.mini:
                self.i = int((self.i + self.mini) / 2)
            else:
                raise StopIteration()
        return self.i, self.epsilons[self.i]


class LinearSearchIterator(object):
    def __init__(self, epsilons, stop_early=True):
        self.epsilons = epsilons
        self.maxi = len(epsilons)-1
        self.i = 0
        self.is_adversarial = False
        self.stop_early = stop_early

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.i < self.maxi and (
        not self.stop_early or not self.is_adversarial):
            self.i += 1
            return self.i, self.epsilons[self.i]
        else:
            raise StopIteration()

