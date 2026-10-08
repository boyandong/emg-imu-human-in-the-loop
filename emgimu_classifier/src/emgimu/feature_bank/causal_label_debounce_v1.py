"""Opt-in causal label confirmation; initialization remains explicitly unknown."""
from numbers import Integral


class CausalLabelDebounceV1:
    def __init__(self, classes=(0, 1, 2), confirmations=2):
        classes = tuple(classes)
        if (not classes or any(isinstance(c, bool) or not isinstance(c, Integral)
                               or c < 0 for c in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError('Classes must be unique nonnegative integers')
        if isinstance(confirmations, bool) or not isinstance(confirmations, Integral) or confirmations < 1:
            raise ValueError('Confirmations must be a positive integer')
        self.classes = frozenset(classes)
        self.confirmations = int(confirmations)
        self.stable = -1
        self.candidate = None
        self.count = 0

    def update(self, label):
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in self.classes:
            raise ValueError('Unknown or noninteger label')
        label = int(label)
        if label == self.stable:
            self.candidate, self.count = None, 0
        else:
            self.count = self.count + 1 if label == self.candidate else 1
            self.candidate = label
            if self.count >= self.confirmations:
                self.stable = label
                self.candidate, self.count = None, 0
        return self.stable
