from copy import deepcopy

import numpy as np


BINARY_CHTYPE = 'binary'
INTEGER_CHTYPE = 'integer'
REAL_CHTYPE = 'real'
PERMUTATION_CHTYPE = 'permutation'


class Crossover:
    def __init__(self) -> None:
        pass

    def __call__(self, ch1, ch2):
        pass


class OnePointCrossover(Crossover):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, ch1, ch2):
        ch1_, ch2_ = deepcopy(ch1), deepcopy(ch2)
        length = len(ch1.genotype)
        k = np.random.randint(1, length)
        
        temp = ch1_.genotype[k:]
        ch1_.genotype[k:] = ch2_.genotype[k:]
        ch2_.genotype[k:] = temp

        return ch1_, ch2_

    def __repr__(self):
        return 'OnePointCrossover()'


class TwoPointCrossover(Crossover):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, ch1, ch2):
        ch1_, ch2_ = deepcopy(ch1), deepcopy(ch2)
        length = len(ch1.genotype)
        if length <= 2:
            p1, p2 = 0, 1
        else:
            p1, p2 = np.random.choice(range(1, length - 1), 2, replace=False)
            if p1 > p2:
                p1, p2 = p2, p1
        
        temp = ch1_.genotype[p1:p2]
        ch1_.genotype[p1:p2] = ch2_.genotype[p1:p2]
        ch2_.genotype[p1:p2] = temp

        return ch1_, ch2_

    def __repr__(self):
        return 'TwoPointCrossover()'


class UniformCrossover(Crossover):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, ch1, ch2):
        ch1_, ch2_ = deepcopy(ch1), deepcopy(ch2)
        length = len(ch1.genotype)
        if length == 1:
            return ch1_, ch2_
        
        is_switched = (np.random.random((length,)) >= 0.5)
        for i in range(length):
            if is_switched[i]:
                temp = ch1_.genotype[i]
                ch1_.genotype[i] = ch2_.genotype[i]
                ch2_.genotype[i] = temp

        return ch1_, ch2_

    def __repr__(self):
        return 'UniformCrossover()'


class PartiallyMappedCrossover(Crossover):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, ch1, ch2):
        ch1_, ch2_ = deepcopy(ch1), deepcopy(ch2)
        length = len(ch1.genotype)

        p1, p2 = np.random.choice(range(1, length - 1), 2, replace=False)
        if p1 > p2:
            p1, p2 = p2, p1

        chunk1, chunk2 = ch1_.genotype[p1:p2], ch2_.genotype[p1:p2]
        mapping = dict(zip(chunk1, chunk2))

        while True:
            cond = False
            for k, v in mapping.items():
                if v in mapping:
                    mapping[k] = mapping[v]
                    mapping.pop(v)
                    cond = True
                    break
            if not cond:
                break

        temp = ch1_.genotype[p1:p2].copy()
        ch1_.genotype[p1:p2] = ch2_.genotype[p1:p2].copy()
        ch2_.genotype[p1:p2] = temp.copy()

        reverse_mapping = {v: k for k, v in mapping.items()}

        for i in range(length):
            if (p1 <= i and i < p2):
                continue

            if ch1_.genotype[i] in mapping:
                ch1_.genotype[i] = mapping[ch1_.genotype[i]]
            elif ch1_.genotype[i] in reverse_mapping:
                ch1_.genotype[i] = reverse_mapping[ch1_.genotype[i]]

            if ch2_.genotype[i] in mapping:
                ch2_.genotype[i] = mapping[ch2_.genotype[i]]
            elif ch2_.genotype[i] in reverse_mapping:
                ch2_.genotype[i] = reverse_mapping[ch2_.genotype[i]]

        return ch1_, ch2_

    def __repr__(self):
        return 'PartiallyMappedCrossover()'


__all__ = {
    'one_point': OnePointCrossover,
    'two_point': TwoPointCrossover,
    'pmx': PartiallyMappedCrossover
}


_default_crossover_by_chromosome_type = {
    BINARY_CHTYPE: TwoPointCrossover,
    INTEGER_CHTYPE: TwoPointCrossover,
    REAL_CHTYPE: TwoPointCrossover,
    PERMUTATION_CHTYPE: PartiallyMappedCrossover
}


def get_crossover_method_by_name(crossover_name):
    return __all__[crossover_name]


def get_default_crossover_by_chromosome_type(chromosome_type):
    return _default_crossover_by_chromosome_type[chromosome_type]
