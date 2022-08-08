from copy import deepcopy

import numpy as np


BINARY_CHTYPE = 'binary'
INTEGER_CHTYPE = 'integer'
REAL_CHTYPE = 'real'
PERMUTATION_CHTYPE = 'permutation'


class Mutation:
    def __init__(self) -> None:
        pass

    def __call__(self, chromosome):
        pass


class BitFlipMutation(Mutation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, chromosome):
        ch = deepcopy(chromosome)
        length = len(ch.genotype)
        index = np.random.randint(length)
        ch.genotype[index] = 1 - ch.genotype[index]

        return ch

    def __repr__(self):
        return 'BitFlipMutation()'


class SwapMutation(Mutation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, chromosome):
        ch = deepcopy(chromosome)
        length = len(ch.genotype)
        
        p1, p2 = np.random.choice(range(length), 2, replace=False)
        temp = ch.genotype[p1]
        ch.genotype[p1] = ch.genotype[p2]
        ch.genotype[p2] = temp

        return ch

    def __repr__(self):
        return 'SwapMutation()'


__all__ = {
    'bitflip': BitFlipMutation,
    'swap': SwapMutation
}


_default_mutation_by_chromosome_type = {
    BINARY_CHTYPE: BitFlipMutation,
    INTEGER_CHTYPE: SwapMutation,
    REAL_CHTYPE: SwapMutation,
    PERMUTATION_CHTYPE: SwapMutation
}


def get_mutation_method_by_name(mutation_name):
    return __all__[mutation_name]


def get_default_mutation_by_chromosome_type(chromosome_type):
    return _default_mutation_by_chromosome_type[chromosome_type]
