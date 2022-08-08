from copy import deepcopy

import numpy as np


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

