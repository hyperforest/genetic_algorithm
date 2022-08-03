import numpy as np


def _check_genotype_is_array(genotype):
    ok = isinstance(genotype, list)
    if isinstance(genotype, np.ndarray):
        ok = (genotype.ndim == 1)
    
    if not ok:
        msg = f'`genotype` should be a list or 1D Numpy array'
        raise TypeError(msg)


def _check_operator(operator):
    ok = (
        isinstance(operator, str)
        or callable(operator)
    )
    if not ok:
        msg = f'{operator.__name__} should be a str or callable'
        raise TypeError(msg)


class Chromosome:
    def __init__(self, seed=None) -> None:
        self.seed = seed

    def build_genotype(self, genotype=None):
        raise NotImplementedError('`build_genotype` method has not declared')

    def set_operators(self, crossover, mutation):
        _check_operator(crossover)
        _check_operator(mutation)

        self.crossover = crossover
        self.mutate = mutation

    def __repr__(self):
        return self.genotype.__repr__()


class BinaryChromosome(Chromosome):
    def __init__(self, length, seed=None) -> None:
        super().__init__(seed=seed)
        self.length = length

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = np.random.randint(2, size=self.length)

    
class IntegerChromosome(Chromosome):
    def __init__(self, length, min_value=0, max_value=(2 ** 31), seed=None) -> None:
        super().__init__(seed=seed)
        self.length = length
        self.min_value = min_value
        self.max_value = max_value

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = np.random.randint(low=self.min_value,
                                              high=self.max_value,
                                              size=self.length)


class RealNumberChromosome(Chromosome):
    def __init__(self, length, min_value=0., max_value=1., seed=None) -> None:
        super().__init__(seed=seed)
        self.length = length
        self.min_value = min_value
        self.max_value = max_value

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = (
                (self.max_value - self.min_value) * np.random.rand(self.length)
                + self.min_value
            )


class PermutationChromosome(Chromosome):
    def __init__(self, length, seed=None) -> None:
        super().__init__(seed=seed)
        self.length = length

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = np.random.permutation(self.length)

