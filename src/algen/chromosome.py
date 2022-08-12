import numpy as np


BINARY_CHTYPE = 'binary'
INTEGER_CHTYPE = 'integer'
REAL_CHTYPE = 'real'
PERMUTATION_CHTYPE = 'permutation'


def _check_genotype_is_array(genotype):
    ok = isinstance(genotype, list)
    if isinstance(genotype, np.ndarray):
        ok = (genotype.ndim == 1)
    
    if not ok:
        msg = f'`genotype` should be a list or 1D Numpy array'
        raise TypeError(msg)


class Chromosome:
    '''Base class for Chromosome with genotype of Numpy array.
    Should not be directly used. Instead, create a subclass for this class.
    '''
    def __init__(self, **kwargs) -> None:
        pass

    def build_genotype(self, genotype=None):
        self.genotype = genotype

    def __repr__(self):
        return self.genotype.__repr__()


class BinaryChromosome(Chromosome):
    '''Chromosome with genotype of binary (0/1) Numpy array.
    '''
    def __init__(self, length, **kwargs) -> None:
        super().__init__(**kwargs)
        self.length = length

        self.build_genotype()

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = np.random.randint(2, size=self.length)

    def __repr__(self):
        repr = ''.join([
            f'BinaryChromosome(length={self.length}, '
            f'genotype={self.genotype.__repr__()}'
        ])
        return repr

    
class IntegerChromosome(Chromosome):
    def __init__(self, length, min_value=None, max_value=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.length = length
        self.min_value = min_value or 1
        self.max_value = max_value or (2 ** 31 - 1)

        self.build_genotype()

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = np.random.randint(low=self.min_value,
                                              high=self.max_value,
                                              size=self.length)

    def __repr__(self):
        repr = ''.join([
            f'IntegerChromosome(length={self.length}, '
            f'min_value={self.min_value}, '
            f'max_value={self.max_value}, '
            f'genotype={self.genotype.__repr__()}'
        ])
        return repr


class RealNumberChromosome(Chromosome):
    def __init__(self, length, min_value=None, max_value=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.length = length
        self.min_value = min_value or 0.
        self.max_value = max_value or 1.

        self.build_genotype()

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = (
                (self.max_value - self.min_value) * np.random.rand(self.length)
                + self.min_value
            )

    def __repr__(self):
        repr = ''.join([
            f'RealNumberChromosome(length={self.length}, '
            f'min_value={self.min_value}, '
            f'max_value={self.max_value}, '
            f'genotype={self.genotype.__repr__()}'
        ])
        return repr


class PermutationChromosome(Chromosome):
    def __init__(self, length, **kwargs) -> None:
        super().__init__(**kwargs)
        self.length = length

        self.build_genotype()

    def build_genotype(self, genotype=None):
        if not isinstance(genotype, type(None)):
            _check_genotype_is_array(genotype)
            self.genotype = genotype
        else:
            self.genotype = np.random.permutation(self.length)

    def __repr__(self):
        repr = ''.join([
            f'PermutationChromosome(length={self.length}, '
            f'genotype={self.genotype.__repr__()}'
        ])
        return repr


__all__ = {
    BINARY_CHTYPE: BinaryChromosome,
    INTEGER_CHTYPE: IntegerChromosome,
    REAL_CHTYPE: RealNumberChromosome,
    PERMUTATION_CHTYPE: PermutationChromosome
}


def get_chromosome_by_name(chromosome_name):
    return __all__[chromosome_name]
