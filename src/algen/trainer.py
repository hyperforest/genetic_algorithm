from datetime import datetime
import numpy as np
import pandas as pd

from .chromosome import BinaryChromosome, IntegerChromosome, PermutationChromosome, RealNumberChromosome
from .crossover import PartiallyMappedCrossover, TwoPointCrossover
from .mutation import BitFlipMutation, SwapMutation
from .selection import RouletteWheelSelection, TournamentSelection

class Trainer:
    '''Class to run a genetic algorithm.

    Arguments
    ---------

    - fitness_function : a callable which takes a Chromosome object as a parameter
        and returns a scalar value (greater fitness value means better chromosome,
        see example)

    - chromosome_type : str in {'binary', 'integer', 'real', 'permutation'},
        default 'binary'. Built-in chromosome type to be used. Ignored when parameter
        `population` is set.

    - population : None, or Numpy array of custom Chromosome object.
        Pre-defined population. Useful when a custom Chromosome object is used.
        If so, a Numpy array of shape `(pop_size,)` is expected

    - chromosome_length : int, the length of chromosome genotype (default 10).
        Ignored when parameter `population` is set

    - pop_size : int (default 10). The population size.
        Ignored when parameter `population` is set

    - selection : a custom Selection object, or str in {'rws', 'tournament'}
        (default 'rws'). The selection method used to generate parents for crossover.
        - `'rws'` : Roulette Wheel Selection
        - `'tournament'` : Tournament selection with tournament size of 2

    - crossover : str or a Crossover object (default `auto`).
        If 'auto':
        - if `chromosome_type` is in {'binary', 'integer', 'real'}, two-point
            crossover will be used
        - if `chromosome_type` is 'permutation`, partially-mapped crossover
            (PMX) will be used
        Should be set to custom Crossover object if custom Chromosome is used.

    - mutation : str or a Mutation object (default `auto`).
        If 'auto':
        - if `chromosome_type` is in {'integer', 'real', 'permutation}, swap
            mutation will be used
        - if `chromosome_type` is 'binary`, bit-flip mutation will be used
        Should be set to custom Mutation object if custom Chromosome is used.

    - crossover_rate : float, the crossover rate (default 0.9).

    - mutation_rate : float, the mutation rate (default 0.1).

    - min_value : None, or int or float. The minimum value for chromosome genotype
        (default None).
        - If `chromosome_type` is 'integer', `min_value` will be set to 1
        - If `chromosome_type` is 'real', `min_value` will be set to 0
        Otherwise, this parameter will be ignored

    - max_value : None, or int or float. The maximum value for chromosome genotype
        (default None).
        - If `chromosome_type` is 'integer', `max_value` will be set to 2 ** 31 - 1
        - If `chromosome_type` is 'real', `max_value` will be set to 1
        Otherwise, this parameter will be ignored

    - seed : None, or int (default None). The seed of randomness. Set to int
        if reproducibility for each run is desired.
    '''

    def __init__(
        self,
        fitness_function,
        chromosome_type='binary',
        population=None,
        chromosome_length=10,
        pop_size=10,
        selection='rws',
        crossover='auto',
        mutation='auto',
        crossover_rate=0.9,
        mutation_rate=0.1,
        min_value=None,
        max_value=None,
        seed=None
    ):
        self.fitness_function = fitness_function
        self.chromosome_type = chromosome_type
        self.population = population
        self.chromosome_length = chromosome_length
        self.pop_size = pop_size
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed

        np.random.seed(seed)
        self.populate()
        self.build_operators()

    def __repr__(self):
        repr = '\n'.join([
            f'Trainer(',
            f'  chromosome_type={self.chromosome_type},',
            f'  population={self.population},',
            f'  chromosome_length={self.chromosome_length},'
        ])

        return repr

    def populate(self):
        if self.chromosome_type == 'binary':
            self.population = np.array([
                BinaryChromosome(length=self.chromosome_length)
                for _ in range(self.pop_size)
            ])
        elif self.chromosome_type == 'integer':
            self.population = np.array([
                IntegerChromosome(
                    length=self.chromosome_length,
                    min_value=self.min_value,
                    max_value=self.max_value
                ) for _ in range(self.pop_size)
            ])
        elif self.chromosome_type == 'real':
            self.population = np.array([
                RealNumberChromosome(
                    length=self.chromosome_length,
                    min_value=self.min_value,
                    max_value=self.max_value
                ) for _ in range(self.pop_size)
            ])
        elif self.chromosome_type == 'permutation':
            self.population = np.array([
                PermutationChromosome(length=self.chromosome_length)
                for _ in range(self.pop_size)
            ])

    def build_operators(self):
        if self.selection == 'rws':
            self.selection = RouletteWheelSelection()
        elif self.selection == 'tournament':
            self.selection = TournamentSelection()

        if self.crossover == 'auto':
            if self.chromosome_type in ('binary', 'integer', 'real'):
                self.crossover = TwoPointCrossover()
            elif self.chromosome_type == 'permutation':
                self.crossover = PartiallyMappedCrossover()

        if self.mutation == 'auto':
            if self.chromosome_type == 'binary':
                self.mutation = BitFlipMutation()
            elif self.chromosome_type in ('integer', 'real', 'permutation'):
                self.mutation = SwapMutation()

    def run(self, num_generations, verbose=2, start_from_gen=0):
        self.best_gen, self.best_chromosome, self.best_fitness = 0, None, -float('inf')
        
        if not getattr(self, '_ever_run', None):
            self.history = {
                'result': pd.DataFrame({
                    'generation': [],
                    'best': [],
                    'average': []
                })
            }

        fitness = np.array([
            self.fitness_function(pop) for pop in self.population
        ])

        start = datetime.now()

        for gen in range(start_from_gen, start_from_gen + num_generations):
            start_gen = datetime.now()

            parents = self.selection(self.population, fitness_values=fitness)
            
            offsprings = []
            for i in range(0, self.pop_size - (self.pop_size % 2), 2):
                r = np.random.rand()
                if r >= self.crossover_rate:
                    offspring1, offpsring2 = parents[i], parents[i + 1]
                else:
                    offspring1, offpsring2 = self.crossover(parents[i], parents[i + 1])
                offsprings.extend([offspring1, offpsring2])

            r = np.random.randn(len(offsprings))
            indices = np.where(r < self.mutation_rate)[0]
            for index in indices:
                offsprings[index] = self.mutation(offsprings[index])

            self.population = np.array(offsprings)
            fitness = np.array([
                self.fitness_function(pop) for pop in self.population
            ])

            best_fitness_this_gen, average_fitness = fitness.max(), fitness.mean()
            finish_gen = datetime.now()
            gen_time_str = str(finish_gen - start_gen).split('.')[0]

            if verbose >= 2:
                msg = ' - '.join([
                    f'[Generation {gen}] - {gen_time_str}',
                    f'Best fitness: {best_fitness_this_gen:.4f}',
                    f'Average fitness: {average_fitness:.4f}'
                ])
                print(msg)

            self.history['result'] = pd.concat([
                self.history['result'], pd.DataFrame({
                    'generation': [gen],
                    'best': [best_fitness_this_gen],
                    'average': [average_fitness]
                })
            ])

            if best_fitness_this_gen >= self.best_fitness:
                self.best_gen = gen
                self.best_fitness = best_fitness_this_gen
                self.best_chromosome = self.population[fitness.argmax()]

        finish = datetime.now()
        elapsed_time = finish - start
        elapsed_time_str = str(elapsed_time).split('.')[0]
        average_time = elapsed_time.seconds / num_generations

        if verbose >= 1:
            print('Best generation :', self.best_gen)
            print('Best fitness    : %.4f' % self.best_fitness)
            print('Best chromosome :', self.best_chromosome)
            print(f'Finished in {elapsed_time_str} (avg {average_time:.2f}s/gen)')

        self._ever_run = True
        self.history['total_runtime'] = elapsed_time.seconds
        self.history['avg_runtime'] = elapsed_time.seconds / num_generations
        self.history['result'].reset_index(drop=True, inplace=True)

        return self.history

