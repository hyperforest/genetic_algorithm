from datetime import datetime
import numpy as np
import pandas as pd

from algen.callbacks import CallbackList, History, TimeReporter

from .chromosome import get_chromosome_by_name
from .crossover import Crossover, get_crossover_method_by_name, get_default_crossover_by_chromosome_type
from .mutation import Mutation, get_default_mutation_by_chromosome_type, get_mutation_method_by_name
from .selection import Selection, get_selection_method_by_name

class Trainer:
    '''Class to run a genetic algorithm.

    Arguments
    ---------

    - fitness_function : a callable which takes a Chromosome object as a parameter
        and returns a scalar value (greater fitness value means better chromosome,
        see example)

    - chromosome_type : None, or str in {'binary', 'integer', 'real', 'permutation'}
        (default None). Built-in chromosome type to be used.
        Should be an str if parameter `init_pop` is not specified, otherwise
        this parameter will be ignored

    - init_pop : None, or Numpy array of custom Chromosome object.
        Pre-defined population. Useful when a custom Chromosome object is used.
        If so, a Numpy array of shape `(pop_size,)` is expected

    - chromosome_length : int, the length of chromosome genotype (default 10).
        Ignored when parameter `init_pop` is specified

    - pop_size : int (default 10). The population size, should be an even number.
        Ignored when parameter `init_pop` is specified

    - selection : a custom Selection object, or str in {'rws', 'tournament'}
        (default 'rws'). The selection method used to generate parents for crossover.
        - `'rws'` : Roulette Wheel Selection
        - `'tournament'` : Tournament selection with tournament size of 2

    - crossover : a custom Crossover object, or str in {'auto', 'one_point',
        'two_point', 'pmx'} (default `auto`).
        If 'auto':
        - if `chromosome_type` is in {'binary', 'integer', 'real'}, two-point
            crossover will be used
        - if `chromosome_type` is 'permutation`, partially-mapped crossover
            (PMX) will be used
        Should be set to a custom Crossover object if custom Chromosome is used.

    - mutation : a Mutation object, or str in {'auto', 'bitflip', 'swap'} (default `auto`).
        If 'auto':
        - if `chromosome_type` is in {'integer', 'real', 'permutation}, swap
            mutation will be used
        - if `chromosome_type` is 'binary`, bit-flip mutation will be used
        Should be set to a custom Mutation object if custom Chromosome is used.

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
        chromosome_type=None,
        init_pop=None,
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
        np.random.seed(seed)

        self.fitness_function = fitness_function
        self.chromosome_type_ = chromosome_type
        self.init_pop_ = init_pop
        self.chromosome_length = chromosome_length
        self.pop_size = pop_size
        self.selection_ = selection
        self.crossover_ = crossover
        self.mutation_ = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed

        self.populate()
        self._build_selection_method()
        self._build_crossover_method()
        self._build_mutation_method()

    def __repr__(self):
        repr = ',\n\t'.join([
            f'chromosome_type={self.chromosome_type_}',
            f'chromosome_length={self.chromosome_length}',
            f'pop_size={self.pop_size}',
            f'selection={self.selection.__repr__()}',
            f'crossover={self.crossover.__repr__()}',
            f'mutation={self.mutation.__repr__()}',
            f'crossover_rate={self.crossover_rate}',
            f'mutation_rate={self.mutation_rate}',
            f'min_value={self.min_value}',
            f'max_value={self.max_value}',
            f'seed={self.seed}'
        ])
        repr = f'Trainer({repr})'

        return repr

    def populate(self):
        if isinstance(self.chromosome_type_, str):
            self.population = np.array([
                get_chromosome_by_name(self.chromosome_type_)(
                    length=self.chromosome_length,
                    min_value=self.min_value,
                    max_value=self.max_value
                ) for _ in range(self.pop_size)
            ])
        elif isinstance(self.init_pop_, np.ndarray):
            self.population = self.init_pop_
        
    def _build_selection_method(self):
        if isinstance(self.selection_, str):
            self.selection = get_selection_method_by_name(self.selection_)()
        elif isinstance(self.selection_, Selection):
            self.selection = self.selection_
        else:
            msg = f'{self.selection_} is not a valid selection method.'
            raise TypeError(msg)

    def _build_crossover_method(self):
        if isinstance(self.crossover_, str):
            if self.crossover_ == 'auto':
                self.crossover = get_default_crossover_by_chromosome_type(
                    self.chromosome_type_
                )()
            else:
                self.crossover = get_crossover_method_by_name(self.crossover_)()
        elif isinstance(self.crossover_, Crossover):
            self.crossover = self.crossover_
        else:
            msg = f'{self.crossover_} is not a valid crossover method.'
            raise TypeError(msg)

    def _build_mutation_method(self):
        if isinstance(self.mutation_, str):
            if self.mutation_ == 'auto':
                self.mutation = get_default_mutation_by_chromosome_type(
                    self.chromosome_type_
                )()
            else:
                self.mutation = get_mutation_method_by_name(self.mutation_)()
        elif isinstance(self.mutation_, Mutation):
            self.mutation = self.mutation_
        else:
            msg = f'{self.mutation_} is not a valid mutation method.'
            raise TypeError(msg)

    def calculate_fitness(self):
        return np.array([
            self.fitness_function(pop) for pop in self.population
        ])

    def _single_call_crossover(self, parents):
        offsprings = []
        
        for i in range(0, self.pop_size, 2):
            r = np.random.rand()
            if r >= self.crossover_rate:
                offspring1, offpsring2 = parents[i], parents[i + 1]
            else:
                offspring1, offpsring2 = self.crossover(parents[i], parents[i + 1])
            offsprings.extend([offspring1, offpsring2])
        
        return offsprings

    def _single_call_mutation(self, offsprings):
        r = np.random.randn(len(offsprings))
        indices = np.where(r < self.mutation_rate)[0]
        for index in indices:
            offsprings[index] = self.mutation(offsprings[index])
        return offsprings

    def run(self, num_generations, verbose=2, callbacks=None, start_from_gen=0):
        callbacks = callbacks or []
        callbacks = CallbackList(callbacks)
        callbacks.extend([
            History(verbose=verbose),
            TimeReporter(verbose=verbose)
        ])
        callbacks.set_trainer(self)

        fitness = self.calculate_fitness()
        logs = dict(
            num_generations=num_generations,
            population=self.population.copy(),
            fitness=fitness.copy()
        )
        callbacks.on_run_begin(logs=logs)

        for gen in range(start_from_gen, start_from_gen + num_generations):
            logs = dict(
                population=self.population.copy(),
                fitness=fitness.copy()
            )
            callbacks.on_generation_begin(gen=gen, logs=logs)

            callbacks.on_selection_begin(logs=logs)
            parents = self.selection(self.population, fitness_values=fitness)
            callbacks.on_selection_end(parents=parents, logs=logs)

            callbacks.on_crossover_begin(parents=parents, logs=logs)
            offsprings = self._single_call_crossover(parents)
            callbacks.on_crossover_end(offsprings=offsprings, logs=logs)

            callbacks.on_mutation_begin(offsprings=offsprings, logs=logs)
            offsprings = self._single_call_mutation(offsprings)
            callbacks.on_mutation_end(offsprings=offsprings, logs=logs)
            
            self.population = np.array(offsprings)
            fitness = self.calculate_fitness()

            logs = dict(
                population=self.population.copy(),
                fitness=fitness.copy()
            )
            callbacks.on_generation_end(gen=gen, logs=logs)

        logs = dict(
            num_generations=num_generations,
            population=self.population.copy(),
            fitness=fitness.copy()
        )
        callbacks.on_run_end(logs=logs)

        return self.history

