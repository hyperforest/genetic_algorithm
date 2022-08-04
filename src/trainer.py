from datetime import datetime
import numpy as np
import pandas as pd

from chromosome import BinaryChromosome, IntegerChromosome, PermutationChromosome, RealNumberChromosome
from crossover import PartiallyMappedCrossover, TwoPointCrossover
from mutation import BitFlipMutation, SwapMutation
from selection import RouletteWheelSelection, TournamentSelection

class Trainer:
    def __init__(self,
        chromosome_type,
        chromosome_length,
        fitness_function,
        pop_size=10,
        selection='rws',
        crossover='auto',
        mutation='auto',
        crossover_rate=0.9,
        mutation_rate=0.1,
        min_value=None,
        max_value=None,
        seed=None,
        population=None
    ) -> None:

        self.chromosome_type = chromosome_type
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.pop_size = pop_size
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.population = population

        np.random.seed(seed)
        self.populate()
        self.build_operators()

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

        for _ in range(self.pop_size):
            self.population[_].build_genotype()

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
        self.history['result'].reset_index(drop=True)

        return self.history

