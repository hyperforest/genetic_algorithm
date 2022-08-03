from time import time
import numpy as np

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

    def run(self, num_generations):
        history = {'best': [], 'average': []}
        best_gen, self.best_chromosome, self.best_fitness = 0, None, -float('inf')

        fitness = np.array([
            self.fitness_function(pop) for pop in self.population
        ])

        start = time()

        for gen in range(num_generations):
            start_gen = time()
            parents_indices = self.selection(self.population, fitness_values=fitness)
            parents = self.population[parents_indices].copy()
            offsprings = []

            for i in range(0, self.pop_size, 2):
                r = np.random.rand()
                if r >= self.crossover_rate:
                    offspring1, offpsring2 = parents[i], parents[i + 1]
                else:
                    offspring1, offpsring2 = self.crossover(parents[i], parents[i + 1])
                offsprings.extend([offspring1, offpsring2])

            for i in range(len(offsprings)):
                r = np.random.rand()
                if r < self.mutation_rate:
                    offsprings[i] = self.mutation(offsprings[i])

            self.population = np.array(offsprings)
            fitness = np.array([
                self.fitness_function(pop) for pop in self.population
            ])

            best_fitness_this_gen, average_fitness = fitness.max(), fitness.mean()
            finish_gen = time()
            msg = ' ---- '.join([
                f'Iteration [{gen}] - {finish_gen - start_gen:.2f}s',
                f'Best fitness: {best_fitness_this_gen:.4f}',
                f'Average fitness: {average_fitness:.4f}'
            ])
            print(msg)

            history['best'].append(best_fitness_this_gen)
            history['average'].append(average_fitness)

            if best_fitness_this_gen >= self.best_fitness:
                best_gen = gen
                self.best_fitness = best_fitness_this_gen
                self.best_chromosome = self.population[fitness.argmax()]

        print('Best generation :', best_gen)
        print('Best fitness    : %.4f' % self.best_fitness)
        print('Best chromosome :', self.best_chromosome)

        finish = time()
        elapsed_time = finish - start
        average_time = elapsed_time / num_generations
        print(f'Finished in {elapsed_time:.2f}s (avg {average_time:.2f}s/gen)')

        return history

