import numpy as np

from datetime import datetime


class Callback:
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_run_begin(self, logs=None):
        pass

    def on_run_end(self, logs=None):
        pass

    def on_generation_begin(self, gen, logs=None):
        pass

    def on_generation_end(self, gen, logs=None):
        pass

    def on_selection_begin(self, logs=None):
        pass

    def on_selection_end(self, parents, logs=None):
        pass

    def on_crossover_begin(self, parents, logs=None):
        pass

    def on_crossover_end(self, offsprings, logs=None):
        pass

    def on_mutation_begin(self, offsprings, logs=None):
        pass

    def on_mutation_end(self, offsprings, logs=None):
        pass


class CallbackList:
    def __init__(self, callbacks=None, trainer=None):
        self.callbacks = callbacks or []
        if trainer:
            self.set_trainer(trainer)

    def add(self, callback):
        self.callbacks.append(callback)

    def extend(self, callback_list):
        self.callbacks.extend(callback_list)

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_run_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_run_begin(logs=logs)

    def on_run_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_run_end(logs=logs)

    def on_generation_begin(self, gen, logs=None):
        for callback in self.callbacks:
            callback.on_generation_begin(gen, logs=logs)

    def on_generation_end(self, gen, logs=None):
        for callback in self.callbacks:
            callback.on_generation_end(gen, logs=logs)

    def on_selection_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_selection_begin(logs=logs)

    def on_selection_end(self, parents, logs=None):
        for callback in self.callbacks:
            callback.on_selection_end(parents, logs=logs)

    def on_crossover_begin(self, parents, logs=None):
        for callback in self.callbacks:
            callback.on_crossover_begin(parents, logs=logs)

    def on_crossover_end(self, offsprings, logs=None):
        for callback in self.callbacks:
            callback.on_crossover_end(offsprings, logs=logs)

    def on_mutation_begin(self, offsprings, logs=None):
        for callback in self.callbacks:
            callback.on_mutation_begin(offsprings, logs=logs)

    def on_mutation_end(self, offsprings, logs=None):
        for callback in self.callbacks:
            callback.on_mutation_end(offsprings, logs=logs)

class TimeReporter(Callback):
    def __init__(self, verbose=1):
        self.verbose = verbose

    def on_run_begin(self, logs=None):
        self.num_generations = logs.get('num_generations')
        self.start_run_time = datetime.now()

    def on_run_end(self, logs=None):
        self.finish_run_time = datetime.now()
        self.elapsed_run_time = self.finish_run_time - self.start_run_time
        self.avg_elapsed_run_time = self.elapsed_run_time.seconds / self.num_generations
        self.elapsed_run_time_str = str(self.elapsed_run_time).split('.')[0]

        if self.verbose >= 1:
            msg = f'Finished in {self.elapsed_run_time_str}'
            msg += f' (avg {self.avg_elapsed_run_time:.2f}s/generation)'
            print(msg)

    def on_generation_begin(self, gen, logs=None):
        self.start_gen_time = datetime.now()

    def on_generation_end(self, gen, logs=None):
        self.finish_gen_time = datetime.now()
        self.elapsed_gen_time = (self.finish_gen_time - self.start_gen_time).seconds
        
        if self.verbose >= 3:
            msg = f'[Generation {gen}] - Finished in {self.elapsed_gen_time:.1f}s'
            print(msg)


class History(Callback):
    def __init__(self, verbose=2):
        super().__init__()
        self.verbose = verbose
    
    def on_run_begin(self, logs=None):
        num_gen = logs.get('num_generations')
        fitness = logs.get('fitness')

        self.generation = []
        self.fitness = np.zeros((num_gen, self.trainer.pop_size), dtype=fitness.dtype)

        self._best_fitness_so_far = -float('inf')
        self._best_chromosome_so_far = None

    def on_run_end(self, logs=None):
        if self.verbose >= 1:
            print('Best generation :', self.best_generation())
            print('Best fitness    :', self.best_fitness())
            print('Best chromosome :', self.best_chromosome().genotype.__repr__())

    def on_generation_end(self, gen, logs=None):
        fitness = logs.get('fitness')

        self.generation.append(gen)
        index = len(self.generation) - 1
        self.fitness[index, :] = fitness

        best_fitness = fitness.max()
        best_index = fitness.argmax()
        if best_fitness >= self._best_fitness_so_far:
            self._best_fitness_so_far = best_fitness
            self._best_chromosome_so_far = self.trainer.population[best_index]

        self.trainer.history = self

        if self.verbose >= 2:
            average_fitness = fitness.mean()

            best_fitness_str = f'{best_fitness}'
            best_fitness_so_far_str = f'{self._best_fitness_so_far}'
            if isinstance(best_fitness, float):
                best_fitness_str = f'{best_fitness:.4f}'
                best_fitness_so_far_str = f'{self._best_fitness_so_far:.4f}'

            msg = ' - '.join([
                f'[Generation {gen}]',
                f'Average fitness: {average_fitness:.4f}',
                f'Best fitness: {best_fitness_str}',
                f'Best fitness so far: {best_fitness_so_far_str}'
            ])
            print(msg)

    def best_generation(self):
        best_fitness_each_gen = self.fitness.max(axis=1)
        best_fitness_index = best_fitness_each_gen.argmax()
        return self.generation[best_fitness_index]

    def best_fitness(self):
        return self._best_fitness_so_far

    def best_chromosome(self):
        return self._best_chromosome_so_far

    def best_fitness_each_gen(self):
        return self.fitness.max(axis=1)

    def average_fitness_each_gen(self):
        return self.fitness.mean(axis=1)
