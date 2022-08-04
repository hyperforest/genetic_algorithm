import numpy as np


class Selection:
    def __init__(self) -> None:
        pass

    def __call__(self, population) -> None:
        pass

class RouletteWheelSelection(Selection):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, population, fitness_values) -> None:
        pop_size = len(population)
        fitness_sum = np.sum(fitness_values)
        if fitness_sum != 0:
            prob = fitness_values / fitness_sum
        else:
            prob = np.ones(pop_size, dtype=float) / pop_size
        
        parents_indices = []

        for _ in range(pop_size):
            index = np.random.choice(np.arange(pop_size), p=prob)
            parents_indices.append(index)

        return population[parents_indices].copy()


class TournamentSelection(Selection):
    def __init__(self, tournament_size=2) -> None:
        super().__init__()
        self.tournament_size = tournament_size

    def __call__(self, population, fitness_values) -> None:
        pop_size = len(population)
        parents_indices = []

        for _ in range(pop_size):
            indices = np.random.choice(np.arange(pop_size), self.tournament_size, replace=True)
            best_index = fitness_values[indices].argmax()
            parents_indices.append(indices[best_index])

        return population[parents_indices].copy()

