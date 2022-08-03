import numpy as np


class Selection:
    def __init__(self, seed=None) -> None:
        self.seed = seed

    def __call__(self, population) -> None:
        pass

class RouletteWheelSelection(Selection):
    def __init__(self, seed=None) -> None:
        super().__init__(seed=seed)

    def __call__(self, population, fitness_values) -> None:
        pop_size = len(population)
        prob = np.cumsum(fitness_values / np.sum(fitness_values))
        parents_indices = []

        for _ in range(pop_size):
            r = np.random.rand()
            index = np.where(prob >= r)[0][0]
            parents_indices.append(index)

        return parents_indices


class TournamentSelection(Selection):
    def __init__(self, tournament_size=2, seed=None) -> None:
        super().__init__(seed=seed)
        self.tournament_size = tournament_size

    def __call__(self, population, fitness_values) -> None:
        pop_size = len(population)
        parents_indices = []

        for _ in range(pop_size):
            indices = np.random.choice(np.arange(pop_size), self.tournament_size, replace=True)
            best_index = fitness_values[indices].argmax()
            parents_indices.append(indices[best_index])

        return parents_indices

