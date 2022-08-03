'''
Example for Knapsack Problem
'''

from matplotlib import pyplot as plt
import numpy as np
from trainer import Trainer

seed = 69420
num_generations = 10
pop_size = 20
crossover_rate = 0.9
mutation_rate = 0.1

np.random.seed(seed)
N = 10
weights = np.random.choice(np.arange(9) + 1, N, replace=True)
values = np.random.permutation(N) + 1
cap = int(0.3 * weights.sum())

print('Cap     :', cap)
print('Weights :', weights)
print('Values  :', values)

def fitness(chromosome):
    genotype = chromosome.genotype
    mask = np.where(genotype == 1)
    v, w = values[mask], weights[mask]
    if w.sum() > cap:
        return 0
    return v.sum()


trainer = Trainer(
    chromosome_type='binary',
    chromosome_length=N,
    fitness_function=fitness,
    pop_size=pop_size,
    selection='rws',
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    seed=seed
)

history = trainer.run(num_generations=num_generations)

plt.plot(history['best'], label='best')
plt.plot(history['average'], label='average')
step = num_generations // 10
plt.xticks(range(0, num_generations + step, step))

lo, hi = int(min(history['average'])), max(history['best'])
step = (hi - lo) // 10

plt.yticks(range(lo, hi + step, step))
plt.yticks()
plt.grid()
plt.legend()
plt.xlabel('Generation')
plt.title('Fitness value')
plt.show()
