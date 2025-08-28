import random

# Parámetros del algoritmo genético
POP_SIZE = 100
GENOME_LENGTH = 3 #Red Green Blue
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 100
GIVEN_COLOR = [128, 64, 192]  # Color objetivo

def create_individual():
  return [random.randint(0, 255) for _ in range(GENOME_LENGTH)]

def create_population():
  return [create_individual() for _ in range(POP_SIZE)]


def fitness(individual):
  # Fitness basado en la diferencia absoluta con el color objetivo
  return -sum(abs(individual[i] - GIVEN_COLOR[i]) for i in range(GENOME_LENGTH))

def selection(population):
  selected = []
  for _ in range(POP_SIZE):
    a, b = random.sample(population, 2)
    selected.append(a if fitness(a) > fitness(b) else b)
  return selected

def crossover(parent1, parent2):
  if random.random() < CROSSOVER_RATE:
    point = random.randint(1, GENOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
  else:
    return parent1[:], parent2[:]

def mutation(individual):
  return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in individual]

def genetic_algorithm():
  population = create_population()
  best = max(population, key=fitness)
  for _ in range(GENERATIONS):
    selected = selection(population)
    new_population = []
    for i in range(0, POP_SIZE, 2):
      parent1 = selected[i]
      parent2 = selected[i+1]
      child1, child2 = crossover(parent1, parent2)
      new_population.append(mutation(child1))
      new_population.append(mutation(child2))
    population = new_population
    candidate = max(population, key=fitness)
    if fitness(candidate) > fitness(best):
      best = candidate
  return best, fitness(best)

if __name__ == "__main__":
  best_individual, fit = genetic_algorithm()
  print("Mejor individuo (RGB):", best_individual)
  print("Fitness:", fit)