import random
from deap import base, creator, tools, algorithms

# Problema MaxOnes: maximizar el número de unos en una lista binaria

# Definir el tipo de individuo y el objetivo (maximizar)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Parámetros
IND_SIZE = 100  # Tamaño del individuo (número de bits)

toolbox = base.Toolbox()
# Atributo: bit aleatorio (0 o 1)
toolbox.register("attr_bool", random.randint, 0, 1)
# Individuo: lista de bits
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
# Población: lista de individuos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación: contar los unos
def evalMaxOnes(individual):
  return sum(individual),

toolbox.register("evaluate", evalMaxOnes)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
  random.seed(42)
  pop = toolbox.population(n=300)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", lambda x: sum(x)/len(x))
  stats.register("max", max)

  pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                   stats=stats, halloffame=hof, verbose=True)

  print("Mejor individuo:", hof[0])
  print("Número de unos:", sum(hof[0]))

if __name__ == "__main__":
  main()