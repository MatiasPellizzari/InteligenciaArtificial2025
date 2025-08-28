import random
from deap import base, creator, tools, algorithms

# Problema: Encontrar un número entero específico representado en binario

# Definir el tipo de individuo y el objetivo (maximizar)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Parámetros
IND_SIZE = 8  # Número de bits para representar el entero
TARGET_NUMBER = 173  # Número entero objetivo

toolbox = base.Toolbox()
# Atributo: bit aleatorio (0 o 1)
toolbox.register("attr_bool", random.randint, 0, 1)
# Individuo: lista de bits
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
# Población: lista de individuos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Funcion de evaluacion: diferencia absoluta con el numero objetivo
def evalIntRepresentation(individual):
  # Convertir la lista de bits en un entero
  integer_value = int("".join(map(str, individual)), 2)
  return -abs(integer_value - TARGET_NUMBER),

toolbox.register("evaluate", evalIntRepresentation)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
  random.seed(42)
  pop = toolbox.population(n=300)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", lambda x: sum(x)/len(x))
  stats.register("max", max)

  pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, 
                   stats=stats, halloffame=hof, verbose=True)

  best_individual = hof[0]
  best_value = int("".join(map(str, best_individual)), 2)
  print("Mejor individuo:", best_individual)
  print("Valor entero:", best_value)
  print("Diferencia con el objetivo:", abs(best_value - TARGET_NUMBER))

if __name__ == "__main__":
  main()


