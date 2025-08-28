import random

# Parámetros del algoritmo genético
LONGITUD_CROMOSOMA = 20
TAMANO_POBLACION = 50
GENERACIONES = 100
PROB_CRUCE = 0.7
PROB_MUTACION = 0.01

def crear_individuo():
  return [random.randint(0, 1) for _ in range(LONGITUD_CROMOSOMA)]

def crear_poblacion():
  return [crear_individuo() for _ in range(TAMANO_POBLACION)]

def fitness(individuo):
  return sum(individuo)

def seleccion(poblacion):
  # Selección por torneo
  seleccionados = []
  for _ in range(TAMANO_POBLACION):
    a, b = random.sample(poblacion, 2)
    seleccionados.append(a if fitness(a) > fitness(b) else b)
  return seleccionados

def cruce(padre1, padre2):
  if random.random() < PROB_CRUCE:
    punto = random.randint(1, LONGITUD_CROMOSOMA - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2
  else:
    return padre1[:], padre2[:]

def mutacion(individuo):
  return [bit if random.random() > PROB_MUTACION else 1 - bit for bit in individuo]

def algoritmo_genetico():
  poblacion = crear_poblacion()
  mejor = max(poblacion, key=fitness)
  for _ in range(GENERACIONES):
    seleccionados = seleccion(poblacion)
    nueva_poblacion = []
    for i in range(0, TAMANO_POBLACION, 2):
      padre1 = seleccionados[i]
      padre2 = seleccionados[i+1]
      hijo1, hijo2 = cruce(padre1, padre2)
      nueva_poblacion.append(mutacion(hijo1))
      nueva_poblacion.append(mutacion(hijo2))
    poblacion = nueva_poblacion
    candidato = max(poblacion, key=fitness)
    if fitness(candidato) > fitness(mejor):
      mejor = candidato
  return mejor, fitness(mejor)

if __name__ == "__main__":
  solucion, valor = algoritmo_genetico()
  print("Mejor solución encontrada:", solucion)
  print("Cantidad de unos:", valor)