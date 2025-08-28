import random

def crear_individuo(n):
  # Cada individuo es una permutación de 0..n-1 (una reina por fila, posición en columna)
  individuo = list(range(n))
  random.shuffle(individuo)
  return individuo

def fitness(individuo):
  n = len(individuo)
  colisiones = 0

  # Colisiones verticales (mismo columna)
  for col in range(n):
    if individuo.count(col) > 1:
      colisiones += individuo.count(col) - 1

  # Colisiones diagonales
  for i in range(n):
    for j in range(i+1, n):
      if abs(individuo[i] - individuo[j]) == abs(i - j):
        colisiones += 1

  return -colisiones  # Menos colisiones es mejor

def seleccion(poblacion, scores):
  # Selección por torneo
  torneo = random.sample(list(zip(poblacion, scores)), 3)
  torneo.sort(key=lambda x: x[1], reverse=True)
  return torneo[0][0]

def cruzar(padre, madre):
  n = len(padre)
  corte = random.randint(0, n-1)
  hijo = padre[:corte] + [x for x in madre if x not in padre[:corte]]
  return hijo

def mutar(individuo):
  n = len(individuo)
  i, j = random.sample(range(n), 2)
  individuo[i], individuo[j] = individuo[j], individuo[i]

def algoritmo_genetico(n, tam_poblacion=100, generaciones=1000):
  poblacion = [crear_individuo(n) for _ in range(tam_poblacion)]
  for gen in range(generaciones):
    scores = [fitness(ind) for ind in poblacion]
    if 0 in scores:
      solucion = poblacion[scores.index(0)]
      print(f"Solución encontrada en generación {gen}: {solucion}")
      return solucion
    nueva_poblacion = []
    for _ in range(tam_poblacion):
      padre = seleccion(poblacion, scores)
      madre = seleccion(poblacion, scores)
      hijo = cruzar(padre, madre)
      if random.random() < 0.3:
        mutar(hijo)
      nueva_poblacion.append(hijo)
    poblacion = nueva_poblacion
  print("No se encontró solución.")
  return None

if __name__ == "__main__":
  N = 8  # Cambia N para el tamaño del tablero
  algoritmo_genetico(N)