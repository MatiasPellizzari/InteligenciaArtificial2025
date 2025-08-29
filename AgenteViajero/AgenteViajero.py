import random


# Parámetros del algoritmo genético
POP_SIZE = 500
GENOME_LENGTH = 30
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.4
GENERATIONS = 500
# Definición de las rutas posibles entre ciudades con costos asociados
# Una matriz donde un valor >0 indica el costo del camino, y 0 indica que no hay camino

ROUTES = [
    [0, 100, 0, 0, 250, 0, 0, 0, 0, 0],    # Ciudad 0
    [100, 0, 220, 0, 0, 180, 0, 0, 0, 0],  # Ciudad 1
    [0, 220, 0, 170, 0, 0, 190, 0, 0, 0],  # Ciudad 2
    [0, 0, 170, 0, 0, 0, 0, 210, 0, 0],    # Ciudad 3
    [250, 0, 0, 0, 0, 160, 0, 0, 0, 0],    # Ciudad 4
    [0, 180, 0, 0, 160, 0, 230, 0, 0, 0],  # Ciudad 5
    [0, 0, 190, 0, 0, 230, 0, 150, 0, 0],  # Ciudad 6
    [0, 0, 0, 210, 0, 0, 150, 0, 120, 0],  # Ciudad 7
    [0, 0, 0, 0, 0, 0, 0, 120, 0, 300],    # Ciudad 8
    [0, 0, 0, 0, 0, 0, 0, 0, 300, 0],      # Ciudad 9
]

def decode(individual):
    # Convierte el genoma binario en una ruta de ciudades
    # Cada grupo de 3 bits representa una ciudad (0-7)
    route = []
    for i in range(0, GENOME_LENGTH, 3):
        city_bits = individual[i:i+3]
        if len(city_bits) < 3:
            continue
        city = int(''.join(str(bit) for bit in city_bits), 2)
        route.append(city)
    # Filtra rutas inválidas (no hay camino entre ciudades consecutivas)
    valid_route = [route[0]] if route else []
    total_cost = 0
    for i in range(1, len(route)):
        prev = valid_route[-1]
        curr = route[i]
        cost = ROUTES[prev][curr]
        if cost > 0:
            valid_route.append(curr)
            total_cost += cost
        else:
            break
    return valid_route, total_cost

def create_individual():
    # Crea un individuo que representa una ruta válida de ciudades
    individual = []
    current_city = random.randint(0, 7)
    individual += [int(x) for x in f"{current_city:03b}"]
    for _ in range((GENOME_LENGTH // 3) - 1):
        # Busca ciudades conectadas desde la ciudad actual
        connections = [i for i, cost in enumerate(ROUTES[current_city]) if cost > 0]
        if not connections:
            # Si no hay conexiones, elige una ciudad aleatoria
            next_city = random.randint(0, 7)
        else:
            next_city = random.choice(connections)
        individual += [int(x) for x in f"{next_city:03b}"]
        current_city = next_city
    return individual

def create_population():
  return [create_individual() for _ in range(POP_SIZE)]


def fitness(individual):
    # Fitness basado en el costo total de la ruta (mayor costo = mejor fitness)
    _, total_cost = decode(individual)
    return total_cost

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
    # Elige una posición de ciudad al azar para mutar
    city_count = GENOME_LENGTH // 3
    idx = random.randint(0, city_count - 1)
    # Decodifica la ruta actual
    route, _ = decode(individual)
    if not route or idx >= len(route):
        return individual[:]  # No hay ruta válida, retorna igual

    mutated = individual[:]
    # Mutar la ciudad en idx por una ciudad conectada diferente
    prev_city = route[idx - 1] if idx > 0 else route[0]
    connections = [i for i, cost in enumerate(ROUTES[prev_city]) if cost > 0 and i != route[idx]]
    if not connections:
        return individual[:]  # No hay conexiones alternativas

    new_city = random.choice(connections)
    # Cambia los bits de la ciudad mutada
    bits = [int(x) for x in f"{new_city:03b}"]
    mutated[idx*3:idx*3+3] = bits

    # Actualiza las siguientes ciudades para que la ruta siga siendo válida
    current_city = new_city
    for j in range(idx + 1, city_count):
        connections = [i for i, cost in enumerate(ROUTES[current_city]) if cost > 0]
        if not connections:
            # Si no hay conexiones, elige aleatoria
            next_city = random.randint(0, 7)
        else:
            next_city = random.choice(connections)
        bits = [int(x) for x in f"{next_city:03b}"]
        mutated[j*3:j*3+3] = bits
        current_city = next_city

    return mutated

def genetic_algorithm():
    population = create_population()
    best = max(population, key=lambda ind: decode(ind)[1])
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
        candidate = max(population, key=lambda ind: decode(ind)[1])
        if decode(candidate)[1] > decode(best)[1]:
            best = candidate
    route, total_cost = decode(best)
    return best, route, total_cost

if __name__ == "__main__":
  best_individual, integer_value, fit = genetic_algorithm()
  print("Mejor individuo:", best_individual)
  print("Valor entero:", integer_value)
  print("Fitness:", fit)