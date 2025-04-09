# Move damage and stamina costs
MOVE_BASE_DAMAGE = {
    4: 10,  # Punch
    5: 15,  # Kick
    6: 25   # Signature
}

MOVE_BASE_STAMINA = {
    4: 5,   # Punch
    5: 7,   # Kick
    6: 20,  # Signature
    7: 0    # Defend
}

# Genetic algorithm parameters
POPULATION_SIZE = 30
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATIONS = 10

# Simulated annealing parameters
INITIAL_TEMPERATURE = 1.0
COOLING_RATE = 0.95