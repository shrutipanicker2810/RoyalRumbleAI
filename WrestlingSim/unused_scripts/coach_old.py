# Purpose: Optimizes wrestler pairings and strategies between matches using Simulated Annealing.
# What Happens:
# - Uses Simulated Annealing to select wrestler pairs and 
#   strategies (e.g., aggressive vs. defensive) based on match outcomes.
# - Evaluates performance using rewards and popularity from env.py.
# - Introduces controlled randomness for unpredictability.

import numpy as np
from run_match import run_match_pygame
from env import WrestlingEnv
from wrestler import Wrestler
import random

class Coach:
    def __init__(self, wrestlers, env, temperature=1000, cooling_rate=0.95):
        self.wrestlers = wrestlers
        self.env = env
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.strategies = ["aggressive", "defensive", "balanced"]  # Possible strategies

    def evaluate_pairing(self, wrestler1, wrestler2, strategy1, strategy2):
        """Run a match and return the combined fitness of the pairing."""
        # Modify wrestler genes based on strategy
        w1_genes = self.adjust_genes(wrestler1.genes.copy(), strategy1)
        w2_genes = self.adjust_genes(wrestler2.genes.copy(), strategy2)
        wrestler1.genes = w1_genes
        wrestler2.genes = w2_genes
        
        final_state = run_match_pygame(wrestler1, wrestler2, self.env, render=False, verbose=False)
        fitness = wrestler1.fitness + wrestler2.fitness  # Combined fitness
        return fitness

    def adjust_genes(self, genes, strategy):
        """Adjust genes based on selected strategy."""
        if strategy == "aggressive":
            genes[0] = min(1.0, genes[0] + 0.2)  # Boost strength
            genes[1] = max(0.0, genes[1] - 0.1)  # Reduce agility
            genes[2] = max(0.0, genes[2] - 0.1)  # Reduce defensiveness
        elif strategy == "defensive":
            genes[0] = max(0.0, genes[0] - 0.1)
            genes[1] = max(0.0, genes[1] - 0.1)
            genes[2] = min(1.0, genes[2] + 0.2)  # Boost defensiveness
        elif strategy == "balanced":
            # Move genes toward a balanced value (e.g., 0.5)
            genes[0] = genes[0] + (0.5 - genes[0]) * 0.3  # Adjust strength toward 0.5
            genes[1] = genes[1] + (0.5 - genes[1]) * 0.3  # Adjust agility toward 0.5
            genes[2] = genes[2] + (0.5 - genes[2]) * 0.3  # Adjust defensiveness toward 0.5
        return genes

    def simulated_annealing(self, max_iterations=100):
        """Optimize wrestler pairings and strategies."""
        current_pairing = random.sample(self.wrestlers, 2)
        current_strategies = [random.choice(self.strategies), random.choice(self.strategies)]
        current_fitness = self.evaluate_pairing(current_pairing[0], current_pairing[1], 
                                                current_strategies[0], current_strategies[1])

        best_pairing = current_pairing[:]
        best_strategies = current_strategies[:]
        best_fitness = current_fitness

        for _ in range(max_iterations):
            # Generate neighbor solution
            new_pairing = random.sample(self.wrestlers, 2)
            new_strategies = [random.choice(self.strategies), random.choice(self.strategies)]
            new_fitness = self.evaluate_pairing(new_pairing[0], new_pairing[1], 
                                                new_strategies[0], new_strategies[1])

            # Acceptance probability
            delta = new_fitness - current_fitness
            if delta > 0 or random.random() < np.exp(delta / self.temperature):
                current_pairing = new_pairing[:]
                current_strategies = new_strategies[:]
                current_fitness = new_fitness

            if current_fitness > best_fitness:
                best_pairing = current_pairing[:]
                best_strategies = current_strategies[:]
                best_fitness = current_fitness

            self.temperature *= self.cooling_rate  # Cool down

        print(f"Best Pairing: {best_pairing[0].name} ({best_strategies[0]}) vs {best_pairing[1].name} ({best_strategies[1]})")
        print(f"Best Fitness: {best_fitness}")
        return best_pairing, best_strategies

if __name__ == "__main__":
    env = WrestlingEnv()
    wrestlers = [
        Wrestler(env, "John Cena", 0, 85, 185, 114, 10),
        Wrestler(env, "The Rock", 1, 90, 196, 118, 10),
        Wrestler(env, "Triple H", 2, 88, 193, 116, 9)
    ]
    coach = Coach(wrestlers, env)
    coach.simulated_annealing()