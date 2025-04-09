# Purpose: Evolves wrestlers over time using a Genetic Algorithm.
# What Happens:
# - Calculates fitness (wins, rewards, popularity) from match data.
# - Performs selection, crossover, and mutation on wrestler genes.
# - Updates the wrestler pool for the next “season.”

import numpy as np
from wrestler import Wrestler
from env import WrestlingEnv
import random
from run_match import run_match_pygame

class Evolution:
    def __init__(self, wrestlers, population_size=30, mutation_rate=0.1, crossover_rate=0.8):
        self.wrestlers = wrestlers
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.env = WrestlingEnv()

    def selection(self):
        """Select top performers based on fitness."""
        sorted_wrestlers = sorted(self.wrestlers, key=lambda w: w.fitness, reverse=True)
        return sorted_wrestlers[:self.population_size // 2]  # Top 50%

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(0, len(parent1.genes))
            child_genes = np.concatenate((parent1.genes[:crossover_point], parent2.genes[crossover_point:]))
        else:
            child_genes = parent1.genes.copy()
        return child_genes

    def mutation(self, genes):
        """Mutate genes with a small probability."""
        for i in range(len(genes)):
            if random.random() < self.mutation_rate:
                genes[i] = np.clip(genes[i] + np.random.normal(0, 0.1), 0, 1)
        return genes

    def evolve(self, generations=10):
        """Run the genetic algorithm for a number of generations."""
        for generation in range(generations):
            print(f"Generation {generation + 1}")
            elites = self.selection()
            next_gen = elites.copy()

            while len(next_gen) < self.population_size:
                parent1, parent2 = random.sample(elites, 2)
                child_genes = self.crossover(parent1, parent2)
                child_genes = self.mutation(child_genes)
                
                # Create new wrestler with evolved genes
                new_wrestler = Wrestler(
                    self.env, f"Evolved_{len(next_gen)}", len(next_gen),
                    popularity=parent1.popularity, height=parent1.height,
                    weight=parent1.weight, experience=parent1.experience
                )
                new_wrestler.genes = child_genes
                next_gen.append(new_wrestler)

            self.wrestlers = next_gen

            # Run matches to update fitness, with health reset and output suppressed
            for i in range(0, len(self.wrestlers), 2):
                if i + 1 < len(self.wrestlers):
                    w1, w2 = self.wrestlers[i], self.wrestlers[i + 1]
                    # Reset health and stamina
                    w1.health = w1.max_health
                    w2.health = w2.max_health
                    w1.stamina = 100
                    w2.stamina = 100
                    final_state = run_match_pygame(w1, w2, self.env, render=False, verbose=False)
                    
                    # More robust winner/loser determination
                    winner_name = final_state.get("winner")
                    if winner_name is None:
                        # Fallback: Determine winner based on health
                        winner = w1 if w1.health >= w2.health else w2
                        loser = w2 if w1.health >= w2.health else w1
                    else:
                        # Find winner by name, with fallback
                        try:
                            winner = next(w for w in [w1, w2] if w.name == winner_name)
                            loser = next(w for w in [w1, w2] if w.name != winner_name)
                        except StopIteration:
                            # If names don't match (shouldn't happen), use health
                            winner = w1 if w1.health >= w2.health else w2
                            loser = w2 if w1.health >= w2.health else w1
                    
                    w1.update_performance(w2.name, final_state["rewards"][0], w1 == winner)
                    w2.update_performance(w1.name, final_state["rewards"][1], w2 == winner)

            print(f"Average Fitness: {np.mean([w.fitness for w in self.wrestlers])}")

        return self.wrestlers

if __name__ == "__main__":
    env = WrestlingEnv()
    initial_wrestlers = [
        Wrestler(env, "John Cena", 0, 85, 185, 114, 10),
        Wrestler(env, "The Rock", 1, 90, 196, 118, 10)
    ]
    evolution = Evolution(initial_wrestlers)
    evolved_wrestlers = evolution.evolve()