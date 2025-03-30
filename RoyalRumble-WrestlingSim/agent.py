# Purpose: Controls real-time decisions via the MDP policy.
# What Happens:
# - Takes the current state from env.py and selects an action (e.g., punch, defend).
# - Starts with a rule-based policy (e.g., “defend if health < 30”), later upgradable to RL.
# - Uses wrestler genes to influence decisions (e.g., high strength favors punches).


import numpy as np
class WrestlingAgent:
    def __init__(self, wrestler):
        self.wrestler = wrestler
        self.unchanged_health_steps = 0  # Track steps where health doesn't change

    def set_unchanged_health_steps(self, steps):
        self.unchanged_health_steps = steps

    def choose_action(self, observation):
        if self.wrestler.stamina <= 0:
            return 4  # No-op if out of stamina

        # Genes: [strength, agility, defensiveness]
        genes = self.wrestler.genes
        strength, agility, defensiveness = genes[0], genes[1], genes[2]

        # Base action probabilities influenced by genes
        action_probs = np.zeros(5)  # [Punch, Kick, Defend, Signature, No-op]
        action_probs[0] = strength * 0.4  # Punch
        action_probs[1] = agility * 0.3   # Kick
        action_probs[2] = defensiveness * 0.3  # Defend
        action_probs[3] = strength * 0.2   # Signature (requires high strength)
        action_probs[4] = 0.1              # No-op (base probability)

        # Reduce No-op probability if health hasn't changed for a while
        if self.unchanged_health_steps > 2:
            action_probs[4] *= 0.5  # Reduce No-op probability by 50%
            # Redistribute the probability to offensive actions
            action_probs[0] += 0.1  # Increase Punch
            action_probs[1] += 0.05 # Increase Kick
            action_probs[3] += 0.05 # Increase Signature

        # Normalize probabilities
        action_probs /= action_probs.sum()

        # Choose action based on probabilities
        action = np.random.choice(5, p=action_probs)

        # Update stamina based on action
        stamina_costs = {0: 5, 1: 5, 2: 2, 3: 10, 4: 0}
        self.wrestler.stamina = max(0, self.wrestler.stamina - stamina_costs[action])

        return action
    
if __name__ == "__main__":
    from env import WrestlingEnv
    from wrestler import Wrestler
    env = WrestlingEnv()
    wrestler = Wrestler(env, "John Cena", 0, 85, 185, 114, 10)
    agent = WrestlingAgent(wrestler)
    obs = env.reset()
    action = agent.choose_action(obs[0])
    print(f"Chosen action: {action}")