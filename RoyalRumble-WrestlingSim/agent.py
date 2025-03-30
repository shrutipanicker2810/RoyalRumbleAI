# Purpose: Controls real-time decisions via the MDP policy.
# What Happens:
# - Takes the current state from env.py and selects an action (e.g., punch, defend).
# - Starts with a rule-based policy (e.g., “defend if health < 30”), later upgradable to RL.
# - Uses wrestler genes to influence decisions (e.g., high strength favors punches).


import numpy as np
class WrestlingAgent:
    def __init__(self, wrestler):
        self.wrestler = wrestler

    def choose_action(self, state):
        health, stamina, *rest = state
        if health < 15 and stamina > 20:
            return 2  # Defend
        elif stamina < 10:
            return 4  # No-op
        elif self.wrestler.genes[0] > 0.7 and stamina > 30:  # Stricter conditions for Signature
            return 3  # Signature
        elif self.wrestler.genes[1] > 0.5 and stamina > 5:  # Encourage Kick
            return 1  # Kick
        elif stamina > 5:  # Encourage Punch
            return 0  # Punch
        return 4  # No-op