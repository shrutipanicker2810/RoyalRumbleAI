# Purpose: Provides helper functions.
# What Happens:
# Includes logging (e.g., print match stats), random seed setting, and utility functions (e.g., distance calculation).

# def log_match(wrestler1, wrestler2, reward):
#     print(f"Match: {wrestler1.id} vs {wrestler2.id}, Reward: {reward}")

import numpy as np

def set_seed(seed):
    np.random.seed(seed)

def log_match(w1, w2, rewards):
    print(f"Match Log: Wrestler {w1.id} vs Wrestler {w2.id}, Rewards: {rewards}")