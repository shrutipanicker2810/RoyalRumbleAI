# Purpose: Runs a single wrestling match.
# What Happens:
# - Initializes two wrestlers from wrestler.py and the environment from env.py.
# - Loops through timesteps, calling agent.py for actions and env.py for updates.
# - Logs results (rewards, winner) for coach.py and evolution.py.

# env = WrestlingEnv()
# wrestler1, wrestler2 = wrestlers[0], wrestlers[1]
# while not done:
#     action = agent.choose_action(obs, wrestler1)
#     obs, reward, done, _ = env.step(action)

import numpy as np
from env import WrestlingEnv
from wrestler import Wrestler
from agent import WrestlingAgent
import time  # For slowing down the simulation

def run_match(wrestler1, wrestler2, env, render=True):  # Changed default to render=True
    env.wrestlers = [wrestler1, wrestler2]
    wrestler1.set_match_position(0)
    wrestler2.set_match_position(1)
    wrestler1._opponents = [wrestler2]
    wrestler2._opponents = [wrestler1]

    if wrestler1.health == wrestler1.max_health:
        wrestler1.health = wrestler1.max_health
    if wrestler2.health == wrestler2.max_health:
        wrestler2.health = wrestler2.max_health
    wrestler1.stamina = 100
    wrestler2.stamina = 100
    obs = env.reset()

    total_rewards = [0, 0]
    done = False
    timestep = 0
    agent1 = WrestlingAgent(wrestler1)
    agent2 = WrestlingAgent(wrestler2)
    action_names = {0: "Punch", 1: "Kick", 2: "Defend", 3: "Signature", 4: "No-op"}

    print(f"\nMatch: {wrestler1.name} (Health: {wrestler1.health}) vs {wrestler2.name} (Health: {wrestler2.health})")
    print("---------------------------------------------")

    while not done:
        action0 = 4 if wrestler1.stunned else agent1.choose_action(obs[0])
        action1 = 4 if wrestler2.stunned else agent2.choose_action(obs[1])
        actions = [action0, action1]
        if wrestler1.stunned: wrestler1.stunned = False
        if wrestler2.stunned: wrestler2.stunned = False
        obs, rewards, dones, infos = env.step(actions)
        total_rewards = [total_rewards[i] + rewards[i] for i in range(2)]
        timestep += 1

        print(f"Timestep {timestep}:")
        print(f"  {wrestler1.name}: {action_names[actions[0]]}, Reward: {rewards[0]} (Total: {total_rewards[0]}), Health: {wrestler1.health}")
        print(f"  {wrestler2.name}: {action_names[actions[1]]}, Reward: {rewards[1]} (Total: {total_rewards[1]}), Health: {wrestler2.health}")

        if render:
            env.render()
            time.sleep(0.05)  # Slow down to 20 FPS for visibility

        done = any(dones) or wrestler1.health <= 0 or wrestler2.health <= 0
        if "win" in infos[0] or "win" in infos[1]:
            done = True

    # Determine winner and update performance
    if total_rewards[0] > total_rewards[1]:
        winner, loser = wrestler1, wrestler2
    elif total_rewards[1] > total_rewards[0]:
        winner, loser = wrestler2, wrestler1
    else:
        winner = wrestler1 if wrestler1.health >= wrestler2.health else wrestler2
        loser = wrestler2 if wrestler1.health >= wrestler2.health else wrestler1

    wrestler1.update_performance(wrestler2.name, total_rewards[0], wrestler1 == winner)
    wrestler2.update_performance(wrestler1.name, total_rewards[1], wrestler2 == winner)

    print("---------------------------------------------")
    print(f"Match Result: {wrestler1.name} Total Reward: {total_rewards[0]}, {wrestler2.name} Total Reward: {total_rewards[1]}")
    print(f"Final Health: {wrestler1.name}: {wrestler1.health}, {wrestler2.name}: {wrestler2.health}")
    print(f"Final Stamina: {wrestler1.name}: {wrestler1.stamina}, {wrestler2.name}: {wrestler2.stamina}")

    final_state = {
        "rewards": total_rewards,
        "health": [wrestler1.health, wrestler2.health],
        "stamina": [wrestler1.stamina, wrestler2.stamina]
    }
    return final_state

if __name__ == "__main__":
    env = WrestlingEnv()
    w1 = Wrestler(env, "John Cena", 0, 85, 185, 114, 10)
    w2 = Wrestler(env, "The Rock", 1, 90, 196, 118, 10)
    run_match(w1, w2, env, render=True)