import pygame
import numpy as np
import random
import math
from wrestler import Wrestler
from agent import WrestlingAgent
from env import WrestlingEnv

def run_match_pygame(wrestler1, wrestler2, env, viz=None, render=True, verbose=True):
    env.wrestlers = [wrestler1, wrestler2]
    wrestler1.set_match_position(0)
    wrestler2.set_match_position(1)
    wrestler1._opponents = [wrestler2]
    wrestler2._opponents = [wrestler1]

    # Reset stats
    wrestler1.health = wrestler1.max_health
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

    last_health1, last_health2 = wrestler1.health, wrestler2.health
    unchanged_health_steps = 0
    max_unchanged_steps = 5

    if verbose:
        print(f"\nMatch: {wrestler1.name} (Health: {wrestler1.health}) vs {wrestler2.name} (Health: {wrestler2.health})")
        print("---------------------------------------------")

    while not done:
        if render and viz:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    viz.close()
                    return None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        viz.handle_panel_click(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    viz.handle_scroll(-event.y * 20)  # Scroll speed

        # Choose actions
        agent1.set_unchanged_health_steps(unchanged_health_steps)
        agent2.set_unchanged_health_steps(unchanged_health_steps)

        action0 = 4 if wrestler1.stunned else agent1.choose_action(obs[0])
        action1 = 4 if wrestler2.stunned else agent2.choose_action(obs[1])
        actions = [action0, action1]
        
        # Store moves before processing
        wrestler1.last_move = action0
        wrestler2.last_move = action1

        # Step environment
        obs, rewards, dones, infos = env.step(actions)
        total_rewards = [total_rewards[i] + rewards[i] for i in range(2)]
        timestep += 1

        # # Reset positions
        wrestler1.reset(xyz=(-0.2, 0.0, 1.0))
        wrestler2.reset(xyz=(0.2, 0.0, 1.0))

        # Check for unchanged health
        if wrestler1.health == last_health1 and wrestler2.health == last_health2:
            unchanged_health_steps += 1
        else:
            unchanged_health_steps = 0
        last_health1, last_health2 = wrestler1.health, wrestler2.health

        if unchanged_health_steps >= max_unchanged_steps:
            if verbose:
                print(f"Match ended early due to {max_unchanged_steps} timesteps with no health change.")
            break

        if verbose:
            print(f"Timestep {timestep}:")
            print(f"  {wrestler1.name}: {action_names[actions[0]]}, Reward: {rewards[0]} (Total: {total_rewards[0]}), Health: {wrestler1.health}")
            print(f"  {wrestler2.name}: {action_names[actions[1]]}, Reward: {rewards[1]} (Total: {total_rewards[1]}), Health: {wrestler2.health}")

        if render and viz:
            viz.render([wrestler1, wrestler2], actions)
            pygame.time.delay(500)
            if 3 in actions:  # Extra delay for signature moves
                pygame.time.delay(500)

        done = any(dones) or wrestler1.health <= 0 or wrestler2.health <= 0

    # Determine winner
    if total_rewards[0] > total_rewards[1]:
        winner, loser = wrestler1, wrestler2
    elif total_rewards[1] > total_rewards[0]:
        winner, loser = wrestler2, wrestler1
    else:
        winner = wrestler1 if wrestler1.health >= wrestler2.health else wrestler2
        loser = wrestler2 if wrestler1.health >= wrestler2.health else wrestler1

    wrestler1.update_performance(wrestler2.name, total_rewards[0], wrestler1 == winner)
    wrestler2.update_performance(wrestler1.name, total_rewards[1], wrestler2 == winner)

    if verbose:
        print("---------------------------------------------")
        print(f"Match Result: {wrestler1.name} Total Reward: {total_rewards[0]}, {wrestler2.name} Total Reward: {total_rewards[1]}")
        print(f"Final Health: {wrestler1.name}: {wrestler1.health}, {wrestler2.name}: {wrestler2.health}")

    if viz:
        viz.add_match_log(winner, loser, winner.health)
    
    return {
        "rewards": total_rewards,
        "health": [wrestler1.health, wrestler2.health],
        "winner": winner.name,
        "loser": loser
    }