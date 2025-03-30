# Purpose: Defines the Wrestler class with attributes and genes.
# What Happens:
# - Stores static traits (ID, name), dynamic stats (health, stamina, popularity), and genetic traits (strength, agility) for evolution.
# - Links to agent.py for decision-making and evolution.py for GA.

import numpy as np
import mujoco

class Wrestler:
    def __init__(self, env, name, id, popularity, height, weight, experience):
        self._env = env
        self.name = name  # Changed from scope to name
        self.id = id
        self.popularity = popularity  # 1-10 scale
        self.height = height  # in cm
        self.weight = weight  # in kg
        self.experience = experience  # 0-100 scale
        self.health = self.calculate_max_health()  # Initial health based on parameters
        self.max_health = self.health  # Store max health for reference
        self.stamina = 100  # Starting stamina remains 100
        self.genes = np.random.uniform(0, 1, 3)  # Keep genes for action selection
        self.fitness = 0
        self.last_action = None
        self._opponents = []
        self.stunned = False
        self.qpos_start_idx = 0
        self.qpos_end_idx = 0
        self.ctrl_start_idx = 0
        self.ctrl_end_idx = 0
        # New attributes for tracking
        self.wins = 0
        self.losses = 0
        self.total_rewards = 0
        self.match_history = [] # List of dicts: {"opponent": str, "reward": float, "won": bool}

    def update_performance(self, opponent_name, reward, won):
        """Update wrestler's performance metrics after a match."""
        self.match_history.append({"opponent": opponent_name, "reward": reward, "won": won})
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.total_rewards += reward
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        """Calculate fitness based on wins, rewards, and popularity."""
        return (self.wins * 100) + (self.total_rewards * 0.1) + (self.popularity * 10)

    def is_eliminated(self):
        return self.health <= 0

    def _set_joint_indices(self):
        # Use match position (0 or 1) instead of global id
        match_pos = 0 if self.id == self._env.wrestlers[0].id else 1
        self.qpos_start_idx = match_pos * 16
        self.qpos_end_idx = self.qpos_start_idx + 16
        self.ctrl_start_idx = match_pos * 9
        self.ctrl_end_idx = self.ctrl_start_idx + 9

    def set_opponents(self, opponents):
        self._opponents = opponents

    def get_qpos(self):
        return self._env.data.qpos[self.qpos_start_idx:self.qpos_start_idx + 3]  # Just the x, y, z position

    def set_xyz(self, xyz):
        self._env.data.qpos[self.qpos_start_idx:self.qpos_start_idx + 3] = xyz

    def get_obs(self):
        self_pos = self.get_qpos()[:3]
        self_joints = self._env.data.qpos[self.qpos_start_idx + 3:self.qpos_end_idx]
        opp = self._opponents[0]
        opp_pos = opp.get_qpos()[:3]
        return np.concatenate([np.array([self.health, self.stamina, self.last_action or 0]),
                               self_pos, self_joints, opp_pos])

    def apply_action(self, action):
        ctrl = np.zeros(9)
        move_step = 0.1  # Move toward opponent
        opp_pos = self._opponents[0].get_qpos()[:2]
        self_pos = self.get_qpos()[:2]
        direction = (opp_pos - self_pos) / np.linalg.norm(opp_pos - self_pos) if np.linalg.norm(opp_pos - self_pos) > 0 else np.zeros(2)
        
        if action == 0:  # Punch
            ctrl[1] = self.genes[0]
            ctrl[2] = self.genes[0]
            self.stamina -= 5
            new_pos = self_pos + direction * move_step
            self.set_xyz(np.array([new_pos[0], new_pos[1], 1.0]))
        elif action == 1:  # Kick
            ctrl[5] = self.genes[1]
            ctrl[6] = self.genes[1]
            self.stamina -= 7
            new_pos = self_pos + direction * move_step
            self.set_xyz(np.array([new_pos[0], new_pos[1], 1.0]))
        elif action == 2:  # Defend
            ctrl[:] = 0.1
            self.stamina -= 1
        elif action == 3:  # Signature
            ctrl[3] = self.genes[0] * 1.5
            ctrl[4] = self.genes[0] * 1.5
            self.stamina -= 20
            new_pos = self_pos + direction * move_step
            self.set_xyz(np.array([new_pos[0], new_pos[1], 1.0]))
        elif action == 4:  # No-op
            self.stamina += 5
        self.stamina = max(0, min(100, self.stamina))
        self.last_action = action
        self._env.data.ctrl[self.ctrl_start_idx:self.ctrl_end_idx] = ctrl

    def reset(self, xyz=None):
        # Reset stamina and position, but keep current health
        self.stamina = 100
        self.stunned = False
        if xyz is not None:
            self.set_xyz(xyz)
        self.last_action = None
        
    def set_match_position(self, match_pos):
        # Set joint indices based on match position (0 or 1)
        self.qpos_start_idx = match_pos * 16
        self.qpos_end_idx = self.qpos_start_idx + 16
        self.ctrl_start_idx = match_pos * 9
        self.ctrl_end_idx = self.ctrl_start_idx + 9

    def calculate_max_health(self):
        # Health equation: combination of weight (durability), experience (resilience), and height (slight influence)
        base_health = 50  # Minimum health
        weight_factor = self.weight * 1.5  # Heavier wrestlers can take more hits (0.5 per kg)
        experience_factor = self.experience * 2  # More experienced are tougher (1 per point)
        height_factor = self.height * 0.5  # Taller wrestlers slightly tougher (0.1 per cm)
        popularity_factor = self.popularity * 2
        total_health = base_health + weight_factor + experience_factor + height_factor + popularity_factor
        return total_health 
    
    # Health Calculation Explanation
    # Equation: Health = 50 + (Weight * 0.5) + (Experience * 1.0) + (Height * 0.1)
    # - Base Health (50): Ensures every wrestler has a minimum capacity.
    # - Weight * 1.5: Reflects physical durability (e.g., 100 kg adds 50 health).
    # - Experience * 2: Represents resilience from years in the ring (e.g., 90 experience adds 90 health).
    # - Height * 0.5: Slight bonus for reach/size (e.g., 200 cm adds 20 health).