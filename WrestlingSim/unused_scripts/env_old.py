import numpy as np
from gym import spaces
from gym.utils import EzPickle

class WrestlingEnv(EzPickle):
    WIN_REWARD = 500.0
    LOSS_PENALTY = -500.0
    HIT_REWARD = 10.0
    DEFEND_BONUS = 5.0
    TIMESTEP_LIMIT = 1000

    def __init__(self, ring_size=3.0):
        EzPickle.__init__(self)
        self.ring_size = ring_size
        self.timestep_limit = self.TIMESTEP_LIMIT
        self.num_steps = 0
        self.wrestlers = []
        self.positions = []
        self.action_space = spaces.Tuple([spaces.Discrete(5), spaces.Discrete(5)])
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(7,)) for _ in range(2)
        ])

    def step(self, actions):
        self.entry_timer += 1
        
        # Add new wrestler every 20 timesteps
        if (self.entry_timer % 20 == 0 and 
            len(self.wrestlers) > len(self.active_wrestlers) + len(self.eliminated_wrestlers)):
            self._add_new_wrestler()
        
        rewards = {w.id: 0 for w in self.active_wrestlers}
        dones = {w.id: False for w in self.active_wrestlers}
        infos = {w.id: {} for w in self.active_wrestlers}
        
        # Select combatants
        initiator, responder = self._select_combatants()
        
        if initiator and responder:
            # Get positions
            init_pos = self.positions[initiator.match_pos]
            resp_pos = self.positions[responder.match_pos]
            
            # Calculate center direction
            center = np.array([0, 0])
            center_dir = (center - init_pos) / max(np.linalg.norm(center - init_pos), 0.1)
            
            # Initiator moves toward center and opponent
            attack_dir = (resp_pos - init_pos) / max(np.linalg.norm(resp_pos - init_pos), 0.1)
            move_dir = (0.3 * center_dir + 0.7 * attack_dir)  # 70% toward opponent, 30% toward center
            move_dir /= max(np.linalg.norm(move_dir), 0.1)
            
            # Apply initiator movement
            init_pos += move_dir * 0.15  # Faster movement for attack
            init_pos = np.clip(init_pos, -self.ring_size, self.ring_size)
            self.positions[initiator.match_pos] = init_pos
            
            # Process initiator action
            action = actions.get(initiator.id, 4)
            initiator.apply_action(action)
            
            # Check if attack lands
            distance = np.linalg.norm(init_pos - resp_pos)
            if action in [0, 1, 3] and distance < 1.2:
                damage = 20 if action == 0 else 25 if action == 1 else 35
                responder.health -= damage
                rewards[initiator.id] += damage
                
                if responder.health <= 0:
                    self._handle_elimination(responder)
                    rewards[initiator.id] += self.WIN_REWARD
                    dones[responder.id] = True
                    infos[initiator.id]["win"] = True
                    infos[responder.id]["lose"] = True
            
            elif action == 2:  # Defend
                rewards[initiator.id] += self.DEFEND_BONUS
            
            # Responder can choose to evade or counter
            if actions.get(responder.id, 4) == 2:  # Defending
                # Move away from initiator
                evade_dir = (resp_pos - init_pos) / max(np.linalg.norm(resp_pos - init_pos), 0.1)
                resp_pos += evade_dir * 0.1
                resp_pos = np.clip(resp_pos, -self.ring_size, self.ring_size)
                self.positions[responder.match_pos] = resp_pos
        
        # Check match end condition
        if len(self.active_wrestlers) <= 1:
            for w in self.active_wrestlers:
                dones[w.id] = True
                if len(self.active_wrestlers) == 1:
                    infos[w.id]["winner"] = True
        
        return self._get_obs(), rewards, dones, infos

    def reset(self):
        self.num_steps = 0
        self.positions = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
        if self.wrestlers:
            for i, wrestler in enumerate(self.wrestlers):
                wrestler.reset(xyz=(self.positions[i][0], self.positions[i][1], 1.0))
        return self._get_obs()

    def _get_obs(self):
        if len(self.wrestlers) != 2:
            return (np.zeros(7), np.zeros(7))

        obs = []
        for i, wrestler in enumerate(self.wrestlers):
            self_pos = self.positions[i]
            opp_pos = self.positions[1 - i]
            obs.append(np.concatenate([
                np.array([wrestler.health, wrestler.stamina, wrestler.last_action or 0]),
                self_pos,
                opp_pos
            ]))
        return tuple(obs)

    def _update_simulation(self):
        for wrestler in self.active_wrestlers:
            if wrestler.last_action == 4:  # No-op - recover stamina
                continue
                
            self_pos = self.positions[wrestler.match_pos]
            
            # Calculate movement direction (combination of center and nearest opponent)
            center_vec = -self_pos / max(np.linalg.norm(self_pos), 0.1)  # Toward center
            
            opp_vec = np.zeros(2)
            if wrestler._opponents:
                nearest_opp = min(wrestler._opponents, 
                                key=lambda w: np.linalg.norm(self.positions[w.match_pos] - self_pos))
                opp_pos = self.positions[nearest_opp.match_pos]
                opp_vec = (opp_pos - self_pos) / max(np.linalg.norm(opp_pos - self_pos), 0.1)
            
            # Combined direction (60% toward center, 40% toward opponent)
            direction = (0.6 * center_vec + 0.4 * opp_vec)
            direction /= max(np.linalg.norm(direction), 0.1)  # Normalize
            
            # Apply movement
            move_speed = 0.05 if wrestler.last_action == 2 else 0.1  # Slower when defending
            new_pos = self_pos + direction * move_speed
            
            # Keep within ring bounds
            new_pos[0] = np.clip(new_pos[0], -self.ring_size, self.ring_size)
            new_pos[1] = np.clip(new_pos[1], -self.ring_size, self.ring_size)
            
            self.positions[wrestler.match_pos] = new_pos  

    def hit_opponent(self, wrestler_idx, opponent_idx):
        dist = np.linalg.norm(self.positions[wrestler_idx] - self.positions[opponent_idx])
        return dist < 1.2

    def out_of_ring(self, wrestler_idx):
        pos = self.positions[wrestler_idx]
        return np.max(np.abs(pos)) >= self.ring_size