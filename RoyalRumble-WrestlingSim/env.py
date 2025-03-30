# Purpose: Implements the wrestling environment and MDP framework.
# What Happens:
# - Loads wrestler.xml and ring.xml into MuJoCo.
# - Defines the state space (health, stamina, last move, etc.) and 
#   action space (punch, kick, defend, signature move).
# - Handles the MDP’s step() function: executes actions, updates states 
#   (e.g., health -= 10), calculates rewards (e.g., +10 for a hit), and 
#   checks for match end (e.g., health = 0).
# - Integrates with agent.py for action selection.

import os
import tempfile
import numpy as np
from gym import spaces
from gym.utils import EzPickle
import mujoco
import xml.etree.ElementTree as ET
import copy
#from mujoco import MjViewer
import glfw

class WrestlingEnv(EzPickle):
    WIN_REWARD = 500.0
    LOSS_PENALTY = -500.0
    HIT_REWARD = 10.0
    DEFEND_BONUS = 5.0
    TIMESTEP_LIMIT = 1000

    def __init__(self, frame_skip=5, ring_size=3.0):
        EzPickle.__init__(self)
        self.ring_size = ring_size
        self.frame_skip = frame_skip
        self.timestep_limit = self.TIMESTEP_LIMIT
        self.num_steps = 0
        self.mujoco_init = False

        # Scene construction
        scene_xml_path = os.path.join(os.path.dirname(__file__), "assets", "ring.xml")
        wrestler_xml_path = os.path.join(os.path.dirname(__file__), "assets", "wrestler.xml")
        agent_scopes = ["wrestler0", "wrestler1"]

        # Load and parse XMLs
        scene_tree = ET.parse(scene_xml_path)
        scene_root = scene_tree.getroot()
        wrestler_tree = ET.parse(wrestler_xml_path)

        # Find worldbody in scene
        scene_worldbody = scene_root.find('worldbody')

        # Create actuator and sensor sections in the scene if they don't exist
        if scene_root.find('actuator') is None:
            scene_root.append(ET.Element('actuator'))
        if scene_root.find('sensor') is None:
            scene_root.append(ET.Element('sensor'))
        scene_actuator = scene_root.find('actuator')
        scene_sensor = scene_root.find('sensor')

        # Add wrestlers to scene
        init_poses = [(-0.5, 0, 1.0), (0.5, 0, 1.0)]
        for i, scope in enumerate(agent_scopes):
            wrestler_root = copy.deepcopy(wrestler_tree.getroot())
            agent_body = wrestler_root.find('body')
            if agent_body is None:
                raise ValueError("No <body> element found in wrestler.xml")

            agent_body_copy = ET.Element('body')
            for attr, value in agent_body.attrib.items():
                agent_body_copy.set(attr, value)
            for child in agent_body:
                agent_body_copy.append(copy.deepcopy(child))

            agent_body_copy.set('name', scope)
            agent_body_copy.set('pos', ' '.join(map(str, init_poses[i])))

            def prefix_names(element, prefix):
                if 'name' in element.attrib:
                    current_name = element.get('name')
                    if not current_name.startswith(f"{prefix}/"):
                        element.set('name', f"{prefix}/{current_name}")
                for attr in ['joint', 'geom', 'motor', 'sensor', 'body']:
                    if attr in element.attrib:
                        current_ref = element.get(attr)
                        if not current_ref.startswith(f"{prefix}/"):
                            element.set(attr, f"{prefix}/{current_ref}")
                for child in element:
                    prefix_names(child, prefix)

            prefix_names(agent_body_copy, scope)
            scene_worldbody.append(agent_body_copy)

            wrestler_actuators = wrestler_root.find('actuator')
            if wrestler_actuators is not None:
                for actuator in wrestler_actuators:
                    actuator_copy = copy.deepcopy(actuator)
                    prefix_names(actuator_copy, scope)
                    scene_actuator.append(actuator_copy)

            wrestler_sensors = wrestler_root.find('sensor')
            if wrestler_sensors is not None:
                for sensor in wrestler_sensors:
                    sensor_copy = copy.deepcopy(sensor)
                    prefix_names(sensor_copy, scope)
                    scene_sensor.append(sensor_copy)

        with tempfile.TemporaryDirectory() as tmpdir_name:
            scene_filepath = os.path.join(tmpdir_name, "scene.xml")
            scene_tree.write(scene_filepath)
            # with open(scene_filepath, 'r') as f:
            #     print(f.read())
            self.model = mujoco.MjModel.from_xml_path(scene_filepath)
            self.data = mujoco.MjData(self.model)

        self.mujoco_init = True
        self.wrestlers = []
        self.action_space = spaces.Tuple([spaces.Discrete(5), spaces.Discrete(5)])  # 0=punch, 1=kick, 2=defend, 3=signature, 4=no-op
        self.observation_space = spaces.Tuple([spaces.Box(low=-np.inf, high=np.inf, shape=(self.get_obs_size(),)) for _ in range(2)])

        # Initialize renderer
        self.window = None
        self.context = None
        self.cam = None
        self.vopt = None
        self.pert = None
        self.scn = None

    def get_obs_size(self):
        return 3 + 3 + 9 + 9 + 3

    def step(self, actions):
        if not self.mujoco_init:
            return self._get_obs(), (0, 0), (False, False), ({}, {})

        dones = [False, False]
        rewards = [0.0, 0.0]
        infos = [{}, {}]
        self.num_steps += 1

        for i, (wrestler, action) in enumerate(zip(self.wrestlers, actions)):
                wrestler.apply_action(action)
                opponent = self.wrestlers[1 - i]
                if action in [0, 1, 3]:  # Punch, Kick, Signature
                    if self.hit_opponent(wrestler, opponent):
                        damage = 20 if action == 0 else 25 if action == 1 else 35  # Punch: 20, Kick: 25, Signature: 35
                        opponent.health -= damage
                        rewards[i] += damage  # Reward based on damage dealt
                        if opponent.health <= 0:
                            rewards[i] += self.WIN_REWARD
                            rewards[1 - i] -= self.LOSS_PENALTY
                            infos[i]["win"] = True
                            dones = [True, True]
                elif action == 2:  # Defend
                    rewards[i] += 5

        self.do_simulation(self.frame_skip)

        obs = self._get_obs()
        for i, wrestler in enumerate(self.wrestlers):
            opp = self.wrestlers[1 - i]
            state = obs[i]
            health, stamina = state[0], state[1]

            # Reward for hitting the opponent
            if actions[i] in [0, 1, 3]:
                if self.hit_opponent(wrestler, opp):
                    if actions[i] == 3:
                        rewards[i] += 15.0
                        opp.health -= 15
                        opp.stunned = True  # Stun the opponent
                    else:
                        rewards[i] += self.HIT_REWARD
                        opp.health -= 10
                    infos[i]["hit"] = True

            # Reward for defending
            elif actions[i] == 2:
                rewards[i] += self.DEFEND_BONUS

            # Check health and out-of-ring after applying hits
            if health <= 0 or self.out_of_ring(wrestler):
                rewards[i] += self.LOSS_PENALTY
                rewards[1 - i] += self.WIN_REWARD
                dones[i] = True
                infos[i]["lose"] = True
                infos[1 - i]["win"] = True
            elif opp.health <= 0:  # Check if this wrestler's action caused the opponent to lose
                rewards[i] += self.WIN_REWARD
                rewards[1 - i] += self.LOSS_PENALTY
                dones[1 - i] = True
                infos[i]["win"] = True
                infos[1 - i]["lose"] = True
            elif self.num_steps >= self.timestep_limit:  
                dones[i] = True

        #return obs, tuple(rewards), tuple(dones), tuple(infos)
        return obs, rewards, dones, infos

    def do_simulation(self, n_frames):
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self):
        return tuple(wrestler.get_obs() for wrestler in self.wrestlers)

    def out_of_ring(self, wrestler):
        xyz = wrestler.get_qpos()[:3]
        return xyz[2] < 0.3 or np.max(np.abs(xyz[:2])) >= self.ring_size + 0.5

    def hit_opponent(self, wrestler, opponent):
        dist = np.linalg.norm(wrestler.get_qpos()[:2] - opponent.get_qpos()[:2])
        hit = dist < 1.2
        #print(f"Distance: {dist}, Hit: {hit}")
        return hit

    def reset(self):
        self.num_steps = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for i, wrestler in enumerate(self.wrestlers):
            wrestler.reset(xyz=(0.5 if i == 0 else -0.5, 0, 1.0))
        return self._get_obs()

    def render(self, mode="human"):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        if self.window is None:
            self.window = glfw.create_window(800, 600, "Wrestling Simulation", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window")
            glfw.make_context_current(self.window)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.cam = mujoco.MjvCamera()
            self.cam.distance = 5.0
            self.cam.azimuth = 90
            self.cam.elevation = -20
            self.vopt = mujoco.MjvOption()
            self.pert = mujoco.MjvPerturb()
            self.scn = mujoco.MjvScene(self.model, maxgeom=10000)

        # Update the scene
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)

        # Define the viewport
        viewport = mujoco.MjrRect(0, 0, 800, 600)

        # Render the scene
        mujoco.mjr_render(viewport, self.scn, self.context)

        # Add text overlay for match logs using mjr_overlay
        if len(self.wrestlers) == 2:
            wrestler1, wrestler2 = self.wrestlers
            # Prepare the text to display
            text_lines = [
                f"{wrestler1.name} vs {wrestler2.name}",
                f"{wrestler1.name}: Health: {wrestler1.health:.1f}, Stamina: {wrestler1.stamina:.1f}",
                f"{wrestler2.name}: Health: {wrestler2.health:.1f}, Stamina: {wrestler2.stamina:.1f}"
            ]

            # Draw each line of text, adjusting the viewport's bottom position
            base_viewport = mujoco.MjrRect(0, 0, 800, 600)  # Create a copy of the viewport
            for i, line in enumerate(text_lines):
                adjusted_viewport = mujoco.MjrRect(
                    base_viewport.left,
                    base_viewport.bottom + i * 30,  # Shift the bottom edge up for each line
                    base_viewport.width,
                    base_viewport.height
                )
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,      # Font to use
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,  # Position (bottom-left corner)
                    adjusted_viewport,                 
                    line.encode('utf-8'),              
                    None,                              
                    self.context                       
                )

        # Swap buffers and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None
            self.context = None

