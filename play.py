"""
Demonstrates RoboSumo with pre-trained policies.
"""
import click
import gym
import os

import numpy as np
import tensorflow as tf

import robosumo.envs
from robosumo.envs.sumo import SumoEnv
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, set_from_flat

POLICY_FUNC = {
    "mlp": MLPPolicy,
    "lstm": LSTMPolicy,
}


@click.command()
@click.option("--env", type=str,
              default="RoboSumo-Bug-vs-Spider-v0", show_default=True,
              help="Name of the environment.")
@click.option("--policy-names", nargs=2, type=click.Choice(["mlp", "lstm"]),
              default=("mlp", "mlp"), show_default=True,
              help="Policy names.")
@click.option("--param-versions", nargs=2, type=int,
              default=(1, 1), show_default=True,
              help="Policy parameter versions.")
@click.option("--max_episodes", type=int,
              default=20, show_default=True,
              help="Number of episodes.")

def main(env, policy_names, param_versions, max_episodes):
    # Construct paths to parameters
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    params_dir = os.path.abspath(os.path.join(curr_dir, "robosumo/policy_zoo/assets"))

    agent_names = [env.split('-')[1].lower(), env.split('-')[3].lower()]
    param_paths = []
    for a, p, v in zip(agent_names, policy_names, param_versions):
        param_paths.append(
            os.path.abspath(os.path.join(params_dir, a, p, "agent-params-v%d.npy" % v))
        )

    # Create environment
    #env = gym.make(env)
    #env = gym.make(env, agent_names=["ant", "ant"])
    agent_names = [env.split('-')[1].lower(), env.split('-')[3].lower()]
    env = SumoEnv(agent_names=agent_names) 

    for agent in env.agents:
        agent._adjust_z = -0.5

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    # Initialize policies
    print("Policy Names:", policy_names)
    print("Available Policies:", list(POLICY_FUNC.keys()))

    # Ensure the policy type is valid
    for name in policy_names:
        if name not in POLICY_FUNC:
            raise ValueError(f"Invalid policy type '{name}'. Available: {list(POLICY_FUNC.keys())}")

    # Ensure observation & action space validity
    for i in range(len(policy_names)):
        print(f"Agent {i}: Observation Space -> {env.observation_space.spaces[i]}")
        print(f"Agent {i}: Action Space -> {env.action_space.spaces[i]}")

    # Initialize policies
    policy = []
    for i, name in enumerate(policy_names):
        scope = "policy" + str(i)
        policy_instance = POLICY_FUNC[name](
            scope=scope, reuse=False,
            ob_space=env.observation_space.spaces[i],
            ac_space=env.action_space.spaces[i],
            hiddens=[64, 64], normalize=True
        )
        
        if policy_instance is None:
            raise RuntimeError(f"Failed to initialize policy for agent {i} using {name}")
        
        policy.append(policy_instance)
    print("✅ Policies Successfully Created:", policy)



    sess.run(tf.variables_initializer(tf.global_variables()))

    # Load policy parameters
    params = [load_params(path) for path in param_paths]
    for i in range(len(policy)):
        set_from_flat(policy[i].get_variables(), params[i])

    # Play matches between the agents
    num_episodes, nstep = 0, 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    print("Env before reset - ", env)
    print("Registered environments - before :", gym.envs.registry.keys())
    observation = env.reset()
    print("Registered environments - after :", gym.envs.registry.keys())

    print("Type of observation:", type(observation))
    print("Observation after env.reset - ",observation)
    print("-" * 5 + "Episode %d " % (num_episodes + 1) + "-" * 5)
    while num_episodes < max_episodes:
        #env.render()
        action = tuple([
            pi.act(stochastic=True, observation=observation[i])[0]
            for i, pi in enumerate(policy)
        ])
        observation, reward, done, infos = env.step(action)
        env._render(mode="human")


        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
        if done[0]:
            num_episodes += 1
            draw = True
            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}"
                          .format(i, total_scores, num_episodes))
            if draw:
                print("Match tied: Agent {}, Scores: {}, Total Episodes: {}"
                      .format(i, total_scores, num_episodes))
            observation = env.reset()
            nstep = 0
            total_reward = [0.0  for _ in range(len(policy))]

            for i in range(len(policy)):
                policy[i].reset()

            if num_episodes < max_episodes:
                print("-" * 5 + "Episode %d " % (num_episodes + 1) + "-" * 5)


if __name__ == "__main__":
    main()  # ✅ Correct for Click

