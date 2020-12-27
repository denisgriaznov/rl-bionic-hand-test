import gym

from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def main():
    unity_env = UnityEnvironment("../ArmRobot/Build/ArmRobot",worker_id=1)
    env = UnityToGymWrapper(unity_env, True)
    policy_kwargs = dict(net_arch=[128, 128, 64, 32])
    model = DQN(MlpPolicy, env, verbose=1,tensorboard_log="./hand_tensorboard/" ,policy_kwargs=policy_kwargs)
    #model = PPO2.load("PPO_50000")
    model.learn(total_timesteps=100000)
    model.save("DQN_100000_Rdist")
    env.close()


if __name__ == '__main__':
    main()