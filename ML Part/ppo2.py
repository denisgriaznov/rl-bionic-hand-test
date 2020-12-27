import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def main():
    unity_env = UnityEnvironment("../ArmRobot/Build/ArmRobot",worker_id=1)
    env = UnityToGymWrapper(unity_env, True)
    policy_kwargs = dict(net_arch=[128, 128, 64, 32])
    model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./hand_tensorboard/" ,policy_kwargs=policy_kwargs)
    #model = PPO2.load("PPO_50000")
    model.learn(total_timesteps=100000)
    model.save("PPO2_100000_R1_033")
    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            print(action.shape)
            #action = action.reshape((15,))
            obs, rewards, done, info = env.step(action)
            env.render()
            print(action)
            print(obs)

    env.close()


if __name__ == '__main__':
    main()