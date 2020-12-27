import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np
import pandas as pd
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def main():
    unity_env = UnityEnvironment("../ArmRobot/Build/ArmRobot",worker_id=1)
    env = UnityToGymWrapper(unity_env, True)
    models = [PPO2.load("PPO2_100000_R1_025") , PPO2.load("PPO2_100000_R1_02") , PPO2.load("PPO2_100000_R1_033")]
    names = ["1 ","2 ","3 "]
    df = pd.DataFrame()
    for model,name in zip(models,names):
        first = np.zeros((100))
        second = np.zeros((100))
        for i in range(100):
            obs = env.reset()
            done = False
            obs_story = [None, None]

            while not done:
                action, _states = model.predict(obs)
                #action = action.reshape((15,))
                obs, rewards, done, info = env.step(action)
                env.render()
                obs_story[0] = obs_story[1]
                obs_story[1] = obs
                if rewards > 0:
                    f = np.exp(-np.abs(np.min(obs_story[0][:4])))
                    s = np.exp(-np.abs(np.min(obs_story[0][4])))
                    #print(f)
                    #print(s)
                    first[i] = f
                    second[i] = s
                    #print(first)
                    #print(second)
        df[name+"up"] = first
        df[name+"down"] = second
    env.close()
    #df.to_csv("comparison.csv")

if __name__ == '__main__':
    main()