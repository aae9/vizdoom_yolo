import VizDoomFunctions as vdf
import VizDoomSetups as vds
import gymnasium as gm
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class DoomEnv(gm.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
        self.game = vds.rl_environment()
        self.actions = [[3],[4],[5],[6], [3,5], [3,6], [4,5], [4,6]]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        #obs = None
        obs = self.get_obs() 
        return obs, {}
    def step(self, action):

        reward = self.game.make_action(self.actions[action], 4)

        done = self.game.is_episode_finished()

        if done:
            obs = np.zeros((84, 84, 1), dtype=np.uint8)
        else:
            obs = self.get_obs()

        return obs, reward, done, False, {}
    def get_obs(self):
        state = vdf.get_state(self.game)
        print("State", state)
        return state
    def render(self):
        pass
    def close(self):
        self.game.close()

env = DummyVecEnv([lambda: DoomEnv()])

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01
)

model.learn(total_timesteps=1000000)
model.save("ppo_doom_model")