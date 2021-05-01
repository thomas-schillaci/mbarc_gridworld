import gym
import torch
from enum import IntEnum

from gym.core import Wrapper
from gym.spaces import Box
from gym_minigrid.wrappers import ImgObsWrapper


class RealState(IntEnum):
    EMPTY = 1
    WALL = 2
    BALL = 6
    GOAL = 8


class State(IntEnum):
    AGENT0 = 0
    AGENT1 = 1
    AGENT2 = 2
    AGENT3 = 3
    EMPTY = 4
    WALL = 5
    BALL = 6
    GOAL = 7
    AGENT_ON_GOAL = 8


class CustomWrapper(Wrapper):

    def __init__(self, env, device):
        super().__init__(env)
        self.device = device
        shape = (self.env.grid.width, self.env.grid.height)
        self.observation_space = Box(low=0, high=len(State), shape=shape, dtype='uint8')
        self.states = len(State)

    def wrap_obs(self, obs):
        del obs

        state = torch.empty((self.env.grid.width, self.env.grid.height), dtype=torch.uint8)

        for j in range(self.env.grid.height):
            for i in range(self.env.grid.width):
                c = self.grid.get(i, j)

                if (i, j) == tuple(self.env.agent_pos):
                    on_goal = False
                    if c is not None:
                        if c.type == 'goal':
                            on_goal = True

                    state[i, j] = State.AGENT_ON_GOAL if on_goal else self.env.agent_dir
                    continue

                if c is None:
                    state[i, j] = State.EMPTY
                    continue

                mapping = {
                    'wall': State.WALL,
                    'ball': State.BALL,
                    'goal': State.GOAL
                }

                state[i, j] = mapping[c.type]

        state = state.to(self.device)
        return state

    def wrap_reward(self, reward):
        reward = 1 if reward > 0 else 0
        reward = torch.tensor(reward, dtype=torch.uint8)
        reward = reward.to(self.device)
        return reward

    def reset(self):
        obs = self.env.reset()
        state = self.wrap_obs(obs)
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state = self.wrap_obs(obs)
        reward = self.wrap_reward(reward)
        return state, reward, done, info


def make(env_name, device, seed=None):
    env = gym.make(env_name)
    env = ImgObsWrapper(env)
    env = CustomWrapper(env, device)

    if seed is not None:
        torch.manual_seed(seed)
        env.seed(seed)

    return env
