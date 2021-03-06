########################################
# MARUADER TRAINS TO DEFEAT 11 ZERGLINGS
########################################

import numpy as np
import pysc2
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2 import maps
import sys

import random
import os
from collections import Counter

import numpy as np
import gym
import visdom
import time

from .env_map import EnvMap

class FourTowersSequential(gym.Env):
    """
    This environement can be used to teach an agent to learn how to choose to fight zerglings over banelings.
    The agent has two actions that it can take in the environment and the enviornment's state is a feature
    map of the different agent IDs which have been scaled from 0-1 to act as input to a convolutional neural
    network.
    """
    def __init__(self):
        self.register_map('/maps/','FourTowerSequentialDecomposed')
        self.sc2_env = sc2_env.SC2Env(
          map_name="FourTowerSequentialDecomposed",
          players=[sc2_env.Agent(sc2_env.Race.terran)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64),
              use_feature_units=True),
          step_mul=16,
          game_steps_per_episode=0,
          score_index=0,
          visualize=True)
        self.current_obs = None
        self.actions_taken = 0
        self.last_mineral_count = 0
        self.reward = 0
        self.zergling_count = 0
        self.roach_count = 0
        self.last_reward = 0
        self.last2_reward = 0
        self.rewards = []
        self.decomposed_rewards = []

    def action_space():
        return Discrete(2)

    def observation_space():
        return (84, 84, 1)

    def print_current_alerts(self):
        return self.current_obs[3]

    def store_last_reward(self, reward):
        self.last_reward = reward

    def reset(self):
        # Move the camera in any direction
        # This runs the ResetEpisode trigger built into the map
        self.decomposed_rewards = []
        self.rewards = []
        self.last_timestep = self.sc2_env.reset()
        action = actions.FUNCTIONS.move_camera([0, 0])
        self.last_timestep = self.sc2_env.step([action])[0]
        observation, state, reward, done, info = self.unpack_timestep(self.last_timestep)
        self.current_obs = observation
        state = self.int_map_to_onehot(state)
        state = np.array(state)
        self.actions_taken = 0 
        return state

    def noop(self):
        return actions.FUNCTIONS.no_op()

    def step(self, action):
        done = False
        dead = False

        ### ACTION TAKING ###

        if self.actions_taken == 0 and self.check_action(self.current_obs, 12):
            if action == 0:
                action = actions.FUNCTIONS.Attack_screen("now", [0,0])
            elif action == 1:
                action = actions.FUNCTIONS.Attack_screen("now", [83,0])
            elif action == 2:
                action = actions.FUNCTIONS.Attack_screen("now", [0,83])
            elif action == 3:
                action = actions.FUNCTIONS.Attack_screen("now", [83,83])
            elif action == 4:
                action = actions.FUNCTIONS.no_op()
            else:
                print("Invalid action: check final layer of network")
                action = actions.FUNCTIONS.no_op()
        else:
            action = actions.FUNCTIONS.no_op()

        ####################

        ### STATE PREPARATION ###

        self.last_timestep = self.sc2_env.step([action])[0]

        observation, state, reward, done_null, info = self.unpack_timestep(self.last_timestep)
        self.current_obs = observation

        state = self.int_map_to_onehot(state)
        state = np.array(state)

        #########################

        ### REWARD PREPARATION AND TERMINATION ###

        from s2clientprotocol import sc2api_pb2 as sc_pb
        data = self.sc2_env._controllers[0]._client.send(observation=sc_pb.RequestObservation())
        data = data.observation.raw_data.units

        for x in data:
            if x.unit_type == 1922:
                roach_reward = x.health
            if x.unit_type == 1923:
                zergling_reward = x.health 

        total_reward = roach_reward + zergling_reward - 4
        reward = total_reward
        self.reward = total_reward
        self.rewards.append(reward)

        if self.last_reward != self.reward:
            done = True
            if self.last_reward > self.reward:
                dead = True
            else:
                dead = False

        self.last_reward = self.reward

        self.decomposed_rewards.append([roach_reward - 2, zergling_reward - 2])

        ###########################################

        return state, reward, done, dead, info

    def register_map(self, map_dir, map_name):
        from pysc2.maps import lib
        map_filename = map_name + '.SC2Map'
        class_definition = dict(prefix=map_dir, filename=map_filename, players=1)
        constructed_class = type(map_name, (lib.Map,), class_definition)
        globals()[map_name] = constructed_class

    def get_mineral_count(self, obs):
        return 0

    def get_vespene_gas_count(self, obs):
        return 0

    def unpack_timestep(self, timestep):  
        observation = timestep 
        state = timestep.observation.feature_screen[6]
        reward = timestep.observation.score_cumulative[0]
        done = timestep.last()
        info = {}
        return observation, state, reward, done, info

    def get_available_actions(self, obs):
        return obs.observation.available_actions

    def check_action(self, obs, action):
        return action in self.get_available_actions(obs)

    def get_health_of_unit_type(self, unit_id):
        total_health = 0
        feature_units = self.current_obs[3]['feature_units']
        for unit in feature_units:
            if unit[0] == unit_id:
                total_health += unit[2]
        return total_health

    def get_unit_count_of_type(self, unit_id):
        count = 0
        feature_units = self.current_obs[3]['feature_units']
        for unit in feature_units:
            if unit[0] == unit_id:
                count += 1
        return count

    def int_map_to_onehot(self, x, vocabulary=None):
        location_zergling = np.argwhere(x == 105.)
        location_roach = np.argwhere(x == 110.)

        if location_zergling.shape[0] == 0 or location_roach.shape[0] == 0:
            onehot = [0,0,0,0,0,0,0,0]
        else:
            onehot = [location_zergling[0][0], location_zergling[0][1], location_roach[0][0], location_roach[0][1], location_roach[1][0], location_roach[1][1], location_roach[2][0], location_roach[2][1]]

        onehot = np.reshape(onehot, [1, 8])
        return onehot