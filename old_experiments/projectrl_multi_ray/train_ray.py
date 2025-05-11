import time
import random
import ray
import gym
import numpy as np
import yaml

from ray import tune
from ray.tune import register_env

from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv

import ray.rllib.agents.ppo as PPO
from ray.rllib.models import ModelCatalog
#from ray.rllib.agents.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from scripts.airsim_env_rll import AirSimDroneEnv
from scripts.naturecnn import NatureCNN

if __name__ == "__main__":

    # Load train environment configs
    with open('scripts/env_config.yml', 'r') as f:
        env_config = yaml.safe_load(f)

    # Load inference configs
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Determine input image shape
    image_shape = (84,84,1) if config["train_mode"]=="depth" else (84,84,3)

    print("Initializing Ray")
    ray.shutdown()
    ray.init(ignore_reinit_error=True) #, num_gpus=1
    #print("Dashboard URL: http://{}".format(ray.get_webui_url()))

    print("Registering the environment")

    # Defining obs space
    if config["train_mode"] == "multi_rgb":
        observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(image_shape[0],image_shape[1]*3,1), 
            dtype=np.uint8)
    else:
        observation_space = gym.spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8)

    # Defining actions space
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) #3.0


    def env_creator(env_config):
        o = observation_space
        a = action_space
        max_conc = 3
        return AirSimDroneEnv(env_config=env_config,
                              observation_space=o,
                              action_space=a, 
                              max_concurrent=max_conc,
                              ip_address="127.0.0.1", 
                              image_shape=image_shape,
                              input_mode=config["train_mode"],
                              num_drones=config["num_drones"])
    
    register_env("airsim-env-v0", env_creator) #or tune.register_env("ext-ma-airsim", lambda env_config: ExternalAirSimMultiAgent(env_config,o,a,max_conc))

    print("Training")
    stop = {
        "episode_reward_mean": 130,
        "timesteps_total": 3500000,
    }

    obs_space_dict = {
        "agent_1": observation_space,
        "agent_2": observation_space,
        "agent_3": observation_space,
    }
    act_space_dict = {
        "agent_1": action_space,
        "agent_2": action_space,
        "agent_3": action_space,
    }

    o_s = observation_space
    a_s = action_space    

    ModelCatalog.register_custom_model("NatureCNN", NatureCNN)

    config = {
        "env": "airsim-env-v0",
        "env_config": {
            "actions_are_logits": False,
        },
        "multiagent": {
            "policies": {
                "pol1": (None, o_s, a_s, {"agent_id": 0,}),
                "pol2": (None, o_s, a_s, {"agent_id": 1,}),
                "pol3": (None, o_s, a_s, {"agent_id": 2,}),
            },
            "policy_mapping_fn": lambda agent_id: "pol1" if agent_id == 0 
                                                         else ("pol2" if agent_id == 1 
                                                         else "pol3"),
        },
        "framework": "torch",
        #"num_gpus": 1,
        "num_workers": 0,
        "evaluation_interval": 4000,
        "evaluation_num_episodes": 20,
        #"lr": 0.00001,
        "clip_param": 0.3,
        #"grad_clip": 4,
        # "model": {
        #     "custom_model": NatureCNN,
        # },
    }

    #config["enable_connectors"]=False

    tune.run("PPO",stop=stop, config=config, verbose=1)
    ray.shutdown()