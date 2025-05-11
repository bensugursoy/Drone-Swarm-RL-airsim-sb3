import yaml
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.base_env.drones import RLlibDrones

if __name__ == "__main__":
    # register new env
    ENV_REGISTRY["drones"] = RLlibDrones
    # initialize env
    env = marl.make_env(environment_name = "drones", 
                        map_name = "template")
    # pick mappo algorithms
    ippo = marl.algos.ippo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, ippo, {"core_arch": "mlp"}) #"core_arch": "mlp", "encode_layer": "128-256"
    # start learning
    ippo.fit(env, model, stop={'episode_reward_mean': 550}, num_gpus=1, #, 'timesteps_total': 3
              num_workers=0, share_policy='individual', checkpoint_freq=500, local_mode=False, checkpoint_end=True) #local_mode=True, checkpoint_end=True