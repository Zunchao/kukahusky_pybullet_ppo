from ray.tune.registry import register_env
from kukahusky_pybullet_ppo.env.mmKukaHuskyGymEnv import MMKukaHuskyGymEnv
import argparse, ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, Model
import tensorflow as tf

class MMKukaHuskyModelClass(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Define the layers of a custom model.
        init_w = tf.contrib.layers.xavier_initializer()
        init_b = tf.constant_initializer(0.001)
        layer1 = tf.layers.dense(input_dict["obs"], 200, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b)
        layer2 = tf.layers.dense(layer1, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b)
        layerout = tf.layers.dense(layer2, num_outputs, activation=tf.nn.relu)

        return layerout, layer2

def env_creator(env_config):
    env = MMKukaHuskyGymEnv(renders=False, isDiscrete=False, action_dim = 9, rewardtype='rdense', randomInitial=False)
    return env  # return an env instance

def train():
    ModelCatalog.register_custom_model("mmkukahusky_model", MMKukaHuskyModelClass)
    ray.init()
    register_env("Mmkukahusky-v0", env_creator)

    # Can optionally call agent.restore(path) to load a checkpoint.
    config = ppo.DEFAULT_CONFIG.copy()

    ray.tune.run_experiments({
        "my_experiment": {
            "run": "PPO",
            "env": "Mmkukahusky-v0",
            "stop": {"episode_reward_mean": 0},
            "config": {
                'env_config': config,
                'model': "mmkukahusky_model",
                "num_gpus": 0,
                "num_workers": 0,
            },
        },
    })
    #config['monitor']=False
    #config["num_gpus"] = 0
    #config["num_workers"] = 0
    #config['model']="mmkukahusky_model"
    #print('config',config)
    #agent = ppo.PPOAgent(config=config, env="Mmkukahusky-v0")
    #for i in range(1000000):
    # Perform one iteration of training the policy with PPO
        #result = agent.train()
        #print(pretty_print(result))

    #if i % 10000 == 0:
       #checkpoint = agent.save()
       #print("checkpoint saved at", checkpoint)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()