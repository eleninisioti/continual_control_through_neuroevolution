""" Script for training CMA-ES """
import sys
import os
sys.path.append(".")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..x    ')))
sys.path.insert(0, "methods")
sys.path.insert(0, "methods/evosax_wrapper") # to be able to import evosax
sys.path.insert(0, "scripts")
from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
import os
import envs
from scripts.train.base.utils import default_env_params
from scripts.train.evosax.simplega.hyperparams import train_gens, hyperparams
import argparse



    
def train_gymnax(num_trials, env_name, continual, population_size, noise_range):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    if continual:
        env_params["noise_range"] = noise_range
    else:
        env_params["noise_range"] = 0.0
    env_config = {"env_type": "gymnax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  "continual": continual}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    optimizer_name = "SimpleGA"
    opt_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    popsize = population_size

    optimizer_name = "SimpleGA"
    optimizer_config = {"optimizer_name": "SimpleGA",
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": popsize,
                                             "es_kws": opt_kws}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
    
def train_craftax(num_trials, env_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_config = {"env_type": "craftax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    optimizer_name = "SimpleGA"
    ga_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    optimizer_config = {"optimizer_name": optimizer_name,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": 256,
                                             "es_kws": {}}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
    
    
def train_minatar_multi(num_trials, continual=False):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    
    env_name = "asterix_and_breakout"
    
    # configure environment

    env_config = {"env_type": "minatar_multi",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {},
                  "continual": False}
    
    
    # configure method
    num_timesteps = 5000*2*8
    num_timesteps = 5000
   # optimizer_name = "SimpleGA"
    opt_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    #es_kws = {"temperature": 1.0,
    #          "sigma_init": 1.0}
    popsize = 256
    
    optimizer_name = "SimpleGA"
    
    optimizer_config = {"optimizer_name": optimizer_name,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": popsize,
                                             "es_kws": {**opt_kws}}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams["Breakout-MinAtar"]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

def train_stepping_gates_all(num_trials):
    train_stepping_gates(num_trials=num_trials, env_name="n_parity", curriculum=False)
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=True)
    train_stepping_gates(num_trials=num_trials, env_name="simple_alu", curriculum=True)
    


def train_ecorobot_all(num_trials):
    train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah", continual=False)
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah", continual=True)

    #train_ecorobot(num_trials=num_trials, env_name="maze_with_stepping_stones", robot_type="discrete_fish")

    #train_ecorobot(num_trials=num_trials, env_name="locomotion_with_obstacles", robot_type="halfcheetah")
    #train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="discrete_fish")
    #train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="ant")

def train_classic_control_lifelong(num_trials):
   
    
    train_gymnax(num_trials=num_trials, env_name="CartPole-v1", continual=True, population_size=512, noise_range=1.0)
    train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", continual=True, population_size=512, noise_range=1.0)
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0", continual=True, population_size=512, noise_range=1.0)

       
   

        


def train_kinetix(num_trials, env_name:

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
   
    # configure environment
    env_params = default_env_params["kinetix"]
    env_params["episode_type"] = "full"
    env_params["curriculum"] = False
    env_config = {"env_type": "kinetix",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {}}
    
    
    # configure optimizer and architecutre
    optimizer_name = "SimpleGA"
    opt_kws = {"sigma_init": 0.001, "elite_ratio":0.5}
    popsize = 1024
    num_timesteps = train_gens["kinetix"]
    optimizer_config = {"optimizer_name": optimizer_name,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": popsize,
                                             "es_kws": opt_kws}}
    
    
    model_config = {"network_type": "kinetix",
                    "model_params": hyperparams["kinetix"]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

def train_kinetix_lifelong(num_trials):
    
    # we start from the first task and then the script will go through the rest
    env_names = [
        "m/h0_unicycle",
    ]

    print(f"Starting with environment: {env_name}")
    train_kinetix(num_trials=num_trials, env_name=env_name)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=10)
    args = parser.parse_args()
    
    
    # will train for the lifelong variations of Acrobot, Cartpole, MountainCar 
    train_classic_control_lifelong(num_trials=args.num_trials)

    # will train Minatar (Breakout, Asterix, SpaceInvaders)
    train_minatar_lifelong(num_trials=args.num_trials)

    # will train Kineitx (medium difficuly tasks))
    train_kinetix_lifelong(num_trials=args.num_trials)
