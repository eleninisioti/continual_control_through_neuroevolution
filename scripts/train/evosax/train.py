""" Script for training CMA-ES """
import sys
import os
sys.path.append(".")
sys.path.append("methods/evosax_wrapper/") # to be able to import evosax
from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
import os
from scripts.train.base.utils import default_env_params
from scripts.train.evosax.hyperparams import train_gens, hyperparams
import argparse


    
def train_gymnax(num_trials, env_name, population_size, noise_range, optimizer):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["noise_range"] = noise_range

    env_config = {"env_type": "gymnax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  "continual": True}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    if optimizer == "SimpleGA":
        opt_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    elif optimizer == "OpenES":
        opt_kws = {
              "sigma_init": 0.1, "elite_ratio": 0.5} # chanfws dfrom 1
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")
    opt_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    popsize = population_size

    optimizer_name = "SimpleGA"
    optimizer_config = {"optimizer_name": optimizer,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer,
                                             "popsize": popsize,
                                             "es_kws": opt_kws}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
    

    
    
    
def train_minatar(num_trials,  optimizer):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    
    env_name = "asterix_and_breakout"
    
    # configure environment

    env_config = {"env_type": "minatar_multi",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {},
                  "continual": True}
    
    
    # configure method
    num_timesteps = 5000*2*8
    if optimizer == "SimpleGA":
        opt_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    elif optimizer == "OpenES":
        opt_kws = {"temperature": 1.0,
              "sigma_init": 1.0} 
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")
    opt_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    popsize = 256
    
    
    optimizer_config = {"optimizer_name": optimizer,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer,
                                             "popsize": popsize,
                                             "es_kws": {**opt_kws}}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams["Breakout-MinAtar"]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()



def train_classic_control_all(num_trials, optimizer):
       
    train_gymnax(num_trials=num_trials, env_name="CartPole-v1",  population_size=512, noise_range=1.0, optimizer=optimizer)
    train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", population_size=512, noise_range=1.0, optimizer=optimizer)
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",  population_size=512, noise_range=1.0, optimizer=optimizer)

        

def train_kinetix_lifelong(num_trials, optimizer):
    
    # we start from the first task and then the script will go through the rest
    env_names = [
        "m/h0_unicycle",
    ]

    print(f"Starting with environment: {env_name}")
    train_kinetix(num_trials=num_trials, env_name=env_name, optimizer=optimizer)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=10)
    parser.add_argument("--optimizer", type=str, help="Choose between SimpleGA and OpenES", default="SimpleGA")
    args = parser.parse_args()
    
    
    # will train for the lifelong variations of Acrobot, Cartpole, MountainCar 
    train_classic_control_all(num_trials=args.num_trials, optimizer=args.optimizer)

    # will train Minatar (Breakout, Asterix, SpaceInvaders)
    train_minatar(num_trials=args.num_trials, optimizer=args.optimizer)

    # will train Kineitx (medium difficuly tasks))
    train_kinetix_all(num_trials=args.num_trials, optimizer=args.optimizer)
