""" Script for training CMA-ES """
import sys
import os
sys.path.append(".")
sys.path.append("methods/evosax_wrapper/") # to be able to import evosax
import envs

from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
import os
from scripts.train.base.utils import default_env_params
from scripts.train.evosax.hyperparams import train_gens, hyperparams
import argparse
import wandb


def train_ecorobot(num_trials, env_name, robot_type, population_size, optimizer):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials,
                  "trial_index": 1}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["robot_type"] = robot_type
    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  "continual": True}
    
    
    # configure method
    # Robot-specific hyperparameters: ant has larger network (16 hidden vs 4), needs more conservative settings
    if robot_type == "ant":
        # Smaller init range and mutation sigma for stability with larger network
        opt_kws = {"sigma_init": 0.01, "elite_ratio": 0.1, "init_min": -0.1}
    elif robot_type == "halfcheetah":
        # Current working settings for halfcheetah
        opt_kws = {"sigma_init": 1.0, "elite_ratio": 0.5, "init_min": -1.0}
    else:
        # Default fallback
        opt_kws = {"sigma_init": 0.5, "elite_ratio": 0.5, "init_min": -0.5}

    num_timesteps = train_gens[env_name]
    optimizer_config = {"optimizer_name": optimizer,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer,
                                             "popsize": population_size,
                                             "es_kws": opt_kws}}
    
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[(env_name, robot_type)]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

def train_brax(num_trials, env_name,  population_size, optimizer):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials,
                  "trial_index": 1}
    
    # configure environment
    env_params = default_env_params[env_name]

    env_config = {"env_type": "brax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  "continual": True}
    
    
    # configure method
    # Robot-specific hyperparameters: ant has larger network (16 hidden vs 4), needs more conservative settings
    if env_name == "ant":
        # Smaller init range and mutation sigma for stability with larger network
        if optimizer == "SimpleGA":
            opt_kws = {"sigma_init":0.01, "elite_ratio": 0.5, "init_min": -0.1}
        elif optimizer == "OpenES":
            opt_kws = {"sigma_init": 0.05, "sigma_decay": 0.999, "sigma_limit": 0.01, "lrate_init": 0.01, "lrate_decay": 0.999, "lrate_limit": 0.001}

        elif optimizer == "CMA_ES":
            opt_kws= {}
    elif env_name == "halfcheetah":
        # Current working settings for halfcheetah
        opt_kws = {"sigma_init": 1.0, "elite_ratio": 0.5, "init_min": -1.0}
    else:
        # Default fallback
        opt_kws = {"sigma_init": 0.5, "elite_ratio": 0.5, "init_min": -0.5}

    num_timesteps = train_gens[env_name]
    optimizer_config = {"optimizer_name": optimizer,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer,
                                             "popsize": population_size,
                                             "es_kws": opt_kws}}
    
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
def train_gymnax(num_trials, env_name, population_size, noise_range, optimizer, perturbe_every_n_gens):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["noise_range"] = noise_range
    env_params["perturbe_every_n_gens"] = perturbe_every_n_gens
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
    opt_kws = {"sigma_init": 0.5    }
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


def train_ecorobot_all(num_trials, optimizer):
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah", population_size=1024, optimizer=optimizer)
    train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="ant", population_size=512, optimizer=optimizer)


def train_ecorobot_hyperparam_sweep(num_trials, env_name,  population_size, optimizer):
    """
    Hyperparameter sweep version of train_ecorobot using WandB sweeps.
    Sweeps over sigma_init and elite_ratio values.
    """
    # Build base config
    env_params = default_env_params[env_name].copy()
    #env_params["robot_type"] = robot_type
    num_timesteps = train_gens[env_name]
    
    base_config = {
        "env_config": {
            "env_type": "brax",
            "env_name": env_name,
            "curriculum": False,
            "env_params": env_params,
            "continual": True
        },
        "optimizer_config": {
            "optimizer_name": optimizer,
            "optimizer_type": "evosax",
            "optimizer_params": {
                "generations": num_timesteps,
                "strategy": optimizer,
                "popsize": population_size,
                "es_kws": {}  # Will be filled from sweep
            }
        },
        "model_config": {
            "network_type": "MLP",
            "model_params": hyperparams[env_name]
        },
        "exp_config": {
            "seed": 0,
            "num_trials": 1
        }
    }
    
    # Define WandB sweep configuration
    sweep_config = {
        "method": "grid",  # Use grid search to try all combinations
        "metric": {
            "name": "current_best_fitness",
            "goal": "maximize"
        },
        "parameters": {
            "sigma_init": {
                "values": [ 0.2, 0.1, 0.01][::-1]
            },
            "elite_ratio": {
                "values": [ 0.1,0.5]
            },
            "init_min": {
                "values": [  -0.0]
            },
            "trial": {
                "values": [0]
            },
            "activation": {
                "values": ["relu"]
            },
            "popsize": {
                "values": [ 1024]
            },
            "num_layers": {
                "values": [2]
            },
            "cross_over_rate": {
                "values": [0.0]
            }
        }
    }
    
    wandb_project = "ecorobot_hyperparam_sweep"
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=wandb_project)
    
    def train():
        # Initialize wandb for this run
        run = wandb.init(
            project=wandb_project,
            config=base_config
        )
        
        # Update optimizer config with sweep hyperparameters
        base_config["optimizer_config"]["optimizer_params"]["es_kws"]["sigma_init"] = run.config.sigma_init
        base_config["optimizer_config"]["optimizer_params"]["es_kws"]["elite_ratio"] = run.config.elite_ratio
        base_config["optimizer_config"]["optimizer_params"]["es_kws"]["init_min"] = run.config.init_min
        base_config["optimizer_config"]["optimizer_params"]["popsize"] = run.config.popsize
        base_config["model_config"]["model_params"]["activation"] = run.config.activation
        base_config["model_config"]["model_params"]["num_layers"] = run.config.num_layers
        base_config["optimizer_config"]["optimizer_params"]["es_kws"]["cross_over_rate"] = run.config.cross_over_rate
        base_config["exp_config"]["trial_index"]=run.config.trial
        
        print(f"\n{'='*60}")
        print(f"Running sweep trial with sigma_init={run.config.sigma_init}, elite_ratio={run.config.elite_ratio}, init_min={run.config.init_min}")
        print(f"{'='*60}\n")
        
        try:
            exp = Experiment(
                env_config=base_config["env_config"],
                optimizer_config=base_config["optimizer_config"],
                model_config=base_config["model_config"],
                exp_config=base_config["exp_config"]
            )
            
            # Run the experiment
            exp.run()
        except Exception as e:
            print(f"Error in sweep trial: {e}")
            raise
    
    # Run the sweep
    print(f"\n{'='*60}")
    print(f"Starting WandB hyperparameter sweep")
    print(f"Each combination will run {num_trials} trials")
    print(f"Sweep ID: {sweep_id}")
    print(f"{'='*60}\n")
    
    wandb.agent(sweep_id, function=train)
   
def train_classic_control_all(num_trials, optimizer):
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1", population_size=512, noise_range=1.0, optimizer=optimizer)

    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1",  population_size=512, noise_range=1.0, optimizer=optimizer)
    #train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", population_size=512, noise_range=1.0, optimizer=optimizer)
    #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",  population_size=512, noise_range=1.0, optimizer=optimizer)


    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1",  population_size=512, noise_range=0.0, optimizer=optimizer)
    #train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", population_size=512, noise_range=0.0, optimizer=optimizer)
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",  population_size=512, noise_range=0.0, optimizer=optimizer)
    
    
def train_classic_control_parameteric(num_trials, optimizer):
    n_gens = [5,10, 20, 50, 100, 200][::-1]
    #n_gens = [200]
    n_gens = [5,200]
    for perturbe_every_n_gens in n_gens:
        train_gymnax(num_trials=num_trials, env_name="CartPole-v1",  population_size=512, noise_range=1.0, optimizer=optimizer, perturbe_every_n_gens=perturbe_every_n_gens)
        train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", population_size=512, noise_range=1.0, optimizer=optimizer, perturbe_every_n_gens=perturbe_every_n_gens)
        #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",  population_size=512, noise_range=1.0, optimizer=optimizer, perturbe_every_n_gens=perturbe_every_n_gens)
        
        

def train_kinetix_lifelong(num_trials, optimizer):
    
    # we start from the first task and then the script will go through the rest
    env_names = [
        "m/h0_unicycle",
    ]

    env_name = env_names[0]  # Use the first environment
    print(f"Starting with environment: {env_name}")
    # train_kinetix(num_trials=num_trials, env_name=env_name, optimizer=optimizer)  # Commented out since function doesn't exist
    

def train_brax_all(num_trials, optimizer):
    train_brax(num_trials=num_trials, env_name="ant",population_size=512, optimizer=optimizer)
    train_brax(num_trials=num_trials, env_name="halfcheetah",population_size=512, optimizer=optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=1)
    parser.add_argument("--optimizer", type=str, help="Choose between SimpleGA and OpenES", default="SimpleGA")
    args = parser.parse_args()
    
    #train_brax(num_trials=args.num_trials, env_name="ant",population_size=512, optimizer=args.optimizer)
    #train_brax(num_trials=args.num_trials, env_name="ant",population_size=1024, optimizer="OpenES")
    #train_brax(num_trials=args.num_trials, env_name="ant",population_size=512, optimizer="CMA_ES")


    #train_ecorobot_all(num_trials=args.num_trials, optimizer=args.optimizer)
    #train_ecorobot_hyperparam_sweep(num_trials=args.num_trials, env_name="ant", population_size=1024, optimizer=args.optimizer)
    #quit()
    #train_ecorobot_hyperparam_sweep(num_trials=args.num_trials, env_name="locomotion", robot_type="halfcheetah", population_size=512, optimizer=args.optimizer)
    
    
     #train_classic_control_parameteric(num_trials=args.num_trials, optimizer=args.optimizer)
    
    # will train for the lifelong variations of Acrobot, Cartpole, MountainCar 
    #train_classic_control_all(num_trials=args.num_trials, optimizer=args.optimizer)
    
    #train_ecorobot_all(num_trials=args.num_trials, optimizer=args.optimizer)

    # will train Minatar (Breakout, Asterix, SpaceInvaders)
    #train_minatar(num_trials=args.num_trials, optimizer=args.optimizer)

    # will train Kineitx (medium difficuly tasks))
    #train_kinetix_lifelong(num_trials=args.num_trials, optimizer=args.optimizer)
