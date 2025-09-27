""" Script for training CMA-ES """
import sys
import os
sys.path.append(".")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, "methods")
sys.path.insert(0, "methods/evosax_wrapper") # to be able to import evosax
sys.path.insert(0, "scripts")
sys.path.insert(0, "methods/Kinetix")

from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
import os
import envs
from scripts.train.base.utils import default_env_params
from scripts.train.evosax.openes.hyperparams import train_gens, hyperparams
import argparse



def train_stepping_gates(num_trials, env_name, curriculum):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["episode_type"] = "full"
    env_params["curriculum"] = curriculum
    env_config = {"env_type": "stepping_gates",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    optimizer_config = {"optimizer_name": "cma_es",
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": "CMA_ES",
                                             "popsize": 256}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

def train_kinetix(num_trials, env_name,  optimizer_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    
    ga_kws = {"sigma_init": 0.05, "elite_ratio":0.5}
    es_kws = {
              "sigma_init": 0.1, "elite_ratio": 0.5} # chanfws dfrom 1
    #optimizer_name = "CMA_ES"
    popsize = 1024
    if optimizer_name == "CMA_ES":
        opt_kws = es_kws
        
    elif optimizer_name == "OpenES":
        opt_kws = {"sigma_init": 0.3}
        opt_kws =   {     "sigma_init": 0.05, "sigma_decay": 0.999, "sigma_limit": 0.01,      "lrate_init": 0.01,
        "lrate_decay": 0.999,
        "lrate_limit": 0.001
    }
        #opt_kws =   {     "sigma_init": 0.03, "sigma_decay": 0.999, "sigma_limit": 0.01,      "lrate_init": 0.005,
        #"lrate_decay": 0.999,
        #"lrate_limit": 0.0005
    #}
    else:
        opt_kws = ga_kws
        popsize= 1024

    
    # configure environment
    env_params = default_env_params["kinetix"]
    env_params["episode_type"] = "full"
    env_params["curriculum"] = False
    env_config = {"env_type": "kinetix",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {}}
    
    
    # configure method
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

def train_ecorobot(num_trials, env_name, robot_type, continual):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    optimizer_name = "OpenES"
    
    
    #optimizer_name = "CMA_ES"
    if robot_type == "halfcheetah":
        popsize = 1024
        opt_kws = {"sigma_init": 0.05,
                   "sigma_decay": 0.999, "sigma_limit": 0.01, "lrate_init": 0.01,
                   "lrate_decay": 0.999, "lrate_limit": 0.001}
    elif robot_type == "ant":
        popsize = 1024
        opt_kws = {"sigma_init": 0.05,
                   "sigma_decay": 0.999, "sigma_limit": 0.01, "lrate_init": 0.01,
                   "lrate_decay": 0.999, "lrate_limit": 0.001}



    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["episode_type"] = "full"
    env_params["curriculum"] = False
    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {"robot_type": robot_type},
                  "continual": continual}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    optimizer_config = {"optimizer_name": optimizer_name,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": popsize,
                                             "es_kws": opt_kws}}
    
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[(env_name, robot_type)]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
def train_gymnax(num_trials, env_name, optimizer_name, continual):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["noise_range"] = 1.0
    env_config = {"env_type": "gymnax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  "continual": continual}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    #optimizer_name = "SimpleGA"
    ga_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    es_kws = {
              "sigma_init": 0.1, "elite_ratio": 0.5} # chanfws dfrom 1
    #optimizer_name = "CMA_ES"
    popsize = 512
    if optimizer_name == "CMA_ES":
        opt_kws = es_kws
        
    elif optimizer_name == "OpenES":
        opt_kws = {"sigma_init": 0.3}
    else:
        opt_kws = ga_kws
    optimizer_config = {"optimizer_name": optimizer_name,
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
    
    
    
def train_minatar_multi(num_trials, optimizer_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    
    env_name = "asterix_and_breakout"
    
    # configure environment

    env_config = {"env_type": "minatar_multi",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {}}
    
    
    # configure method
    num_timesteps = 5000*2*8
   # optimizer_name = "SimpleGA"
    ga_kws = {"sigma_init": 0.5, "elite_ratio":0.5}
    es_kws = {"temperature": 1.0,
              "sigma_init": 1.0}
    
    optimizer_config = {"optimizer_name": optimizer_name,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": 256,
                                             "es_kws": {**es_kws}}}
    
    model_config = {"network_type": "AtariCNN",
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
    

def train_kinetix_all(num_trials, optimizer_name):
    # Run all kinetix environments from h0 to h19
    env_names = [
        "s/h0_unicycle",
        "s/h1_car_left", 
        "s/h2_car_right",
        "s/h3_car_thrust",
        "s/h4_thrust_the_needle",
        "s/h5_angry_birds",
        "s/h6_thrust_over",
        "s/h7_car_flip",
        "s/h8_weird_vehicle",
        "s/h9_spin_the_right_way",
        "s/h10_thrust_right_easy",
        "s/h11_thrust_left_easy",
        "s/h12_thrustfall_left",
        "s/h13_thrustfall_right",
        "s/h14_thrustblock",
        "s/h15_thrustshoot",
        "s/h16_thrustcontrol_right",
        "s/h17_thrustcontrol_left",
        "s/h18_thrust_right_very_easy",
        "s/h19_thrust_left_very_easy"
    ]
    env_names = ["s/h0_weak_thrust"]
    
    for env_name in env_names:
        print(f"Training on environment: {env_name}")
        train_kinetix(num_trials=num_trials, env_name=env_name, optimizer_name=optimizer_name)

def train_ecorobot_all(num_trials):
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah", continual=False)
    train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="ant", continual=False)

    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="ant")
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah", continual=True)
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="ant", continual=True)

    #train_ecorobot(num_trials=num_trials, env_name="maze_with_stepping_stones", robot_type="discrete_fish")

    #train_ecorobot(num_trials=num_trials, env_name="locomotion_with_obstacles", robot_type="halfcheetah")
    #train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="discrete_fish")
    #train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="ant")

def train_gymnax_all(num_trials, optimizer_name):
    #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0", optimizer_name=optimizer_name,continual=True)

    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0", optimizer_name=optimizer_name, continual=True)
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1", optimizer_name=optimizer_name, continual=False)
    #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0")
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1")
    #train_gymnax(num_trials=num_trials, env_name="MountainCarContinuous-v0")
    #train_gymnax(num_trials=num_trials, env_name="Breakout-MinAtar")
    #train_gymnax(num_trials=num_trials, env_name="Asterix-MinAtar")
    #train_gymnax(num_trials=num_trials, env_name="Freeway-MinAtar")
    #train_gymnax(num_trials=num_trials, env_name="SpaceInvaders-MinAtar")
    #train_gymnax(num_trials=num_trials, env_name="Pong-MinAtar")
    pass


def train_craftax_all(num_trials):
    train_craftax(num_trials=num_trials, env_name="craftax")




    


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
