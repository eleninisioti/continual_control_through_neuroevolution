
import functools
import os
import pickle
import yaml
from scripts.train.rl.ppo.hyperparams import hyperparams
from scripts.train.base.experiment import Experiment
from functools import partial
import numpy as onp
import jax.numpy as jnp
import jax
from scripts.train.base.visuals import viz_histogram, viz_heatmap
from scripts.train.rl.ppo.hyperparams import hyperparams
from methods.evosax_wrapper.base.tasks.rl import EcorobotTask
from methods.evosax_wrapper.direct_encodings.model import make_model
from methods.evosax_wrapper.base.training.evolution import EvosaxTrainer
from methods.evosax_wrapper.base.training.logging  import Logger
import equinox as eqx
import evosax
from methods.evosax_wrapper.base.tasks.rl import GatesTask
from methods.evosax_wrapper.base.tasks.rl import GymnaxTask, GymnaxTaskWithPerturbation, MinatarMultiTask, CraftaxTask, KinetixTask
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset
from methods.evosax_wrapper.base.tasks.rl import CraftaxState
import wandb
import gymnax
from kinetix.environment.env import make_kinetix_env
from kinetix.util import generate_params_from_config
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.environment.utils import ActionType, ObservationType
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.util.config import normalise_config
from flax.serialization import to_state_dict

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class EvosaxExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)
        

    def setup_trial_keys(self):
        key = jax.random.PRNGKey(self.config["exp_config"]["trial_seed"])

        self.model_key, self.train_key = jax.random.split(key, 2)
        
    def init_model(self):
      self.model = make_model(self.config, self.model_key, env=self.env,
                              env_params=self.config["env_config"]["gymnax_env_params"],
                              kinetix_config=self.config["env_config"]["kinetix_config"])

    def cleanup(self):
        pass
    
    
    
    def setup_stepping_gates_env(self):
        
        self.env = stepping_gates_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks
        
    def setup_ecorobot_env(self):
        self.env = ecorobot_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks
        self.config["env_config"]["gymnax_env_params"] = None
        self.config["env_config"]["kinetix_config"] = None
        self.for_eval = None

        
        
        
    def setup_kinetix_env(self):
        with open("scripts/train/evosax/kinetix_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
               
        config = normalise_config(config, name="PPO")
        self.config["env_config"]["kinetix_config"] = config

        #observation_type = ObservationType.from_string(config["observation_type"])
        #action_type = ActionType.from_string(config["action_type"])
        env_params, static_env_params = generate_params_from_config(config)
        config["env_params"] = to_state_dict(env_params)
        config["static_env_params"] = to_state_dict(static_env_params)
        
        self.for_eval = {"env_params": env_params, "static_env_params": static_env_params}

        reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)
        self.env = make_kinetix_env(
                                                observation_type=config["observation_type"],
                                                action_type=config["action_type"],
                                                reset_fn=reset_fn,
                                                env_params=env_params,
                                                static_env_params=static_env_params)
        #env_params["noise"] = 2.0

        #if self.config["env_config"]["env_params"]:
        #    env_params = env_params.replace(**self.config["env_config"]["env_params"])
        self.config["env_config"]["gymnax_env_params"] = env_params

        self.config["env_config"]["action_size"] = 20
        obs_size = 20 # random value
        self.config["env_config"]["observation_size"] = obs_size

        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["episode_length"] = 1000
        
        
        
        
        
    def setup_craftax_env(self):
        self.env = CraftaxSymbolicEnvNoAutoReset()
        env_params = self.env.default_params
        self.config["env_config"]["gymnax_env_params"] = env_params

        self.config["env_config"]["observation_size"] = self.env.observation_space(env_params).shape[0]
        print(self.config["env_config"]["observation_size"])
        self.config["env_config"]["action_size"] = self.env.num_actions
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["num_tasks"] = 1
        
        
    def setup_minatar_multienv(self):
        self.config["env_config"]["gymnax_env_params"] = []
        action_size = []
        obs_size = []
        
        if self.config["env_config"]["env_name"] == "asterix_and_breakout":
            #env_names = ["Breakout-MinAtar", "Asterix-MinAtar", "SpaceInvaders-MinAtar", "Freeway-MinAtar"]
           # env_names = [ "SpaceInvaders-MinAtar",   "Breakout-MinAtar"]
            env_names = [ "Breakout-MinAtar", "Asterix-MinAtar", "SpaceInvaders-MinAtar"]
            env_names = ["Breakout-MinAtar"]



        obs_sizes = []
        action_sizes = []
        self.env = []
        for env_name in env_names:
            env, env_params = gymnax.make(env_id=env_name)
            #if self.config["env_config"]["env_params"]:
            #    env_params = env_params.replace(**self.config["env_config"]["env_params"])
            self.config["env_config"]["gymnax_env_params"].append(env_params)
            obs_size =  env.obs_shape
            obs_sizes.append(obs_size[-1])
            action_sizes.append(env.num_actions)
            print(obs_size)
            print(env.num_actions)
            self.env.append(env_name)
            
        action_size = max(action_sizes)
        obs_size = max(obs_sizes)
        

        
            
        self.config["env_config"]["action_size"] = action_size
        self.config["env_config"]["observation_size"] = obs_size
        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["gymnax_env_params"] = None
        self.config["env_config"]["kinetix_config"] = None

        
        
        
        
        
        
        
    def setup_gymnax_env(self):
        self.env, env_params = gymnax.make(env_id=self.config["env_config"]["env_name"])
        
        self.for_eval = {}
        #env_params["noise"] = 2.0

        #if self.config["env_config"]["env_params"]:
        #    env_params = env_params.replace(**self.config["env_config"]["env_params"])
        self.config["env_config"]["gymnax_env_params"] = env_params

        self.config["env_config"]["action_size"] = self.env.num_actions
        if  "MountainCar" in self.config["env_config"]["env_name"]:
            obs_size = 2
        elif "MinAtar" in self.config["env_config"]["env_name"]:
            obs_size =  self.env.obs_shape[-1]
        else:
            obs_size = self.env.obs_shape[0]
        self.config["env_config"]["observation_size"] = obs_size

        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["episode_length"] = self.config["env_config"]["env_params"]["max_steps_in_episode"]
        self.config["env_config"]["kinetix_config"] = None
        
        
        
        






    def metrics_fn(self, log_info,  data, episode_length, task_params, num_nodes, num_edges, noise, current_gravity, n_dormant, diversity):


        def callback(log_info, episode_length,task_params, data, num_nodes, num_edge, noise, current_gravity, n_dormant, diversity      ):
            best_indiv = int(onp.argmax(onp.array(data["fitness"])))

            log_info = {
                "current_best_fitness": onp.max(onp.array(data["fitness"])),
                                "mean_fitness": onp.mean(onp.array(data["fitness"])),

                "mean episode_length": onp.mean(onp.mean(onp.array(episode_length), axis=1)),
                "max episode_length": onp.max(onp.mean(onp.array(episode_length), axis=1)),
                "min episode_length": onp.min(onp.mean(onp.array(episode_length), axis=1)),
                "best indiv episode length": onp.mean(onp.array(episode_length)[best_indiv]),


                "generation": log_info.gen_counter,
                "current_task": task_params,
                "current_gravity": current_gravity,
                "diversity": diversity,
                #"navigability": log_info.navig,
                #"navigability_online": log_info.navig_online,
                #"robustness_fitness": log_info.robustness_fitness,
                #"robustness": log_info.robustness,
                "num_nodes": num_nodes,
                "num_edges": 0,
                "noise": onp.array(noise[0]),
                "mean_n_dormant": onp.mean(onp.array(n_dormant)),
                "max_n_dormant": onp.max(onp.array(n_dormant)),
                "min_n_dormant": onp.min(onp.array(n_dormant)),
                "best_indiv_n_dormant": onp.array(n_dormant)[best_indiv]

            }
            
            max_level = 0
            mean_level = 0 
            
            if "info" in data["data"]:
                for key, value in data["data"]["info"].items():
                    log_info["mean_" + key] = jnp.mean(value)
                    log_info["max_" + key] = jnp.max(value)
                    
                    if "enter_dungeon" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 1
                        
                    if "enter_gnomish_mines" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 2
                            
                    if "enter_sewers" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 3
                            
                    if "enter_vault" in key:
                        best_value = int(jnp.max(value))

                            
                    if "enter_troll_mines" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 4                    
                
                        
                    
            log_info["deepest_level"] = max_level

            wandb.log(log_info)
            
            for key, value in log_info.items():
                if "mean_" not in key and "max_" not in key:
                    print(key, value)
                else:
                    if value > 0.0:
                        print(key, value)

        jax.debug.callback(callback, log_info, episode_length,task_params, data, num_nodes, num_edges, noise, current_gravity, n_dormant, diversity)

    def eval_task(self, best_member, tasks, gens, final_policy=False):

        policy_params = self.params_shaper.reshape_single(best_member)

        policy = eqx.combine(policy_params, self.statics)


        init_policy_state, _ = policy.initialize(jax.random.PRNGKey(0))

        if self.config["env_config"]["env_type"] == "kinetix":
            act_fn = partial(policy,
                             key=self.model_key,
                             state=init_policy_state,
                             obs_size=self.config["env_config"]["observation_size"],
                             action_size=self.config["env_config"]["action_size"])

            
        else:

            act_fn = partial(policy, key=self.model_key, state=init_policy_state, obs_size=self.config["env_config"]["observation_size"], action_size=self.config["env_config"]["action_size"])


        super().run_eval(act_fn, tasks, final_policy=final_policy, gens=gens, for_eval=self.for_eval)

   
    def get_final_policy(self):
        policy_params = self.params_shaper.reshape_single(self.final_state["params"])

        policy = eqx.combine(policy_params, self.statics)

        init_policy_state, dev_states = policy.initialize(jax.random.PRNGKey(0))
        dev_steps = self.config["model_config"]["model_params"]["max_dev_steps"] + 2
        data = jax.tree_map(lambda x: x[dev_steps, ...], dev_states)
        return data.weights

    def train_trial(self):

        def data_fn(data: dict):
            return {}

        logger = Logger(True,
                        metrics_fn=self.metrics_fn,
                        ckpt_freq=100,
                        aim_freq=1,
                        ckpt_dir=self.config["exp_config"]["trial_dir"] + "/data/train")

        fitness_shaper = evosax.FitnessShaper(maximize=True,
                                              centered_rank=False)

        #phenotype_size = (self.model.max_nodes, self.model.max_nodes)
        params, statics = eqx.partition(self.model, eqx.is_array)
        self.statics = statics
        self.params_shaper = evosax.ParameterReshaper(params)

    
        """
        self.env = GatesTask(statics,

                        env=self.config["env_config"]["env_name"],
                        max_steps= self.config["env_config"]["episode_length"],
                        data_fn=data_fn, env_kwargs={**self.config["env_config"]["env_params"]})
        """
        
        if self.config["env_config"]["env_type"] == "ecorobot":
        
            self.env = EcorobotTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=500,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
        elif self.config["env_config"]["env_type"] == "gymnax":
            self.env = GymnaxTaskWithPerturbation(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=self.config["env_config"]["episode_length"],
                                obs_size=self.config["env_config"]["observation_size"],
                                action_size=self.config["env_config"]["action_size"],
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
        elif self.config["env_config"]["env_type"] == "minatar_multi":
            self.env = MinatarMultiTask(statics=self.statics,
                                env=self.env,
                                max_steps=1000,
                                obs_size=self.config["env_config"]["observation_size"],
                                action_size=self.config["env_config"]["action_size"],
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
            
        elif self.config["env_config"]["env_type"] == "craftax":
            self.env = CraftaxTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=1000,
                                obs_size=self.config["env_config"]["observation_size"],
                                action_size=self.config["env_config"]["action_size"],
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
            
        elif self.config["env_config"]["env_type"] == "kinetix":
            self.env = KinetixTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=256,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
        else:
            self.env = GatesTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=1000,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
       



        trainer = EvosaxTrainer(train_steps=self.config["optimizer_config"]["optimizer_params"]["generations"],
                                task=self.env,
                                save_params_fn=self.save_params,
                                perturbe_every_n_gens=self.config["env_config"]["env_params"]["perturbe_every_n_gens"],
                                strategy=self.config["optimizer_config"]["optimizer_params"]["strategy"],
                                params_shaper=self.params_shaper,
                                popsize=self.config["optimizer_config"]["optimizer_params"]["popsize"],
                                fitness_shaper=fitness_shaper,
                                num_tasks = self.env.num_tasks,
                                reward_for_solved=self.env.reward_for_solved,
                                noise_range=self.config["env_config"]["env_params"]["noise_range"],
                                # sigma_init = 0.01,
                                es_kws={**self.config["optimizer_config"]["optimizer_params"]["es_kws"]
                                        },
                                logger=logger,
                                progress_bar=False,
                                n_devices=1,
                                eval_reps=2)
        
        popsize = self.config["optimizer_config"]["optimizer_params"]["popsize"]
        initial_info = {
			'Achievements/wake_up': 0.0,
			'Achievements/make_iron_armour': 0.0,
			'Achievements/eat_bat': 0.0,
			'Achievements/make_stone_pickaxe': 0.0,
			'Achievements/defeat_orc_solider': 0.0,
			'Achievements/enter_sewers': 0.0,
			'Achievements/cast_iceball': 0.0,
			'Achievements/defeat_skeleton': 0.0,
			'Achievements/defeat_ice_elemental': 0.0,
			'Achievements/collect_stone': 0.0,
			'Achievements/make_wood_sword': 0.0,
			'Achievements/cast_fireball': 0.0,
			'Achievements/place_table': 0.0,
			'Achievements/eat_cow': 0.0,
			'Achievements/defeat_orc_mage': 0.0,
			'Achievements/defeat_deep_thing': 0.0,
			'Achievements/eat_plant': 0.0,
			'Achievements/learn_fireball': 0.0,
			'Achievements/collect_drink': 0.0,
			'Achievements/make_torch': 0.0,
			'Achievements/collect_diamond': 0.0,
			'Achievements/defeat_troll': 0.0,
			'Achievements/find_bow': 0.0,
			'Achievements/make_diamond_pickaxe': 0.0,
			'Achievements/open_chest': 0.0,
			'Achievements/defeat_frost_troll': 0.0,
			'Achievements/defeat_knight': 0.0,
			'Achievements/enchant_sword': 0.0,
			'Achievements/make_diamond_sword': 0.0,
			'Achievements/collect_iron': 0.0,
			'Achievements/enter_fire_realm': 0.0,
			'Achievements/defeat_archer': 0.0,
			'Achievements/learn_iceball': 0.0,
			'Achievements/eat_snail': 0.0,
			'Achievements/defeat_fire_elemental': 0.0,
			'Achievements/make_diamond_armour': 0.0,
			'Achievements/collect_sapling': 0.0,
			'Achievements/drink_potion': 0.0,
			'Achievements/enter_gnomish_mines': 0.0,
			'Achievements/place_torch': 0.0,
			'Achievements/enter_dungeon': 0.0,
			'Achievements/collect_sapphire': 0.0,
			'Achievements/make_iron_sword': 0.0,
			'Achievements/defeat_lizard': 0.0,
			'Achievements/enter_ice_realm': 0.0,
			'Achievements/defeat_gnome_warrior': 0.0,
			'Achievements/place_furnace': 0.0,
			'Achievements/defeat_kobold': 0.0,
			'Achievements/damage_necromancer': 0.0,
			'Achievements/collect_ruby': 0.0,
			'Achievements/enter_vault': 0.0,
			'Achievements/make_wood_pickaxe': 0.0,
			'Achievements/defeat_gnome_archer': 0.0,
			'Achievements/defeat_necromancer': 0.0,
			'Achievements/defeat_pigman': 0.0,
			'Achievements/enchant_armour': 0.0,
			'Achievements/enter_graveyard': 0.0,
			'Achievements/collect_wood': 0.0,
			'Achievements/enter_troll_mines': 0.0,
			'Achievements/defeat_zombie': 0.0,
			'Achievements/fire_bow': 0.0,
			'Achievements/make_iron_pickaxe': 0.0,
			'discount': 0.0,
			'Achievements/place_stone': 0.0,
			'Achievements/make_arrow': 0.0,
			'Achievements/collect_coal': 0.0,
			'Achievements/make_stone_sword': 0.0,
			'Achievements/place_plant': 0.0
		}
        #obs, init_env_state = self.env.initialize(jax.random.PRNGKey(0), current_task=0)
        obs, init_env_state = self.env.initialize(jax.random.PRNGKey(0), current_task=0)

        init_env_state = CraftaxState(env_state=init_env_state, obs=obs, reward=0.0, done=False, info=initial_info)
        init_env_state = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), popsize, axis=0), init_env_state)


        #final_info = trainer.init_and_train_(self.train_key, init_env_state=init_env_state)
        final_info, total_noise, archive_history, fitnesses_history = trainer.init_and_train_(self.train_key)
        
        
        
        total_noise_np = onp.array(total_noise)
        onp.savetxt(self.config["exp_config"]["trial_dir"] + "/data/train/total_noise.csv", 
                total_noise_np, delimiter=',', fmt='%.6f')
        
        # Save archive history for PCA visualization
        # archive_history_np = onp.array(archive_history)
        #onp.save(self.config["exp_config"]["trial_dir"] + "/data/train/archive_history.npy", archive_history_np)
        #print(f"Archive history saved with shape: {archive_history_np.shape}")

        #fitnesses_history_np = onp.array(fitnesses_history)
        ##onp.save(self.config["exp_config"]["trial_dir"] + "/data/train/fitnesses_history.npy", fitnesses_history_np)
        #print(f"Fitnesses history saved with shape: {fitnesses_history_np.shape}")

        self.final_state = {"params": final_info.best_member}
        
        
        
    def save_params(self, training_state):

        def callback(info):
            current_gen, current_task, state, interm_policies, best_indiv = info
            last_dev_step = 1
            best_member = jax.tree_map(lambda x: x[best_indiv, ...], state)

            file_path = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task-1) + ".pkl"
            if not os.path.exists(file_path):

                with open(file_path, "wb") as f:
                    pickle.dump( (current_gen,best_member), f)


            interm_policies = jax.tree_map(lambda x: x[best_indiv,0, ...], interm_policies)


            file_path = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/policy_states_task_" + str(
                current_task - 1) + ".pkl"
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    pickle.dump(interm_policies, f)

        jax.debug.callback(callback, training_state)

    def save_training_info(self):


        #TODO: here we need to load from latest generation

        #last_gen = self.config["optimizer_config"]["optimizer_params"]["generations"]
        #with open(self.config["exp_config"]["trial_dir"] + "/data/train/all_info/gen_" + str(last_gen) + "/dev_" +str(self.config["model_config"]["model_params"]["max_dev_steps"]+2) +".pkl", "rb") as f:
        #    policy_state = pickle.load(f)
        policy_state = self.final_state["params"]

        checkpoint_policy_states = []
        for task in range(self.config["env_config"]["num_tasks"]):

            try:

                with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                        task) + ".pkl","rb") as f:
                    data = pickle.load(f)
                    checkpoint_policy_states.append(data)
            except FileNotFoundError:
                continue

        # save final policy matrix
        self.training_info = {"policy_network": {"final": policy_state,
                                                 "checkpoints": checkpoint_policy_states}}

        super().save_training_info()