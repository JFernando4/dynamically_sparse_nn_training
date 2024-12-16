# built-in libraries
import os
import pickle
import time

# third party libraries
import numpy as np
import gymnasium as gym
import torch
from torch.optim import AdamW
from mlproj_manager.util import access_dict, turn_off_debugging_processes
from mlproj_manager.experiments import Experiment

# src import
from src.rl_agents import Buffer, PPO, Agent
from src.networks.ppo_networks import MLPVF, MLPPolicy, initialize_two_layer_network
from src.utils import set_random_seed, parse_terminal_arguments, compute_matrix_rank_summaries, compute_average_weight_magnitude
from src.cbpw_functions.utilities import initialize_weight_dict


class PolicyCollapseExperiment(Experiment):


    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose: bool = True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        turn_off_debugging_processes(access_dict(exp_params, key="debug", default=True, val_type=bool))
        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # set random seed for reproducibility
        set_random_seed(self.run_index)

        """ Experiment Parameters """

        # optimizer parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.rescaled_wd = access_dict(exp_params, "rescaled_wd", default=False, val_type=bool)
        self.adam_beta1 = access_dict(exp_params, "adam_beta1", default=0.99, val_type=float)
        self.adam_beta2 = access_dict(exp_params, "adam_beta2", default=0.99, val_type=float)
        self.adam_eps = access_dict(exp_params, "adam_eps", default=1e-8, val_type=float)
        # PPO parameters
        self.no_clipping = access_dict(exp_params, "no_clipping", default=False, val_type=bool)
        self.max_grad_norm = access_dict(exp_params, "max_grad_norm", default=1e9, val_type=float)
        self.buffer_size = access_dict(exp_params, "buffer_size", default=1e6, val_type=int)
        self.gamma = access_dict(exp_params, "gamma", default=0.99, val_type=float)
        self.gae_lambda = access_dict(exp_params, "gae_lambda", default=0.95, val_type=float)
        self.num_epochs = access_dict(exp_params, "num_epochs", default=10, val_type=int)
        self.n_slices = access_dict(exp_params, "n_slices", default=10, val_type=int)
        self.u_adv_scl = access_dict(exp_params, "u_adv_scl", default=True, val_type=bool)
        self.clip_epsilon = access_dict(exp_params, "clip_epsilon", default=0.2, val_type=float)
        # shrink-and-perturb parameters
        self.perturb_std = access_dict(exp_params, "perturb_std", default=0.0, val_type=float)
        # continual backprop parameters
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        self.cbp_utility = access_dict(exp_params, "cbp_utiltiy", default="contribution", val_type=str)
        self.decay_rate = access_dict(exp_params, "decay_rate", default=0.0, val_type=float)    # also for redo
        self.use_cbp = (self.replacement_rate > 0.0) and (self.maturity_threshold > 0)
        # ReDo parameters
        self.redo_reinit_threshold = access_dict(exp_params, "redo_reinit_threshold", default=0.0, val_type=float)
        self.redo_reinit_freq = access_dict(exp_params, "redo_reinit_freq", default=0, val_type=int)
        self.redo_utility = access_dict(exp_params, "redo_utility", default="original", val_type=str)
        self.use_redo = (self.redo_reinit_threshold > 0.0) and (self.redo_reinit_freq > 0)
        # SWR parameters
        self.reinit_freq = access_dict(exp_params, "reinit_freq", default=0, val_type=int)
        self.drop_factor = access_dict(exp_params, "drop_factor", default=float, val_type=float)
        self.prune_method = access_dict(exp_params, "prune_method", default="none", val_type=str,
                                        choices=["none", "magnitude", "gf", "gr", "mr"])
        self.grow_method = access_dict(exp_params, "grow_method", default="none", val_type=str,
                                       choices=["none", "init", "zero", "truncated"])
        self.use_swr = (self.prune_method != "none") and (self.grow_method != "none")

        # environment parameters
        self.total_env_steps = access_dict(exp_params, "total_env_steps", default=1e6, val_type=int)
        self.env_name = exp_params["env_name"]

        # network parameters
        self.activation_type = access_dict(exp_params, "activation_type", default="ReLU", val_type=str)
        self.hidden_dim = access_dict(exp_params, "hidden_dim", default=16, val_type=int)
        self.use_ln = access_dict(exp_params, "use_ln", default=True, val_type=bool)

        """ Initialize Environment """
        # self.env = gym.make(self.env_name, render_mode="human")
        self.env = gym.make(self.env_name)
        self.env.name = None
        input_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]
        self.current_step = 0

        """ Initialize Networks """
        self.num_hidden_layers = 2
        network_arguments = {
            "input_dim": input_dim,
            "act_type": self.activation_type,
            "h_dim": self.hidden_dim,
            "device": self.device,
            "use_cbp": self.use_cbp,
            "maturity_threshold": self.maturity_threshold,
            "replacement_rate": self.replacement_rate,
            "use_redo": self.use_redo,
            "reinit_frequency": self.redo_reinit_freq,
            "reinit_threshold": self.redo_reinit_threshold,
            "decay_rate": self.decay_rate,
            "use_ln": self.use_ln
        }
        self.policy_network = MLPPolicy(a_dim=a_dim, **network_arguments)
        initialize_two_layer_network(self.policy_network.mean_net)
        self.val_function_network = MLPVF(**network_arguments)
        initialize_two_layer_network(self.val_function_network.v_net)
        self.replay_buffer = Buffer(input_dim, a_dim, self.buffer_size, device=self.device)
        self.optimizer = AdamW

        """ SWR Set Up """
        weight_dict = None
        if self.use_swr:
            weight_dict = initialize_weight_dict(net=self.policy_network.mean_net,
                                                 val_network=self.val_function_network.v_net,
                                                 architecture_type="ppo_networks",
                                                 prune_method=self.prune_method,
                                                 grow_method=self.grow_method,
                                                 drop_factor=self.drop_factor)

        """" Initialize PPO Agent """
        self.learner = PPO(
            pol=self.policy_network,
            buf=self.replay_buffer,
            lr=self.stepsize,
            g=self.gamma,
            vf=self.val_function_network,
            lm=self.gae_lambda,
            Opt=self.optimizer,
            u_epi_up=0,
            device=self.device,
            n_itrs=self.num_epochs,
            n_slices=self.n_slices,
            u_adv_scl=self.u_adv_scl,
            clip_eps=self.clip_epsilon,
            max_grad_norm=self.max_grad_norm,
            wd=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
            no_clipping=self.no_clipping,
            weight_dict=weight_dict,
            swr_reinit_freq=self.reinit_freq
        )
        self.agent = Agent(pol=self.policy_network, learner=self.learner)

        """ Initialize summaries """
        self.to_log = ["dead_units_prop", "pol_weights", "val_weights", "pol_grad_magnitude", "val_grad_magnitude", "stable_rank"]
        self.result_store_frequency = 1000
        self.stable_rank_store_frequency = self.result_store_frequency * 10

        results_dim = self.total_env_steps // self.result_store_frequency
        if "pol_weights" in self.to_log:
            self.results_dict["pol_weights"] = np.zeros(shape=results_dim)
        if "val_weights" in self.to_log:
            self.results_dict["val_weights"] = np.zeros(shape=results_dim)
        feature_activity_summaries_shape = (results_dim, self.num_hidden_layers, self.hidden_dim)
        self.short_term_feature_activity = torch.zeros(size=feature_activity_summaries_shape)
        if "dead_units_prop" in self.to_log:
            self.results_dict["dead_units_prop"] = torch.zeros(size=(results_dim,))
        if "stable_rank" in self.to_log:
            self.results_dict["stable_rank"] = torch.zeros(size=(self.total_env_steps // self.stable_rank_store_frequency, ))
        self.return_per_episode = []
        self.termination_steps = []

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_step"
        self.checkpoint_save_frequency = 1e6                # save 1 million environment steps
        self.delete_old_checkpoints = True

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):

        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "pol_weights": self.policy_network.state_dict(),
            "value_function_weights": self.val_function_network.state_dict(),
            "optimizer_state": self.learner.opt.state_dict(),
            "current_step": self.current_step,
            "short_term_feature_activity": self.short_term_feature_activity,
            "returns": self.return_per_episode,
            "termination_steps": self.termination_steps,
            "partial_results": partial_results
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path) -> bool:
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were successfully loaded
        """

        try:
            with open(file_path, mode="rb") as experiment_checkpoint_file:
                checkpoint = pickle.load(experiment_checkpoint_file)
        except EOFError:
            print("Couldn't load checkpoint pickle file.")
            return False

        self.policy_network.load_state_dict(checkpoint["pol_weights"])
        self.val_function_network.load_state_dict(checkpoint["value_function_weights"])
        self.learner.opt.load_state_dict(checkpoint["optimizer_state"])
        self.current_step = checkpoint["current_step"]
        self.short_term_feature_activity = checkpoint["short_term_feature_activity"]
        self.return_per_episode = checkpoint["returns"]
        self.termination_steps = checkpoint["termination_steps"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            if k not in partial_results.keys():
                print(f"Warning! {k} is not a partial result stored in the checkpoint!")
                continue
            if isinstance(partial_results[k], torch.Tensor):
                self.results_dict[k][:partial_results[k].shape[0]] = partial_results[k].to(self.device)
            elif isinstance(partial_results[k], np.ndarray):
                self.results_dict[k][:partial_results[k].shape[0]] = partial_results[k]
            else:
                self.results_dict[k] = partial_results[k]

        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        return True

    def run(self):

        # load checkpoint if available
        self.load_experiment_checkpoint()

        # train agent
        self.train_agent()

        # format results
        self.format_results()

        # summaries are stored in memory by calling exp.store_results()

    def train_agent(self):
        # trains the agent for self.total_env_steps

        current_return = 0.0
        observation, info = self.env.reset()

        while self.current_step < self.total_env_steps:
            # self.env.render()
            if (self.current_step % self.result_store_frequency == 0) and (len(self.return_per_episode) > 0):
                self._print(f"Current step: {self.current_step}\n\tLast sum of rewards: {self.return_per_episode[-1]:.4f}")

            # get new action
            action, log_prob, dist, new_features = self.agent.get_action(observation)
            # receive new observation and reward
            new_observation, reward, done, truncated, infos = self.env.step(action)
            # save information in the buffer
            self.agent.log_update(observation, action, reward, new_observation, log_prob, dist, done or truncated)
            # compute summaries
            self.compute_results(new_features)
            # update state and return
            observation = new_observation
            current_return += reward

            if done or truncated:
                self.return_per_episode.append(current_return)
                self.termination_steps.append(self.current_step)
                current_return = 0.0
                observation, info = self.env.reset()

            self.current_step += 1

            if self.current_step % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def compute_results(self, new_features):
        """ Computes the results of the experiment """

        for layer_idx in range(self.num_hidden_layers):
            self.short_term_feature_activity[self.current_step % self.result_store_frequency, layer_idx, :] = new_features[layer_idx].detach().clone()

        result_index = self.current_step // self.result_store_frequency
        if "pol_weights" in self.to_log:
            self.results_dict["pol_weights"][result_index] += compute_average_weight_magnitude(self.policy_network.mean_net)[0]
        if "val_weights" in self.to_log:
            self.results_dict["val_weights"][result_index] += compute_average_weight_magnitude(self.val_function_network.v_net)[0]

        if (self.current_step + 1) % self.result_store_frequency == 0:
            # store stable rank summaries
            if "stable_rank" in self.to_log:
                _, _, _, current_stable_rank = compute_matrix_rank_summaries(
                    m=self.short_term_feature_activity[:, -1, :], use_scipy=True)
                self.results_dict["stable_rank"][self.current_step // self.stable_rank_store_frequency] = current_stable_rank

            if "dead_units_prop" in self.to_log:
                reshaped_feature_activity = self.short_term_feature_activity.reshape(-1, self.num_hidden_layers * self.hidden_dim)
                dead_units_prop = (reshaped_feature_activity.mean(dim=0) == 0).float().mean()
                if dead_units_prop > 0.0:
                    print(f"\n\n{dead_units_prop = }\n\n")
                self.results_dict["dead_units_prop"][result_index] = dead_units_prop
                # self.results_dict["dead_units_prop"][result_index] = (self.short_term_feature_activity > 0.0).float().mean(dim=0)

    def format_results(self):
        """
        changes all the results array to numpy arrays
        """
        self.results_dict["return_per_episode"] = np.array(self.return_per_episode)
        self.results_dict["termination_steps"] = np.array(self.termination_steps)
        self.results_dict["dead_units_prop"] = self.results_dict["dead_units_prop"].numpy()
        self.results_dict["stable_rank"] = self.results_dict["stable_rank"].numpy()


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    from mlproj_manager.file_management.file_and_directory_management import read_json_file
    terminal_arguments = parse_terminal_arguments()
    experiment_parameters = read_json_file(terminal_arguments.config_file)
    file_path = os.path.dirname(os.path.abspath(__file__))

    print(experiment_parameters)

    # create result dir from the relevant parameters
    relevant_parameters = experiment_parameters["relevant_parameters"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    # run the experiment
    initial_time = time.perf_counter()
    exp = PolicyCollapseExperiment(experiment_parameters,
                                   results_dir=os.path.join(file_path, "results", results_dir_name),
                                   run_index=terminal_arguments.run_index,
                                   verbose=terminal_arguments.verbose)
    exp.run()
    # store results
    exp.store_results()
    # display runtime
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
