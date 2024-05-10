# built-in libraries
import time
import os
import pickle

# third party libraries
import torch
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification # For models fine-tuned on MNLI
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset

# from ml project manager
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import access_dict, turn_off_debugging_processes, get_random_seeds
from src.cbpw_functions import initialize_weight_dict, initialize_ln_list_bert, setup_cbpw_layer_norm_update_function
from src.utils import parse_terminal_arguments
from src.utils.bert_sentiment_analysis_experiment_utils import CBPWTrainer
from src.utils.cifar100_experiment_utils import save_model_parameters


class BERTSentimentAnalysisExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        random_seeds = get_random_seeds()
        self.random_seed = random_seeds[self.run_index]
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.num_epochs = access_dict(exp_params, "num_epochs", default=100, val_type=int)
        self.evaluation_dataset = access_dict(exp_params, "evaluation_dataset", default="validation", val_type=str,
                                              choices=["validation", "test"])
        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.fixed_wd = access_dict(exp_params, "fixed_wd", default=False, val_type=bool)

        # CBPw parameters
        self.use_cbpw = access_dict(exp_params, "use_cbpw", default=False, val_type=bool)
        self.use_cbpw_ln = access_dict(exp_params, "use_cbpw_ln", default=False, val_type=bool)
        self.topology_update_freq = access_dict(exp_params, "topology_update_freq", default=1, val_type=int)
        pruning_functions_names = ["none", "magnitude", "redo", "gf", "hess_approx"]
        self.prune_method = access_dict(exp_params, "prune_method", default="none", val_type=str, choices=pruning_functions_names)
        grow_methods = ["none", "pm_min", "xavier_normal", "zero"]
        self.grow_method = access_dict(exp_params, "grow_method", default="none", val_type=str, choices=grow_methods)
        assert not ((self.prune_method != "none" and self.grow_method == "none") or (self.prune_method == "none" and self.grow_method != "none"))
        self.drop_factor = access_dict(exp_params, "drop_factor", default=0.0, val_type=float)

        """ Network set up """
        self.batch_size = 30
        self.steps_per_epoch = 67350 // self.batch_size
        self.evaluation_frequency = 1
        self.num_evaluation_steps = self.num_epochs // self.evaluation_frequency
        self.checkpoint_save_frequency = 5
        self.model_parameter_save_frequency = 5
        self.summary_counter = 0
        self.trainer = None
        # initialize network
        config = BertConfig.from_pretrained("prajjwal1/bert-mini")  # Using this configuration.
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
        self.net = BertForSequenceClassification.from_pretrained("prajjwal1/bert-mini")
        # self.net = BertForSequenceClassification(config)
        # Print the configuration of the model.
        self._print(f"Config: {config}")
        self.net.to(self.device)

        # initializes weight dictionary for CBPw
        self.weight_dict, self.ln_list, self.norm_layer_update_func = None, None, None
        if self.use_cbpw:
            self.weight_dict = initialize_weight_dict(self.net, "bert", self.prune_method, self.grow_method, self.drop_factor)
            self.ln_list = initialize_ln_list_bert(self.net)
            self.norm_layer_update_func = setup_cbpw_layer_norm_update_function(self.prune_method, self.drop_factor, True)

        self.initialize_results_dir()
        self.load_accuracy = load_metric("accuracy", trust_remote_code=True)
        self.load_f1 = load_metric("f1", trust_remote_code=True)


    def initialize_results_dir(self):
        self.results_dict[f"{self.evaluation_dataset}_accuracy"] = torch.zeros(self.num_evaluation_steps + 1,
                                                                               dtype=torch.float32, device=self.device)
        self.results_dict[f"{self.evaluation_dataset}_f1"] = torch.zeros(self.num_evaluation_steps + 1,
                                                                         dtype=torch.float32, device=self.device)

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):

        dataset = load_dataset("sst2")
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataset = dataset.map(self.tokenize_batch, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            learning_rate=self.stepsize,
            lr_scheduler_type="constant",
            save_strategy="steps",
            save_steps=self.steps_per_epoch * self.checkpoint_save_frequency,
            save_total_limit=1,
            evaluation_strategy="steps",
            eval_steps=self.steps_per_epoch * self.evaluation_frequency,
            dataloader_num_workers=12,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            output_dir=os.path.join(self.results_dir, f"checkpoint_index_{self.run_index}"),
            seed=self.random_seed
        )

        # Define the Trainer
        self.trainer = CBPWTrainer(
            model=self.net,
            args=training_args,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset[self.evaluation_dataset],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            use_cbpw=self.use_cbpw,
            topology_update_freq=self.topology_update_freq,
            bert_weight_dict=self.weight_dict,
            use_cbpw_ln=self.use_cbpw_ln,
            bert_ln_list=self.ln_list,
            ln_update_function=self.norm_layer_update_func,
            fixed_wd=self.fixed_wd
        )

        self.trainer.evaluate()
        self.trainer.train()

    def log_summaries(self, metrics: dict = None):
        """ Stores the accuracy and f1-measure metrics of the current network """
        self.results_dict[f"{self.evaluation_dataset}_accuracy"][self.summary_counter] += metrics["eval_accuracy"]
        self.results_dict[f"{self.evaluation_dataset}_f1"][self.summary_counter] += metrics["eval_f1"]
        self.summary_counter += 1
        if (self.summary_counter % self.model_parameter_save_frequency) == 0:
            save_model_parameters(self.results_dir, self.run_index, net=self.net,
                                  current_epoch=self.summary_counter * self.evaluation_frequency)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = self.load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = self.load_f1.compute(predictions=predictions, references=labels)["f1"]
        self.log_summaries(metrics={"eval_accuracy": accuracy, "eval_f1": f1})
        return {"accuracy": accuracy, "f1": f1}

    def tokenize_batch(self, batch):
        return self.tokenizer(batch['sentence'], truncation=True)


def main():
    """
    Function for running the experiment from command line given a path to a json config file
    """
    from mlproj_manager.file_management.file_and_directory_management import read_json_file
    terminal_arguments = parse_terminal_arguments()
    experiment_parameters = read_json_file(terminal_arguments.config_file)
    file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_parameters["data_path"] = os.path.join(file_path, "data")
    print(experiment_parameters)

    relevant_parameters = experiment_parameters["relevant_parameters"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    initial_time = time.perf_counter()
    exp = BERTSentimentAnalysisExperiment(experiment_parameters,
                                          results_dir=os.path.join(file_path, "results", results_dir_name),
                                          run_index=terminal_arguments.run_index,verbose=terminal_arguments.verbose)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
