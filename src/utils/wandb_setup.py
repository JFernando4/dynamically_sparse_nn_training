# built-in libraries
import os
import pickle
# third party libraries
import wandb
# my own libraries
from mlproj_manager.util import access_dict
from mlproj_manager.file_management import store_object_with_several_attempts


def initialize_wandb(exp_params: dict, results_dir: str, run_index: int, project: str, entity: str):
    """
    Initializes the weights and biases session
    :param exp_params: dictionary with experiment parameters
    :param results_dir: path to the directory where results are stored
    :param run_index: int corresponding to the index of the run
    :param project: wandb project name
    :param entity: wandb entity name
    :return: wandb session
    """

    # set wandb directory and mode
    os.environ["WANDB_DIR"] = access_dict(exp_params, "wandb_dir", default=results_dir, val_type=str)
    wandb_mode = access_dict(exp_params, "wandb_mode", default="offline", val_type=str,
                             choices=["online", "offline", "disabled"])

    # retrieve tags for the current run
    wandb_tags = access_dict(exp_params, "wandb_tags", default="", val_type=str)  # comma separated string of tags
    tag_list = None if wandb_tags == "" else wandb_tags.split(",")

    # set console log file
    slurm_job_id = "0" if "SLURM_JOBID" not in os.environ else os.environ["SLURM_JOBID"]
    exp_params["slurm_job_id"] = slurm_job_id

    run_id = get_wandb_id(results_dir, run_index)
    run_name = "{0}_index-{1}".format(os.path.basename(results_dir), run_index)
    return wandb.init(project=project, entity=entity, mode=wandb_mode, config=exp_params, tags=tag_list, name=run_name,
                      id=run_id)


def get_wandb_id(results_dir: str, run_index: int):
    """
    Generates and stores a wandb id or loads it if one is already available
    :param results_dir: path to the directory where results are stored
    :param run_index: int corresponding to the index of the run
    :return: wandb run id
    """

    wandb_id_dir = os.path.join(results_dir, "wandb_ids")
    os.makedirs(wandb_id_dir, exist_ok=True)

    wandb_id_filepath = os.path.join(wandb_id_dir, "index-{0}.p".format(run_index))

    # an id was already stored
    if os.path.isfile(wandb_id_filepath):
        with open(wandb_id_filepath, mode="rb") as id_file:
            run_id = pickle.load(id_file)
        return run_id

    # generate a new id
    run_id = wandb.util.generate_id()
    store_object_with_several_attempts(run_id, wandb_id_filepath, storing_format="pickle", num_attempts=10)

    return run_id
