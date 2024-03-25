
import argparse


def parse_terminal_arguments():
    """ Reads experiment arguments """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config_file", action="store", type=str, required=True,
                                 help="JSON file with experiment parameters.")
    argument_parser.add_argument("--run_index", action="store", type=int, default=0,
                                 help="This determines the random seed for the experiment.")
    argument_parser.add_argument("--verbose", action="store_true", default=False)
    return argument_parser.parse_args()