from argparse import ArgumentParser
import os

def add_args(parser):
    """

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--N', type=int,
                        default=10000,
                        help='Number of neurons in the network')

    parser.add_argument('--path_to_results',
                        default='/Users/sofiyagarkot/Desktop/IST/Mayaan/digit_classifier/results',
                        type=str)

    parser.add_argument('--experiment_name',
                        default='0',
                        type=str)

    parser.add_argument('--path_to_mappings',
                        default='/Users/sofiyagarkot/Desktop/IST/Mayaan/digit_classifier/mappings/0',
                        type=str)

    parser.add_argument('--path_to_visuals',
                        default='/Users/sofiyagarkot/Desktop/IST/Mayaan/digit_classifier/results/0/visuals',
                        type=str)


def check_args(parser):

    if not os.path.isdir(parser.path_to_results+"/"+parser.experiment_name):
        os.mkdir(parser.path_to_results)

    parser.path_to_results = parser.path_to_results + "/" + parser.experiment_name

    if not os.path.isdir(parser.path_to_results+"/"+"visuals"):
        parser.path_to_visuals = parser.path_to_results+"/"+"visuals"
        os.mkdir(parser.path_to_results+"/"+"visuals")

    if not os.path.isdir(parser.path_to_mappings):
        os.mkdir(parser.path_to_mappings)

def parse_args():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    check_args(args)

    return args


