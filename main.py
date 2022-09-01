from plasticity_network.config import Config
from plasticity_network.experiment import Experiment
from plasticity_network.argparser import parse_args
# Experiment with some default parameters
args = parse_args()
cfg1 = Config(args)
exp1 = Experiment(cfg1)
exp1.cluster()