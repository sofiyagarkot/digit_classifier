from plasticity_network.config import Config
from plasticity_network.experiment import Experiment
from plasticity_network.argparser import parse_args

import warnings

warnings.filterwarnings("ignore")

# Experiment with some default parameters
args = parse_args()
cfg = Config(args)
exp = Experiment(cfg)
exp.run()
