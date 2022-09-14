from brian2 import *
from brian2 import profiling_summary
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

from plasticity_network.utils import read_mapping
from plasticity_network.config import *


class MyNetwork:
    def __init__(self, cfg):
        self.cfg = cfg
        self._create_network()

    def _create_network(self):

        model_neurons = NeuronGroup(
            self.cfg.N,
            model=lif_eqs,
            dt=1 * ms,
            threshold="v>v_thresh",
            reset="v=v_reset",
            refractory=refract,
            method="euler",
            name="neurongroup",
        )

        model_neurons.ge = 0
        model_neurons.gi = 0

        Ce = Synapses(
            model_neurons,
            model_neurons,
            model=syn_eqs[plast_flag],
            dt=1 * ms,
            on_pre=syn_on_pre_ex[
                plast_flag
            ],  # how to change the equation of the saved object in a network pickle file ?
            on_post=syn_on_post[plast_flag],
            name="exc_synapses",
        )
        Ci = Synapses(
            model_neurons,
            model_neurons,
            model=syn_eqs[plast_flag],
            dt=1 * ms,
            on_pre=syn_on_pre_inh[plast_flag],
            on_post=syn_on_post[plast_flag],
            name="inh_synapses",
        )

        Ce.connect("i<" + str(0.8 * self.cfg.N), p=p_ee)
        Ci.connect("i>=" + str(0.8 * self.cfg.N), p=p_ii)

        # TODO: implement i-to-e and e-to-i

        network = Network(model_neurons, Ce, Ci)

        Ce.weight_e = 0.6 * nsiemens
        Ci.weight_i = 0.6 * nsiemens
        # Ci.weight_i = np.random.lognormal(mean = w_ii_mu, sigma = w_ii_sigma, size=Ci.weight_i.shape) * nsiemens

        Ce.pre_time = 1 * ms
        Ce.post_time = 1 * ms
        Ce.p_allowed = 1 * 1

        Ci.pre_time = 1 * ms
        Ci.post_time = 1 * ms
        Ci.p_allowed = 1 * 1

        model_neurons.v = "v_reset + rand() * (v_thresh  - v_reset)"

        # creating monitors
        # weights_monitor = StateMonitor(
        #     Ce, ["weight_e"], dt=1 * ms, record=True, name="exc_weight_monitor"
        # )
        # weights_monitor_i = StateMonitor(
        #     Ci, ["weight_i"], dt=1 * ms, record=True, name="inh_weight_monitor"
        # )

        spike_mon = SpikeMonitor(model_neurons, name="spike_monitor")

        # weights_monitor.active = False
        # weights_monitor_i.active = False
        #
        # network.add(weights_monitor)
        # network.add(weights_monitor_i)
        network.add(spike_mon)

        self.network = network
        self.network.store(filename=self.cfg.path_to_network)

    def get_input_frequencies_to_neurons(self, dataloader, to_compute=False):
        if to_compute:
            input_signals_to_pictures = []
            for picture_indx, picture in tqdm(enumerate(dataloader.x_test)):
                network_input = np.zeros(self.cfg.N)
                for pixel_index, freq_value in tqdm(enumerate(picture)):
                    j_indices = read_mapping(self.cfg.path_to_mappings, pixel_index)
                    non_zero = np.zeros(self.cfg.N)
                    non_zero[j_indices] = freq_value
                    network_input += non_zero
                input_signals_to_pictures.append(network_input)

            np.save(
                "input_signals_to_pictures.npy", np.asarray(input_signals_to_pictures)
            )
            return np.asarray(input_signals_to_pictures)
        else:
            return np.load("input_signals_to_pictures.npy")

    def forward(self, sample):

        start_time = time.time()
        self.network.restore(filename=self.cfg.path_to_network)

        model_neurons = self.network["neurongroup"]

        poisson_inputs, synapses_ = [], []
        for pixel_indx, rate in enumerate(sample):

            if rate == 0.0:
                continue

            poisson_input = PoissonGroup(1, rates=rate * Hz, dt=1 * ms)

            S = Synapses(poisson_input, model_neurons, dt=1 * ms, on_pre="ge += d_ge")

            if self.cfg.precomputed_mappings:
                j_indices = read_mapping(self.cfg.path_to_mappings, pixel_indx)
                S.connect(i=[0], j=j_indices)
            else:
                raise Exception

            poisson_inputs.append(poisson_input)
            synapses_.append(S)

        self.network.add(poisson_inputs)
        self.network.add(synapses_)
        print("Running simulation...")
        self.network.run(300 * ms)

        # self.network["exc_weight_monitor"].active = True
        # self.network["inh_weight_monitor"].active = True

        # self.network.run(1 * ms)

        print(
            "One iteration took ",
            (time.time() - start_time) / 60.0,
            "m, ",
            (time.time() - start_time) % 60,
            "s",
        )
        # weight_E = np.asarray(self.network["exc_weight_monitor"].weight_e)[:, -1]
        # weight_I = np.asarray(self.network["inh_weight_monitor"].weight_i)[:, -1]

        # print("SHAPES", weight_E.shape, weight_I.shape)

        self.network.remove(poisson_inputs)
        self.network.remove(synapses_)

        del synapses_
        del poisson_inputs

        # self.network["exc_weight_monitor"].active = False
        # self.network["inh_weight_monitor"].active = False

        self.network.store(filename=self.cfg.path_to_network)

        # return weight_E, weight_I
    def test(self, data):
        pass

    def train(self, data):
        pass
    # TODO: implement train func + test func

    # TODO: implement monitors activity change (off on train and on on test)
