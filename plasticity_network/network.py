from brian2 import *
from matplotlib import pyplot as plt
from tqdm import tqdm

from plasticity_network.utils import read_mapping
from plasticity_network.config import *

class MyNetwork:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_neurons = NeuronGroup(cfg.N, model=lif_eqs,
                                    threshold='v>v_thresh', reset='v=v_reset',
                                    refractory=refract, method='euler')

    def get_input_frequencies_to_neurons(self, dataloader):
        input_signals_to_pictures = []
        for picture_indx, picture in tqdm(enumerate(dataloader.x_test)):
            print("Picture n.o.", picture_indx)
            network_input = np.zeros(self.cfg.N)
            for pixel_index, freq_value in tqdm(enumerate(picture)):
                for input_neuron_index in np.load(self.cfg.path_to_mappings+"/"+str(pixel_index)+"_j.npy"):
                    network_input[input_neuron_index] += freq_value
            input_signals_to_pictures.append(network_input)

        np.save("input_signals_to_pictures.npy", np.asarray(input_signals_to_pictures))
        return np.asarray(input_signals_to_pictures)


    def run(self, sample):

        poisson_inputs = []
        input_synapses = []

        print("Adding poisson inputs...")
        for pixel_indx, rate in tqdm(enumerate(sample)):

            poisson_input = PoissonGroup(self.cfg.N, rates=rate * Hz, dt=1 * ms)

            S = Synapses(poisson_input, self.model_neurons, on_pre='ge += d_ge')

            if self.cfg.precomputed_mappings:
                i_indices, j_indices = read_mapping(self.cfg.path_to_mappings, pixel_indx)
                S.connect(i=i_indices, j=j_indices)
            else:
                raise Exception

            poisson_inputs.append(poisson_input)
            input_synapses.append(S)

        print("Defining the synapses")
        self.model_neurons.ge = 0
        self.model_neurons.gi = 0

        Ce = Synapses(self.model_neurons, self.model_neurons, model=syn_eqs[plast_flag],
                      on_pre=syn_on_pre_ex,
                      on_post=syn_on_post[plast_flag])
        Ci = Synapses(self.model_neurons, self.model_neurons, model=syn_eqs[plast_flag],
                      on_pre=syn_on_pre_inh,
                      on_post=syn_on_post[plast_flag])

        Ce.connect('i<'+str(0.8*self.cfg.N), p=0.12) # p --> ???
        Ci.connect('i>='+str(0.8*self.cfg.N), p=0.20) # p --> ???

        network = Network(self.model_neurons, Ce, Ci)

        Ce.weight_e = 0.6 * nsiemens
        Ci.weight_i = 0.6 * nsiemens

        Ce.pre_time = 1 * ms
        Ce.post_time = 1 * ms
        Ce.p_allowed = 1 * 1

        Ci.pre_time = 1 * ms
        Ci.post_time = 1 * ms
        Ci.p_allowed = 1 * 1

        self.model_neurons.v = 'v_reset + rand() * (v_thresh  - v_reset)'

        # creating monitors
        weights_monitor = StateMonitor(Ce, ['weight_e', 'pre_time', 'post_time'],
                                       dt=1 * ms, record=20)
        weights_monitor_i = StateMonitor(Ci, ['weight_i', 'pre_time', 'post_time'],
                                         dt=1 * ms, record=20)

        spike_mon = SpikeMonitor(self.model_neurons)
        state_mon = StateMonitor(self.model_neurons, True, record=20)

        network.add(input_synapses)
        network.add(poisson_inputs)

        network.add(weights_monitor)
        network.add(weights_monitor_i)

        network.add(state_mon)
        network.add(spike_mon)

        print("Running simulation...")
        network.run(10 * ms)

        return spike_mon, state_mon, weights_monitor, weights_monitor_i


