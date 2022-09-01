from plasticity_network.dataLoader import DataLoader
from plasticity_network.network import MyNetwork
from plasticity_network.utils import create_input_mapping, plot_one_neuron_spiking, plot_spiking_activity, plot_tsne_clusters

from sklearn.manifold import TSNE
import numpy as np

class Experiment:
    def __init__(self,
                 config
                 ):

        print("Initializing DataLoader ...")
        self.data = DataLoader()

        print("Creating the network...")
        self.network = MyNetwork(config)
        self.cfg = config

    def cluster(self):
        input_frequencies = self.network.get_input_frequencies_to_neurons(self.data)

        print("Clustering...")

        tsne_embed = TSNE(n_components=2, perplexity=20.0, learning_rate='auto',
                          init='random').fit_transform(input_frequencies)

        np.save("tsne_inputs.npy",tsne_embed)
        plot_tsne_clusters(self.cfg, tsne_embed, self.data.y_test)



    def run(self):

        x_train = self.data.x_train

        print("Creating input mapping...")

        if not self.cfg.precomputed_mappings:
            create_input_mapping(self.network, x_train, self.cfg.path_to_mappings)
            self.cfg.precomputed_mappings = True

        spiking_activity_of_neurons = []

        print("Iterating through the samples")
        for indx, sample in enumerate(x_train):
            print("digit:", self.data.y_train[indx])
            spike_mon, state_mon, weights_monitor, weights_monitor_i = self.network.run(sample)
            spiking_activity_of_neurons.append(spike_mon.count)

            plot_one_neuron_spiking(self.cfg, state_mon)
            plot_spiking_activity(self.cfg, spike_mon)


            break

        spiking_activity_of_neurons = np.asarray(spiking_activity_of_neurons)
        np.save(self.cfg.path_to_results+"/100.npy", spiking_activity_of_neurons)




