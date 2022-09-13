from plasticity_network.dataLoader import DataLoader
from plasticity_network.network import MyNetwork
from plasticity_network.utils import (
    create_input_mapping,
    plot_one_neuron_spiking,
    plot_spiking_activity,
    plot_tsne_clusters,
)

from sklearn.manifold import TSNE
import numpy as np
import pickle as pickle


class Experiment:
    def __init__(self, config):

        print("Initializing DataLoader ...")
        self.data = DataLoader()

        print("Creating the network...")
        self.networkWrapper = MyNetwork(config)
        self.cfg = config

    def cluster(self):
        input_frequencies = self.network.get_input_frequencies_to_neurons(self.data)

        print("Clustering...")

        tsne_embed = TSNE(
            n_components=2, perplexity=20.0, learning_rate="auto", init="random"
        ).fit_transform(input_frequencies)

        np.save("tsne_inputs.npy", tsne_embed)
        plot_tsne_clusters(self.cfg, tsne_embed, self.data.y_test)

    def run(self):

        x_train = self.data.x_train

        print("Creating input mapping...")

        if not self.cfg.precomputed_mappings:
            create_input_mapping(self.network, x_train, self.cfg.path_to_mappings)
            self.cfg.precomputed_mappings = True

        spiking_activity_of_neurons = []

        per_digit_weights_e = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
        per_digit_weights_i = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }

        print("Iterating through the samples")
        for indx, sample in enumerate(x_train):
            digit = int(self.data.y_train[indx][0])

            print("digit:", digit)
            weight_E, weight_I = self.networkWrapper.run(sample)

            spike_mon = self.networkWrapper.network["spike_monitor"]

            spiking_activity_of_neurons.append(spike_mon.count)

            name = str(self.data.y_train[indx][0]) + "neuron_" + str(indx)
            plot_spiking_activity(self.cfg, spike_mon, name)
            np.save(
                self.cfg.path_to_results + "/neuron" + str(indx) + "spike_count.npy",
                np.asarray(spike_mon.count),
            )

            with open(
                self.cfg.path_to_results
                + "/"
                + str(self.data.y_train[indx][0])
                + "spike_trains_neuron"
                + str(indx)
                + ".pkl",
                "wb",
            ) as f:
                pickle.dump(spike_mon.spike_trains(), f)

            per_digit_weights_e[digit].append(weight_E)

            per_digit_weights_i[digit].append(weight_I)

        spiking_activity_of_neurons = np.asarray(spiking_activity_of_neurons)
        np.save(
            self.cfg.path_to_results + "/all_neurons_spiking.npy",
            np.asarray(spiking_activity_of_neurons),
        )

        with open(self.cfg.path_to_results + "/excitatory_weights.pkl", "wb") as f:
            pickle.dump(per_digit_weights_e, f)

        with open(self.cfg.path_to_results + "/inhibitory_weights.pkl", "wb") as f:
            pickle.dump(per_digit_weights_i, f)
