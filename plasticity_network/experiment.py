from plasticity_network.dataLoader import DataLoader
from plasticity_network.network import MyNetwork
from plasticity_network.utils import (
    create_input_mapping,
    plot_tsne_clusters,
)
import umap

from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

class Experiment:
    def __init__(self, config, ):

        print("Initializing DataLoader ...")
        self.data = DataLoader()

        print("Creating the network...")
        self.networkWrapper = MyNetwork(config)
        self.cfg = config

    def cluster(self, method='t-sne', name_to_save = 'clustered_inputs'):
        input_frequencies = self.networkWrapper.get_input_frequencies_to_neurons(self.data)

        print("Clustering...")

        if method == 't-sne':
            reducer = TSNE(n_components=2, perplexity=20.0, learning_rate="auto", init="random")
        elif method == 'umap':
            reducer = umap.UMAP()

        embedding = reducer.fit_transform(input_frequencies)

        np.save(name_to_save+".npy", embedding)
        plot_tsne_clusters(self.cfg, embedding, self.data.y_test.flatten())

    def run(self):

        x_train = self.data.x_train
        x_test = self.data.x_test

        if not self.cfg.precomputed_mappings:
            print("Creating input mapping...")
            create_input_mapping(self.networkWrapper, x_train, self.cfg.path_to_mappings)
            self.cfg.precomputed_mappings = True

        self.networkWrapper.train(x_train)
        for indx, sample in tqdm(enumerate(x_train)):

            self.networkWrapper.run(sample)

            spike_mon = self.networkWrapper.network["spike_monitor"]

            # spiking_activity_of_neurons.append(spike_mon.count)

            # name = str(self.data.y_train[indx][0]) + "neuron_" + str(indx)

            # plot_spiking_activity(self.cfg, spike_mon, name)
            trains = spike_mon.spike_trains()

            isi_result = np.zeros(self.cfg.N, 2)
            for neuron_indx, spike_train in enumerate(trains):
                isi_ = np.diff(spike_train)

                isi_mean = np.mean(isi_)
                isi_std = np.std(isi_)

                isi_result[neuron_indx] = np.array([isi_mean, isi_std])

            np.save(
                self.cfg.path_to_results + "/isi_neuron" + str(indx) + ".npy",
                isi_result
            )

            np.save(
                self.cfg.path_to_results + "/spike_count_neuron" + str(indx) + ".npy",
                np.asarray(spike_mon.count),
            )

            # write_to_txt_file(self.cfg.path_to_results+'/'+str(digit)+'weights_E.txt',weight_E)
            #
            # write_to_txt_file(self.cfg.path_to_results+'/'+str(digit)+'weights_I.txt',weight_I)

            # write_to_txt_file(
            #     self.cfg.path_to_results + "/all_neurons_spiking.txt",
            #     np.asarray(spike_mon.count),
            # )

        # spiking_activity_of_neurons = np.asarray(spike_mon.count)

        # np.save(
        #     self.cfg.path_to_results + "/all_neurons_spiking.npy",
        #     np.asarray(spiking_activity_of_neurons),
        # )

        # with open(self.cfg.path_to_results + "/excitatory_weights.pkl", "wb") as f:
        #     pickle.dump(per_digit_weights_e, f)
        #
        # with open(self.cfg.path_to_results + "/inhibitory_weights.pkl", "wb") as f:
        #     pickle.dump(per_digit_weights_i, f)
