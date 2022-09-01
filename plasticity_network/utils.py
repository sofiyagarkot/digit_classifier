from brian2 import *
import numpy as np
from tqdm import tqdm

def create_input_mapping(network, x_train, path_to_mapping):

    sample = x_train[0]

    for pixel_indx, val in tqdm(enumerate(sample)):
        poisson_input_default = PoissonGroup(network.cfg.N, rates=0 * Hz, dt=1 * ms)
        d_ge = 0.6 * nsiemens

        S_default = Synapses(poisson_input_default, network.model_neurons, 'w:1', on_pre='g_e += d_ge')
        S_default.connect(p=0.07)

        np.save(path_to_mapping+"/"+str(pixel_indx)+'_i.npy', S_default.i[:])
        np.save(path_to_mapping+"/"+str(pixel_indx)+'_j.npy', S_default.j[:])

        #  __________________________________________________________
        # Create a matrix to store the weights and fill it with NaN
        # W = np.full((len(poisson_input_default), len(network.model_neurons)), np.nan)
        # Insert the values from the Synapses object
        # W[S_default.i[:], S_default.j[:]] = S_default.w[:]
        #  save it for future ?
        #  __________________________________________________________


def read_mapping(path, pixel_index):
    '''

    :param path: path to a npy-file
    :return:
    '''

    i = np.load(path+"/"+str(pixel_index)+'_i.npy')
    j = np.load(path+"/"+str(pixel_index)+'_j.npy')

    return i,j

def plot_one_neuron_spiking(cfg, monitor):
    plt.figure(figsize=(30, 20))
    plt.plot(monitor.t, monitor.v.T)
    plt.title("Neuron 20 membrane voltage")
    plt.savefig(cfg.path_to_visuals+"/10ms_neuron20.png")

def plot_spiking_activity(cfg, monitor):

    plt.figure(figsize=(30, 20))
    plt.plot(monitor.t / ms, monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title("Spiking activity of network")
    plt.savefig(cfg.path_to_visuals+"/10ms.png")

def plot_tsne_clusters(cfg, point_cloud, classes):

    plt.figure(figsize=(30, 30))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c = classes)
    plt.title("Clustered activity of inputs to the network")
    plt.savefig(cfg.path_to_visuals+"/tsne_inputs.png")
