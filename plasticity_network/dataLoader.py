from struct import unpack
import numpy as np


class DataLoader:
    def __init__(self, path="./mnist/"):

        print("Reading the input....")
        self.images_train, self.y_train = self._load_dataset(path, train=True)
        self.images_test, self.y_test = self._load_dataset(path, train=False)

        print("Transforming the input into frequencies...")
        self.x_train = self._transform_to_network_input(self.images_train)
        self.x_test = self._transform_to_network_input(self.images_test)

    def _load_dataset(self, path, train=True):
        """

        :return:
        """

        if train:
            images = open(path + "train-images-idx3-ubyte", "rb")
            labels = open(path + "train-labels-idx1-ubyte", "rb")
        else:
            images = open(path + "t10k-images-idx3-ubyte", "rb")
            labels = open(path + "t10k-labels-idx1-ubyte", "rb")
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack(">I", images.read(4))[0]
        rows = unpack(">I", images.read(4))[0]
        cols = unpack(">I", images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack(">I", labels.read(4))[0]

        if number_of_images != N:
            raise Exception("number of labels did not match the number of images")

        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            x[i] = [
                [unpack(">B", images.read(1))[0] for unused_col in range(cols)]
                for unused_row in range(rows)
            ]
            y[i] = unpack(">B", labels.read(1))[0]

        return x, y

    def _transform_to_network_input(self, x):
        data_ = []
        for image in x:

            frequency_values = []
            for row in image:
                for pixel in row:
                    val = pixel * (40 / 255)
                    frequency_values.append(val)

            data_.append(frequency_values)

        return np.asarray(data_)
