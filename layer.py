import numpy as np
import pickle


class Layer:
    pass


class ConvolutionLayer(Layer):
    # convolution layer using 3x3 filter
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters are 3d matrix
        self.filters = np.random.randn(num_filters, 3, 3) / 9
        self.last_input = None

    def __str__(self):
        return "Conv3x3"

    def get_regions(self, img):
        """
        This is a helper generator function which
        generates all 3x3 image regions.
        We use valid padding here.
        """
        height, width = img.shape

        for i in range(height-2):
            for j in range(width-2):
                im_region = img[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, inp):
        """
        This method splits image into regions
        """
        self.last_input = inp

        height, width = inp.shape
        output = np.zeros((height-2, width-2, self.num_filters))
        for im_region, i, j in self.get_regions(inp):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backpropagate(self, gradient, lear_rate=0.005):
        filter_gradient = np.zeros(self.filters.shape)

        for region, i, j in self.get_regions(self.last_input):
            for filter_num in range(self.num_filters):
                filter_gradient[filter_num] = gradient[i, j, filter_num] * region

        self.filters -= lear_rate*filter_gradient
        return None


class PoolingLayer(Layer):
    def __init__(self):
        self.last_input = None

    def __str__(self):
        return "Pooling3x3"

    def iterate_regions(self, img):
        height, width, _ = img.shape

        new_height, new_width = height//2, width//2

        for i in range(new_height):
            for j in range(new_width):
                im_region = img[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, inp):
        """
        This method joins neighbour simillar pixels
        to reduce image size
        """
        self.last_input = inp
        height, width, num_filters = inp.shape
        output = np.zeros((height//2, width//2, num_filters))

        for im_region, i, j in self.iterate_regions(inp):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backpropagate(self, gradient):
        input_gradient = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            height, width, depth = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for h in range(height):
                for w in range(width):
                    for d in range(depth):
                        if im_region[h, w, d] == amax[d]:
                            input_gradient[i*2 + h, j*2 + w, d] = gradient[i, j, d]

        return input_gradient


class SoftMaxLayer(Layer):
    def __init__(self, input_size, nodes):
        self.weights = np.random.randn(input_size, nodes) / input_size
        self.biases = np.zeros(nodes)
        self.last_input = None
        self.last_input_shape = None
        self.last_totals = None
        self.nodes = nodes

    def __str__(self):
        return "SoftMax"

    def forward(self, inp):
        """

        """
        # we store shape of input before it's being flatten
        self.last_input_shape = inp.shape

        inp = inp.flatten()
        # we store flatten input
        self.last_input = inp
        inp_size, nodes = self.weights.shape

        totals = np.dot(inp, self.weights) + self.biases

        self.last_totals = totals
        exp = np.exp(totals)

        return exp/np.sum(exp, axis=0)

    def backpropagate(self, out, label, learn_rate=0.005):
        # we need our initial gradient
        # derived of softmax function
        initial_gradient = np.zeros(self.nodes)
        initial_gradient[label] = -1 / out[label]

        # assertions, if one of this variables does not exist in current context
        # we can not proceed
        assert self.last_totals is not None
        assert self.last_input is not None
        assert self.last_input_shape is not None

        for i, gradient in enumerate(initial_gradient):
            if gradient == 0:
                continue

            # e^t
            t = np.exp(self.last_totals)

            # sum all e totals
            e_sum = np.sum(t)

            quatered_e_sum = (e_sum**2)
            # lets use chaing rule
            totals_gradient = -t[i]*t/quatered_e_sum

            # Quitient rule
            totals_gradient[i] = t[i] * (e_sum - t[i]) / quatered_e_sum

            # gradients weights, biases, input
            weights_gradient = self.last_input
            biases_gradient = 1
            input_gradient = self.weights

            # loss gradient
            loss_gradient = gradient * totals_gradient
            # change weights biases, input
            # T stands form transformate
            weights_gradient = weights_gradient[np.newaxis].T @ loss_gradient[np.newaxis]
            biases_gradient = loss_gradient*biases_gradient
            input_gradient = input_gradient @ loss_gradient

            self.weights -= learn_rate * weights_gradient
            self.biases -= learn_rate * biases_gradient

            return input_gradient.reshape(self.last_input_shape)

