import numpy as np

from data_reader import DataReader
from layer import ConvolutionLayer, PoolingLayer, SoftMaxLayer
from matplotlib import pyplot as plt
import pickle


class Network:
    conv = ConvolutionLayer(8)
    pool = PoolingLayer()
    soft = SoftMaxLayer(31 * 48 * 8, 2)
    reader = DataReader()

    acc_filename = "bestAcc"
    LABEL_MAP = {'Dog': 0, "Cat": 1}

    def __init__(self):
        self.load_models()

    def save_models(self, last_best_acc):
        for layer in [self.conv, self.pool, self.soft]:
            filename = "{}.pickle".format(str(layer))
            pickle_file = open(filename, 'wb')
            pickle.dump(layer, pickle_file)
            pickle_file.close()
        # dump also last acc
        pickle_file = open(self.acc_filename + ".pickle", 'wb')
        pickle.dump(last_best_acc, pickle_file)
        pickle_file.close()

    def load_model(self, file_name):
        try:
            pickle_file = open(file_name + ".pickle", 'rb')
            obj = pickle.load(pickle_file)
        except IOError as e:
            return
        pickle_file.close()
        return obj

    def load_models(self):
        conv = self.load_model(str(self.conv))
        pool = self.load_model(str(self.pool))
        soft = self.load_model(str(self.soft))
        if conv:
            self.conv = conv
        if pool:
            self.pool = pool
        if soft:
            self.soft = soft

    def get_labels(self, data):
        labels = []
        for d in data:
            labels.append(d.label)
        return labels

    def read_data(self, set_type="train"):
        train_set, test_set = self.reader.read_data(10000)
        if set_type == "train":
            return train_set, self.get_labels(train_set)
        else:
            return test_set, self.get_labels(test_set)

    def forward(self, img, label):
        # if our oyt is size of 10 than our gradient will be size of 10
        out = self.conv.forward((img / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.soft.forward(out)

        numeric_label = self.LABEL_MAP[label]

        loss = -1*np.log(out[numeric_label])
        accuracy = 1 if np.argmax(out) == numeric_label else 0

        gradient = self.soft.backpropagate(out, numeric_label)
        gradient = self.pool.backpropagate(gradient)
        self.conv.backpropagate(gradient)

        return accuracy, loss, out

    def train(self, epochs):
        images, labels = self.read_data()
        total_acc = 0
        avg_loss = 0
        prev_best_acc = self.load_model(self.acc_filename)
        if not prev_best_acc:
            prev_best_acc = 0

        for e in range(epochs):
            iterator = 0
            for img in images:
                label = labels[iterator]
                acc, loss, out = self.forward(img.image, label)
                iterator += 1
                total_acc += acc
                avg_loss += loss

            total_acc = total_acc / len(images) * 100
            avg_loss = avg_loss / len(images)
            print("Epoch number: {} accuracy: {}, loss: {}".format(e, total_acc, avg_loss))
            # after each iteration save models if they behave better
            if prev_best_acc < total_acc:
                self.save_models(prev_best_acc)
                prev_best_acc = total_acc

    def classify(self, interactivity=False):
        images, labels = self.read_data(set_type="test")
        iterator = 0
        correct = 0

        for img in images:
            label = labels[iterator]
            out = self.conv.forward((img.image / 255) - 0.5)
            out = self.pool.forward(out)
            out = self.soft.forward(out)
            numeric_label = self.LABEL_MAP[label]
            decision = np.argmax(out)
            if decision == 0:
                guessed_label = "Dog"
            else:
                guessed_label = "Cat"
            if img.label == guessed_label:
                correct += 1
            if interactivity:
                plt.imshow(img.raw_image)
                plt.title(guessed_label)
                plt.show()
            iterator += 1

        print(correct/len(images))