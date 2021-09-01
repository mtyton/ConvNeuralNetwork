import numpy as np
import cv2

from matplotlib import image as img
from matplotlib import pyplot as plt


class SingleData:

    def __init__(self, path, category, shape):
        self.path = path
        self.category = category
        self.shape = shape

    @property
    def label(self):
        return self.category

    def normalize(self, image):
        if len(image.shape) < 3:
            return cv2.resize(image, self.shape)

        try:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(self.path)
            print(image.shape)
            raise e
        return cv2.resize(grey, self.shape)

    @property
    def raw_image(self):
        return img.imread(self.path)

    @property
    def image(self):
        return self.normalize(self.raw_image)

    def __repr__(self):
        return self.path

    def __str__(self):
        return self.path


class DataReader:
    def __init__(self, split_ratio=0.7, shape=(98, 64)):
        self.image_classes = ["Dog", "Cat"]
        self.max_image_number = 12000
        self.split_ratio = split_ratio
        self.shape = shape

    def shuffle(self, data):
        for i in range(len(data)):
            first_index = np.random.randint(len(data))
            second_index = np.random.randint(len(data))
            while second_index == first_index:
                second_index = np.random.randint(len(data))
            data[first_index], data[second_index] = \
                data[second_index], data[first_index]
        return data

    def split_data(self, data):
        size = len(data)
        train_data = data[int(size * self.split_ratio):]
        test_data = data[:int(size*self.split_ratio)]
        return train_data, test_data

    def read_data(self, data_size):
        # we don't need to shuffle because we read random data
        data = []
        for image_class in self.image_classes:
            for i in range(int(data_size/2)):
                path = "PetImages/{}/{}.jpg".format(
                    image_class, i
                )
                data.append(SingleData(
                    path, image_class, self.shape
                ))
        data = self.shuffle(data)
        return self.split_data(data)
