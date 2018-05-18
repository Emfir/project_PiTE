import numpy as np
import pandas as pd
import cv2
import os


class Recognition:
    def __init__(self):
        self.data_folder = 'data/'
        self.listOfFiles = os.listdir(self.data_folder)
        self.source = self.data_folder + 'AGENCY.csv'
        self.height = 20
        self.width = 20
        self.knn = self.train_network()
        print('Training complete')


    def get_sample_digits(self, source):
        digit_low_code = 48
        digit_high_code = 57
        df = pd.read_csv(self.data_folder + source)
        df0 = df[(df.m_label >= digit_low_code) & (df.m_label <= digit_high_code)]
        strength = df0.groupby('strength')['strength'].mean().get_values()
        italic = df0.groupby('italic')['italic'].mean().get_values()
        return strength, italic, df0

    def get_train_set_labeled(self, count, training_digits):
        set_size = 4350
        train_set = np.asarray(training_digits[:set_size], dtype=np.float32) \
            .reshape(-1, self.horizontal_sz * self.vertical_sz)
        k = np.arange(self.board_size, -1, -1)
        count /= 10
        count = round(set_size / 10, 0)
        train_labels = np.array([k] * int(count))
        train_labels = train_labels.reshape(-1, 1)
        return train_set, train_labels

    def prepare_train_set(self):
        count = 0
        training_digits = []
        for source in self.listOfFiles:
            strength, italic, df0 = self.get_sample_digits(source)
            for k in italic:
                for strength_value in strength:
                    df1 = df0[(df0.italic == k) & (df0.strength == round(strength_value, 1))]
                    zm = 0
                    for images in df1.get_values():
                        zm += 1
                        if zm == 11:
                            break
                        bw_digit = self.prepare_digit_train_sample(images)
                        count += 1
                        training_digits.append(bw_digit)

        train_set, train_labels = self.get_train_set_labeled(count, training_digits)
        return train_set, train_labels

    def train_network(self):
        knn = cv2.ml.KNearest_create()
        train_set, train_labels = self.prepare_train_set()
        try:
            knn.train(train_set, cv2.ml.ROW_SAMPLE, train_labels)
        except Exception as e:
            print(e)
            return None
        return knn


