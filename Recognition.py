import numpy as np
import pandas as pd
import cv2
import os
from loggers.ConsoleLogger import ConsoleLogger
from keras import layers
from keras import models
from keras import utils
import matplotlib.pyplot as plt


class Recognition:
    def __init__(self):
        self.data_folder = 'data/'
        self.listOfFiles = os.listdir(self.data_folder)
        self.source = self.data_folder + 'AGENCY.csv'
        self.vertical_sz = 30
        self.horizontal_sz = 20
        self.height = 20
        self.width = 20
        self.board_size = 9
        self.mask_threshold = 140
        self.test_set_size_percent = 10
        self.epochs = 7
        self.__debug = False
        self.__samples_dir = './data_samples'
        self.__logger = ConsoleLogger()
        self.network = self.__train_network()
        self.__logger.log('Training complete')

    def __prepare_digit_train_sample(self, images, count):
        images_low = 12
        images_high = 412
        digit = np.zeros((1, self.width * self.height, 3), np.uint8)
        digit[:, :, 0] = digit[:, :, 1] = digit[:, :, 2] = images[images_low:images_high]
        digit = digit.reshape(self.height, self.width, 3)
        return self.__scale_input_digit(digit, count)

    def __extract_digit_roi(self, digit):
        bounding_box_width = 17
        bounding_box_height = 21
        bw_digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
        bw_digit = cv2.resize(bw_digit, (bounding_box_width, bounding_box_height))
        ret, bw_digit = cv2.threshold(bw_digit, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours = cv2.findContours(bw_digit, 1, 2)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        bw_digit = bw_digit[y:y + h, x:x + w]
        return bw_digit

    def __scale_input_digit(self, digit, count):
        bw_digit = self.__extract_digit_roi(digit)
        r, c = bw_digit.shape
        bw_digit = np.hstack((bw_digit, np.zeros((r, self.horizontal_sz - c))))
        bw_digit = np.vstack((bw_digit, np.zeros((self.vertical_sz - r, self.horizontal_sz))))
        if self.__debug:
            cv2.imwrite(self.__samples_dir + '/dig_{:d}.jpg'.format(count), bw_digit)
        return bw_digit

    def __get_sample_digits(self, source):
        digit_low_code = 48
        digit_high_code = 57
        data_features = pd.read_csv(self.data_folder + source)
        data_features_mod = data_features[
            (data_features.m_label >= digit_low_code) & (data_features.m_label <= digit_high_code)]
        strength = data_features_mod.groupby('strength')['strength'].mean().get_values()
        italic = data_features_mod.groupby('italic')['italic'].mean().get_values()
        return strength, italic, data_features_mod

    def __get_train_set_labeled(self, feature_count, training_digits):
        set_size = len(training_digits)
        train_set = np.asarray(training_digits[:set_size], dtype=np.float32)

        digits_label_vector = np.arange(self.board_size, -1, -1)
        feature_count /= 10
        feature_count = round(set_size / 10, 0)
        train_labels = np.array([digits_label_vector] * int(feature_count))
        train_labels = train_labels.reshape(-1, 1)
        return train_set, train_labels

    def __one_hot_to_dig(self, digits):
        return np.argmax(digits, axis=1).tolist()

    def __setup_train_and_test_set(self, count, training_digits, test_set_size):

        train_set, train_labels = self.__get_train_set_labeled(count, training_digits)
        train_len = train_set.shape[0] * (100 - test_set_size) // 100
        test_set = train_set[train_len:]
        test_labels = train_labels[train_len:]
        train_set = train_set[:train_len]
        train_labels = train_labels[:train_len]
        train_set = train_set.reshape(train_set.shape[0], self.vertical_sz, self.horizontal_sz, 1)
        train_set = train_set.astype('float32') / 255
        test_set = test_set.reshape(test_set.shape[0], self.vertical_sz, self.horizontal_sz, 1)
        test_set = test_set.astype('float32') / 255
        train_labels = utils.to_categorical(train_labels)
        test_labels = utils.to_categorical(test_labels)
        self.__logger.log('Train set length: {:d}'.format(train_set.shape[0]))
        self.__logger.log('Test set length: {:d}'.format(test_set.shape[0]))

        return train_set, train_labels, test_set, test_labels

    def __prepare_train_set(self, test_set_size):
        count = 0
        training_digits = []
        if self.__debug:
            if not os.path.isdir(self.__samples_dir):
                os.mkdir(self.__samples_dir)
        for source in self.listOfFiles:
            strength, italic, data_features_old = self.__get_sample_digits(source)
            for k in italic:
                for strength_value in strength:
                    data_features_new = data_features_old[
                        (data_features_old.italic == k) & (data_features_old.strength == round(strength_value, 1))]
                    field = 0
                    for images in data_features_new.get_values():
                        field += 1
                        if field == 11:
                            break
                        bw_digit = self.__prepare_digit_train_sample(images, count)
                        count += 1
                        training_digits.append(bw_digit)
        return self.__setup_train_and_test_set(count, training_digits, test_set_size)

    def __plot_training_history(self, history):

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'ko', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training, validation accuracy')
        plt.xlabel('Epoch no.')
        plt.ylabel('Normalized accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'ko', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training, validation loss')
        plt.xlabel('Epoch no.')
        plt.ylabel('Normalized loss')
        plt.legend()
        plt.show()

    def __train_network(self):
        first_layer_size = 48
        inner_layers_size = 96
        classes_number = 10
        filter_size = 3
        polling_size = 2

        if os.path.isfile('network.h5'):
            network = models.load_model('network.h5')
            return network

        network = models.Sequential()
        network.add(layers.Conv2D(first_layer_size, (filter_size, filter_size), activation='relu',
                                  input_shape=(self.vertical_sz, self.horizontal_sz, 1)))
        network.add(layers.MaxPooling2D((polling_size, polling_size)))
        network.add(layers.Conv2D(inner_layers_size, (filter_size, filter_size), activation='relu'))
        network.add(layers.MaxPooling2D((polling_size, polling_size)))
        network.add(layers.Conv2D(inner_layers_size, (filter_size, filter_size), activation='relu'))
        network.add(layers.Flatten())
        network.add(layers.Dense(inner_layers_size, activation='relu'))
        network.add(layers.Dense(classes_number, activation='softmax'))

        train_set, train_labels, test_set, test_labels = self.__prepare_train_set(self.test_set_size_percent)

        try:

            # keras
            network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            network.summary()
            history = network.fit(train_set, train_labels, epochs=self.epochs, batch_size=64,
                                  validation_data=(test_set, test_labels))
            test_loss, test_accuracy = network.evaluate(test_set, test_labels)
            self.__logger.log('\nTest accuracy {:f}'.format(test_accuracy))

        except Exception as e:
            self.__logger.log('Training failed')
            self.__logger.log(e)
            return None

        self.__plot_training_history(history)
        network.save('network.h5')

        return network

    def __get_input_sudoku(self, processed_slices):
        test_imgs = []
        img_mask = []
        thresh = self.mask_threshold
        for img in processed_slices:
            if img is not None:
                mask = np.count_nonzero((np.asarray(img) > thresh))
                test_imgs.append(img)
                img_mask.append(mask)

        sudoku_set = np.asarray(test_imgs, dtype=np.float32) \
            .reshape([-1, self.vertical_sz, self.horizontal_sz, 1])  # reshape for keras

        sudoku_set = sudoku_set.astype('float32') / 255

        return sudoku_set, img_mask

    def recognize_digits(self, processed_slices):
        to_recognize, img_mask = self.__get_input_sudoku(processed_slices)
        result = self.network.predict(to_recognize, batch_size=128)
        result = self.__one_hot_to_dig(result)
        return result, img_mask

    def print_sudoku(self, digits_list, img_mask):
        print()
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                if j + self.board_size * i >= len(digits_list) or img_mask[j + self.board_size * i] == 0:
                    print(' ', end=' ')
                else:
                    print(int(digits_list[j + self.board_size * i]), end=' ')
            print()

    def get_sudoku_to_solve(self, digits_list, mask):
        sudoku = []
        for j in range(0, self.board_size):
            for i in range(0, self.board_size):
                if j + self.board_size * i >= len(digits_list) or mask[j + self.board_size * i] == 0:
                    sudoku.append(0)
                else:
                    sudoku.append(int(digits_list[j + self.board_size * i]))
        return sudoku

    def print_solved_sudoku(self, solved_sudoku):
        print()
        for j in range(0, self.board_size):
            for i in range(0, self.board_size):
                print(int(solved_sudoku[j + self.board_size * i]), end=' ')
            print()
