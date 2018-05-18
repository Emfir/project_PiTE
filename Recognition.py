import numpy as np
import pandas as pd
import cv2
import os


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
        self.mask_threshold = 150
        self.knn = self.train_network()
        print('Training complete')

    def prepare_digit_train_sample(self, images):
        images_low = 12
        images_high = 412
        digit = np.zeros((1, self.width * self.height, 3), np.uint8)
        digit[:, :, 0] = digit[:, :, 1] = digit[:, :, 2] = images[images_low:images_high]
        digit = digit.reshape(self.height, self.width, 3)
        return self.scale_input_digit(digit)

    def extract_digit_roi(self, digit):
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

    def scale_input_digit(self, digit):
        bw_digit = self.extract_digit_roi(digit)
        r, c = bw_digit.shape
        bw_digit = np.hstack((bw_digit, np.zeros((r, self.horizontal_sz - c))))
        bw_digit = np.vstack((bw_digit, np.zeros((self.vertical_sz - r, self.horizontal_sz))))
        # cv2.imwrite('/Trash/dig_{:d}.jpg'.format(count), bw_digit)
        return bw_digit

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

    def get_input_sudoku(self, processed_slices):
        test_imgs = []
        img_mask = []
        thresh = self.mask_threshold
        for img in processed_slices:
            if img is not None:
                mask = np.count_nonzero((np.asarray(img) > thresh))
                test_imgs.append(img)
                img_mask.append(mask)

        sudoku_set = np.asarray(test_imgs, dtype=np.float32).reshape(-1, self.horizontal_sz * self.vertical_sz)
        return sudoku_set, img_mask

    def recognize_digits(self, processed_slices):
        to_recognize, img_mask = self.get_input_sudoku(processed_slices)
        ret, result, neighbours, dist = self.knn.findNearest(to_recognize, k=5)
        return result, img_mask

    def print_sudoku(self, digits_list, img_mask):
        print()
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                if j + self.board_size * i >= len(digits_list) or img_mask[j + self.board_size * i] == 0:
                    print(' ', end=' ')
                else:
                    print(int(digits_list[j + self.board_size * i][0]), end=' ')
            print()

    def get_sudoku_to_solve(self, digits_list, mask):
        sudoku = []
        for j in range(0, self.board_size):
            for i in range(0, self.board_size):
                if j + self.board_size * i >= len(digits_list) or mask[j + self.board_size * i] == 0:
                    sudoku.append(0)
                else:
                    sudoku.append(int(digits_list[j + self.board_size * i][0]))
        return sudoku

    def print_solved_sudoku(self, solved_sudoku):
        print()
        for j in range(0, self.board_size):
            for i in range(0, self.board_size):
                print(int(solved_sudoku[j + self.board_size * i]), end=' ')
            print()
