import cv2
import numpy as np


class ImageProcessor:

    def __init__(self):
        self.rho_threshold = 20
        self.theta_threshold = 0.2
        self.lines_cnt_threshold = 170
        self.test_scaling_factor = 1.0
        self.deg = 180
        self.line_len = 100000
        self.crop_offset = 6  # 20 (for perfect image)
        self.hough_lines_cnt = 20

    def theta_modification(self, theta):
        angle_threshold = 170
        if theta / np.pi * self.deg > angle_threshold:
            return theta - np.pi
        else:
            return theta

    def intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        p1 = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
              )
        p2 = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
              )
        return p1, p2


    def collect_lines(self, lines, line_flags):
        filtered_lines = []
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])
        return filtered_lines

    def find_similar_lines(self, lines):
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                theta_i = self.theta_modification(theta_i)
                theta_j = self.theta_modification(theta_j)
                if abs(abs(rho_i) - abs(rho_j)) < self.rho_threshold \
                        and abs(theta_i - theta_j) < self.theta_threshold:
                    similar_lines[i].append(j)
        return similar_lines

    def filter_lines(self, lines):

        # how many lines are similar to a given one
        similar_lines = self.find_similar_lines(lines)

        # ordering the INDICES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:
                # if we already disregarded the ith element in the ordered list
                #  then we don't care (we will not delete anything based on it
                #  and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                theta_i = self.theta_modification(theta_i)
                theta_j = self.theta_modification(theta_j)
                if abs(abs(rho_i) - abs(rho_j)) < self.rho_threshold and abs(theta_i - theta_j) < self.theta_threshold:
                    line_flags[indices[j]] = False
                    # if it is similar and have not been disregarded yet then drop it now

        return self.collect_lines(lines, line_flags)

    def line_coordinates(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + self.line_len * (-b))
        y1 = int(y0 + self.line_len * a)
        x2 = int(x0 - self.line_len * (-b))
        y2 = int(y0 - self.line_len * a)
        return x1, y1, x2, y2

    def cut_slice(self, img, points):
        return img[self.crop_offset + points[0][1]: points[1][1] - self.crop_offset,
               self.crop_offset + points[0][0]: points[1][0] - self.crop_offset].copy()


    def create_slices(self, img, vertical_list, horizontal_list):
        img_list = []
        points = []
        for j in range(len(horizontal_list) - 1):
            for i in range(len(vertical_list) - 1):
                for x in range(2):
                    rho = vertical_list[i + x][0]
                    theta = vertical_list[i + x][1]
                    x1v, y1v, x2v, y2v = self.line_coordinates(rho, theta)

                    rho = horizontal_list[j + x][0]
                    theta = horizontal_list[j + x][1]
                    x1h, y1h, x2h, y2h = self.line_coordinates(rho, theta)

                    p1, p2 = self.intersection(x1v, y1v, x2v, y2v, x1h, y1h, x2h, y2h)
                    p1 = int(p1)
                    p2 = int(p2)
                    points.append((p1, p2))
                temp_img = self.cut_slice(img, points)
                if self.is_slice_blank(temp_img):
                    temp_img[:] = 0
                # bs = 16  # 16 (for perfect image)
                # tempImg = cv2.copyMakeBorder
                # (tempImg, top=bs, bottom=bs, left=bs, right=bs, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
                img_list.append(temp_img)
                points.clear()
        return img_list

    def classify_lines(self, filtered_lines):
        vertical_list = []
        horizontal_list = []
        vertical_threshold_low = 80
        vertical_threshold_high = 100
        horizontal_threshold_low = 170
        horizontal_threshold_high = 10

        for i in range(len(filtered_lines)):
            rho_i, theta_i = filtered_lines[i][0]
            rho_i = abs(rho_i)
            theta_i = self.theta_modification(theta_i)
            if theta_i / np.pi * self.deg > horizontal_threshold_low \
                    or theta_i / np.pi * self.deg < horizontal_threshold_high:
                vertical_list.append([rho_i, theta_i])

            elif theta_i / np.pi * self.deg > vertical_threshold_low \
                    and theta_i / np.pi * self.deg < vertical_threshold_high:
                horizontal_list.append([rho_i, theta_i])
        return vertical_list, horizontal_list


    def sort_line_lists(self, vertical_list, horizontal_list):
        vertical_list.sort(key=lambda x: x[0])
        horizontal_list.sort(key=lambda x: x[0])


 
