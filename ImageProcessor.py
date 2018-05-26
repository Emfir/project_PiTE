import cv2
import numpy as np
from Recognition import Recognition
import solver.ssolver as solver


class ImageProcessor:

    def __init__(self):
        self.rho_threshold = 10
        self.theta_threshold = 0.2
        self.lines_cnt_threshold = 110
        self.test_scaling_factor = 1
        self.deg = 180
        self.v_sz = 30
        self.h_sz = 20
        self.line_len = 100000
        self.crop_offset = 6  # 20 (for perfect image)
        self.hough_lines_cnt = 20
        self.grid_size_px = 360

    def median_of_width(self, list_of_lines):
        vect = []
        for x in range(1, len(list_of_lines)):
            vect.append([(abs(list_of_lines[x][0] )- abs(list_of_lines[x - 1][0] )), x - 1])
        return vect[int (len(vect) / 2) - 1]

    def removing_indexes(self, list_of_indexes, list_of_lines):
        list_of_indexes.sort(reverse=True)
        for x in list_of_indexes:
            list_of_lines.pop(x)
        return  list_of_lines

    def filtering_using_median(self, list_of_lines):
        median, index = self.median_of_width(list_of_lines)

        lower_and_upper_threshold = [6 / 10 * median, 14 / 10 * median]
        bad_indexes = self.up_to_end(list_of_lines, lower_and_upper_threshold, index) +\
                      self.down_to_beginning(list_of_lines, lower_and_upper_threshold, index)


        return self.removing_indexes(bad_indexes, list_of_lines)

    def up_to_end(self,list_of_lines, lower_and_upper_threshold, index_waypoint):
        current = index_waypoint
        next = index_waypoint + 1
        bad_indeks = []
        while next != len(list_of_lines):
            if lower_and_upper_threshold[0] <= (abs(list_of_lines[next][0]) - abs(list_of_lines[current][0])) <= lower_and_upper_threshold[1]:
                current = next
                next += 1
            else:
                bad_indeks.append(next)
                next += 1
        return bad_indeks

    def down_to_beginning(self,list_of_lines, lower_and_upper_threshold, index_waypoint):
        current = index_waypoint + 1
        previous = index_waypoint
        bad_indeks = []
        while previous != -1:

            if lower_and_upper_threshold[0] <= (abs(list_of_lines[current][0]) - abs(list_of_lines[previous][0])) <= \
                    lower_and_upper_threshold[1]:
                current = previous
                previous -= 1
            else:
                bad_indeks.append(previous)
                previous -= 1
        return bad_indeks

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

    def slice_frame(self, frame):
        canny_low = 70
        canny_high = 180
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / self.deg, self.lines_cnt_threshold)
        return lines

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

    def is_slice_blank(self, temp_img):
        return np.count_nonzero((np.asarray(temp_img) > 220).all(axis=2)) == 0

    def create_slices(self, img, vertical_list, horizontal_list):
        img_list = []
        points = []
        cos = 0
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
                cos += 1
                temp_img = self.cut_slice(img, points)
                cv2.imwrite('without_preparation/lines{:d}.png'.format(cos), temp_img)
                if self.is_slice_blank(temp_img):
                    temp_img[:] = 0
                # bs = 16  # 16 (for perfect image)
                # tempImg = cv2.copyMakeBorder
                # (tempImg, top=bs, bottom=bs, left=bs, right=bs, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
                cv2.imwrite('proba/lines{:d}.png'.format(cos), temp_img)


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

    def enhance_slice(self, img_slice):
        kernel_size = 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        temp_img = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
        ret, temp_img = cv2.threshold(temp_img, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # COMMENT THIS LINE (for perfect image)
        temp_img = cv2.erode(temp_img, kernel, iterations=1)
        temp_img = cv2.dilate(temp_img, kernel, iterations=1)
        return temp_img

    def get_bounding_rectangle(self, temp_img):
        contours = cv2.findContours(temp_img, 1, 2)
        cnt = contours[0]
        return cv2.boundingRect(cnt)

    def reshape_slice(self, temp_img, x, y, w, h):
        temp_img = temp_img[y:y + h, x:x + w]
        r, c = temp_img.shape
        temp_img = np.hstack((temp_img, np.zeros((r, self.h_sz - c))))
        temp_img = np.vstack((temp_img, np.zeros((self.v_sz - r, self.h_sz))))
        return temp_img

    def binarize_slices(self, img_list):
        processed_images = []
        # i = 0
        # j = 0
        for img in img_list:
            temp_img = self.enhance_slice(img)
            x, y, w, h = self.get_bounding_rectangle(temp_img)

            if not x == y == w == h == 0:
                temp_img = self.reshape_slice(temp_img, x, y, w, h)
            else:
                temp_img = cv2.resize(temp_img, (self.h_sz, self.v_sz), cv2.INTER_AREA)

            # cv2.imwrite('grid/dig_{:d}{:d}.png'.format(j, i), temp_img)
            processed_images.append(temp_img)
            # i = i + 1
            # if i == 9:
            #    i = 0
            #    j = j + 1
        return processed_images

    def sort_line_lists(self, vertical_list, horizontal_list):
        vertical_list.sort(key=lambda x: x[0])
        horizontal_list.sort(key=lambda x: x[0])

    def rescale_frame(self, image, vertical_list, horizontal_list):
        board_size = int(
            (vertical_list[9][0] - vertical_list[0][0] + horizontal_list[9][0] - horizontal_list[0][0]) / 2)
        scaling_factor = self.grid_size_px / board_size
        vertical_list = [[line[0] * scaling_factor, line[1]] for line in vertical_list]
        horizontal_list = [[line[0] * scaling_factor, line[1]] for line in horizontal_list]
        rescaled_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor,
                                    interpolation=cv2.INTER_AREA)
        return rescaled_image, vertical_list, horizontal_list

    def get_rotation_angle(self, vertical_list):
        return sum(map(lambda line: line[1], vertical_list)) / len(vertical_list)

    def get_rotated_frame(self, frame, rot_angle, vertical_list, horizontal_list):
        r, c, d = frame.shape
        midpoint = (r // 2, c // 2)
        rot_matrix = cv2.getRotationMatrix2D(midpoint, rot_angle * self.deg / np.pi, 1.0)  # scale
        rot_frame = cv2.warpAffine(frame, rot_matrix, (c, r), flags=cv2.INTER_LINEAR)
        return rot_frame, vertical_list, horizontal_list

    def preprocess_frame(self, frame):
        lines = self.slice_frame(frame)
        if lines is None or not lines.any():
            print('No lines were found')
            return None, None, None

        filtered_lines = self.filter_lines(lines)
        vertical_list, horizontal_list = self.classify_lines(filtered_lines)
        self.sort_line_lists(vertical_list, horizontal_list)
        vertical_list = self.filtering_using_median(vertical_list)
        horizontal_list = self.filtering_using_median(horizontal_list)

        gray = frame.copy()
        debug = True
        if debug:

            for rt in vertical_list + horizontal_list:
                rho = rt[0]
                theta = rt[1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + self.line_len * (-b))
                y1 = int(y0 + self.line_len * a)
                x2 = int(x0 - self.line_len * (-b))
                y2 = int(y0 - self.line_len * a)
                cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow('frame', gray)
            cv2.waitKey(500)
            cv2.imwrite('lines.png', gray)


        if len(vertical_list) + len (horizontal_list) != self.hough_lines_cnt:
            print('Bad lines count {:d}'.format(len(vertical_list) + len (horizontal_list)))

            return None, None, None


        self.sort_line_lists(vertical_list, horizontal_list)
        print (len(vertical_list), len(horizontal_list))






        if len(vertical_list) != self.hough_lines_cnt / 2 or len(horizontal_list) != self.hough_lines_cnt / 2:
            print('Bad lines count v:{:d} h:{:d}'.format(len(vertical_list), len(horizontal_list)))
            return None, None, None

        rot_angle = self.get_rotation_angle(vertical_list)
        rot_angle_threshold = 0.15
        if abs(rot_angle) > rot_angle_threshold:
            print('Angle too big {:f}'.format(rot_angle))
            return None, None, None
        return rot_angle, vertical_list, horizontal_list

    def process_frame(self, frame):

        for self.rho_threshold in range(5, 100, 5):
            for self.lines_cnt_threshold in range (100, 170, 5):
                rot_angle, vertical_list, horizontal_list = self.preprocess_frame(frame)
                if not rot_angle:
                    continue
                rot_frame, vertical_list, horizontal_list = self.get_rotated_frame(frame, rot_angle, vertical_list,
                                                                                   horizontal_list)
                rot_angle, vertical_list, horizontal_list = self.preprocess_frame(rot_frame)
                if rot_angle != None:
                    break


            if rot_angle != None:
                break

        print ( self.rho_threshold, self.lines_cnt_threshold)

        if not rot_angle:
            return
        inverted_frame = cv2.bitwise_not(rot_frame)
        rescaled_frame, vertical_list, horizontal_list = self.rescale_frame(inverted_frame, vertical_list,
                                                                            horizontal_list)

        img_list = self.create_slices(rescaled_frame, vertical_list, horizontal_list)
        try:
            processed_images = self.binarize_slices(img_list)
            return processed_images
        except ValueError as e:
            print ( e)
            return None


if __name__ == '__main__':
    print('Training start')
    recognition = Recognition()
    frame = cv2.imread('rand.jpg')
    ip = ImageProcessor()
    frame = cv2.resize(frame, None, fx=ip.test_scaling_factor, fy=ip.test_scaling_factor, interpolation=cv2.INTER_CUBIC)
    processed_images = ip.process_frame(frame)
    result, mask = recognition.recognize_digits(processed_images)
    recognition.print_sudoku(result, mask)
    sudoku = recognition.get_sudoku_to_solve(result, mask)
    solved = solver.solve(sudoku)
    recognition.print_solved_sudoku(solved)
