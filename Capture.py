import cv2
from Recognition import Recognition
from ImageProcessor import ImageProcessor
import solver.ssolver as solver


class Capture:

    def __init__(self):
        print('Start training')
        self.recognition = Recognition()
        self.image_processor = ImageProcessor()
        self.wait_ms = 500
        self.camera_device = 0

    def start_capture(self):
        cap = cv2.VideoCapture(self.camera_device)
        while (True):
            ret, frame = cap.read()
            processed_images = self.image_processor.process_frame(frame)
            if processed_images:
                try:
                    result, mask = self.recognition.recognize_digits(processed_images)
                    self.recognition.print_sudoku(result, mask)
                    sudoku = self.recognition.get_sudoku_to_solve(result, mask)
                    solved = solver.solve(sudoku)
                    self.recognition.print_solved_sudoku(solved)
                except Exception:
                    pass

            cv2.imshow('frame',frame)
            if cv2.waitKey(self.wait_ms) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Capture().start_capture()
