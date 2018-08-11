import cv2
from Recognition import Recognition
from ImageProcessor import ImageProcessor
import solver.ssolver as solver
from loggers.ConsoleLogger import ConsoleLogger


class Capture:

    def __init__(self):
        self.camera_device = 0
        self.__logger = ConsoleLogger()
        self.__logger.log('Init start...')
        self.__recognition = Recognition()
        self.__image_processor = ImageProcessor()
        self.__wait_ms = 20
        self.__logger.log('Init end')

    def start_capture(self):
        cap = cv2.VideoCapture(self.camera_device)
        while True:
            ret, frame = cap.read()
            if frame is not None:


                processed_images = self.__image_processor.process_frame(frame)
                if processed_images:
                    try:
                        result, mask = self.__recognition.recognize_digits(processed_images)
                        self.__recognition.print_sudoku(result, mask)
                        sudoku = self.__recognition.get_sudoku_to_solve(result, mask)
                        solved = solver.solve(sudoku)
                        self.__recognition.print_solved_sudoku(solved)
                    except Exception:
                        pass

                cv2.imshow('frame', frame)
                if cv2.waitKey(self.__wait_ms) & 0xFF == ord('q'):
                    break
            else:
                self.__logger.log('Cannot read data from camera')
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Capture().start_capture()
