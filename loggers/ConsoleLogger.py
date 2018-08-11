class ConsoleLogger:
    def __init__(self):
        self.__prefix = 'INFO: '

    def log(self, *text):
        text = [str(item) for item in text]
        print(self.__prefix + ''.join(text))
