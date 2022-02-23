import pandas as pd


class DataHandler:

    def __init__(self):
        self.total_data = []
        self.index = -1

    def load_data(self, path):
        try:
            self.total_data = pd.read_csv(path)
            print("{} data is setted")
        except:
            print("error load_data")

    def next_data(self):
        self.index += 1
        if len(self.total_data) is self.index:
            return 0
        return self.total_data[self.index]

