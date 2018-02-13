from copy import copy
import pandas as pd

class DataClass():
    def gather_data(self):
        self.gather_data_submethod()

    def process_data(self):
        self.process_data_submethod()

class HistoricData(DataClass):
    def __init__(self, file_path):
        self.file_path = file_path

    def gather_data_submethod(self):
        self.raw_data = pd.read_csv(self.file_path)

    def process_data_submethod(self):
        processed_data = copy(self.raw_data)
        self.processed_data = processed_data
