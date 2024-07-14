import json

import pandas as pd
from tkinter import filedialog
from tkinter import *


class ParseLogs:
    def __init__(self, key_to_extract):
        df = self.open_data()
        save_directory = self.ask_directory()


    def open_data(self):
            with open('output/logs.json', 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df

    def ask_directory(self):
        root = Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()
        return folder_selected