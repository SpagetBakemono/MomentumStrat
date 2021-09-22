# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:57:29 2021

@author: Aditya
"""

import pandas as pd
import os

class save_data():
    
    def __init__(self, index100Path, index50Path, index50nPath, stockPath):
        self.index100Path = index100Path
        self.index50Path = index50Path
        self.index50nPath = index50nPath
        self.stockPath = stockPath
        self.stockDict = {}
        self.tickers = []
    
    def make_dataframe(self):
        for filename in os.listdir(self.stockPath):
            if(filename!="Univ"):
                print("dfizing ",filename)
                file_df = pd.read_excel(self.stockPath+"/"+filename)
                self.stockDict[filename[6:-5]] = file_df
        for filename in self.stockDict:
            print("saving ",filename)
            self.tickers.append(filename)
            self.stockDict[filename].to_hdf('Saved/' + filename + '.h5', key=filename, mode='w')