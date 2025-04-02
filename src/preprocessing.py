# Standard Library Imports
import os
from pathlib import Path

# Third-Party Library Imports
import pandas as pd
import numpy as np


class BuildDataSet:
    def __init__(self,config):
            self.config = config


    def load_data(self): # Load and csv file in the train folder and concate them
        data_path = Path(os.getcwd()) / self.config["data"]["paths"]["train"]
        data_name = list (data_path.glob("*.csv"))
        if data_name:
            data = [pd.read_csv(file, header=None) for file in data_name]
            data = pd.concat(data, axis=1)
        data = np.array(data)
        
        return data


    def process_data(self): # In this function we mines the amb_tempreture from water_tempreture
        data = self.load_data()

        for i in range(0, data.shape[1], self.config["data"]["configuration"]["num_feature"]):
            data[:, i+self.config["data"]["feature_position"]["water_tempreture"]] = data[:, i+self.config["data"]["feature_position"]["water_tempreture"]] - data[:, i+self.config["data"]["feature_position"]["amb_tempreture"]]
        
        return data


    def normalizing(self):
        data = self.process_data()

        for i in range(0, data.shape[1], self.config["data"]["configuration"]["num_feature"]):
            temp = np.array(data[:, i+self.config["data"]["feature_position"]["water_tempreture"]])
            temp = (temp - self.config["data"]["configuration"]["min_value"]) / (self.config["data"]["configuration"]["max_value"] - self.config["data"]["configuration"]["min_value"])
            data[:, i+self.config["data"]["feature_position"]["water_tempreture"]] = temp.tolist()

        return data
    

    def create_random_window(self):

        data = self.normalizing()
        window_data = []
        for j in range (0, data.shape[1], self.config["data"]["configuration"]["num_feature"]):
            for i in range(0, self.config["data"]["configuration"]["max_row"] - self.config["data"]["configuration"]["window_size"] + 1, self.config["data"]["configuration"]["step"]):
                window_data.append(data[i : i + self.config["data"]["configuration"]["window_size"], j : j + self.config["data"]["configuration"]["num_feature"]])

        window_data = np.array(window_data)
        return window_data 


    def labeled_data(self):
        window_data = self.create_random_window()

        x = []
        y = []

        for i in range (int(len(window_data))):
            x.append(window_data[i][:,(self.config["data"]["feature_position"]["water_tempreture"], self.config["data"]["feature_position"]["fan_state"])])
            y.append(np.mean(window_data[i][:,self.config["data"]["feature_position"]["flag"]]))

        x = np.array(x)
        y = np.array(y)
        return x,y



    def split_data(self):
            
        x,y = self.labeled_data()

        indices = np.arange(len(x))  
        np.random.shuffle(indices)  # Shuffle indices

        # Apply shuffled indices
        x = x[indices]
        y = y[indices]

        # Compute split index
        split_idx = int((1 - self.config["training"]["test_size"]) * len(x))

        # Manual split
        x_train, x_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Save Test data in npz type
        np.savez(self.config["output"]["path"] + "/test_data.npz", x_test=x_test, y_test=y_test)
        np.savez(self.config["output"]["path"] + "/train_data.npz", x_train=x_train, y_train=y_train)

        return (x_train, y_train)