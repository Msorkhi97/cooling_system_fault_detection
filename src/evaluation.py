import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras



class TrainModel():
    def __init__(self,config):
        self.config = config

    
    def classifire(self):
        test_data = np.load("test_data.npz")
        x_test = test_data["x_test"]
        y_test = test_data["y_test"]


        model = keras.models.load_model("best_model.keras")
        predictions = model.predict(x_test)

        healthy_threshold = 0.5
        result_list = []
        problem_list = []
        for indice in range(len(x_test)):
            if y_test[indice] == 0:
                if predictions[indice] < healthy_threshold:
                    result_list.append(indice)
                else:
                    problem_list.append(indice)
            elif y_test[indice] == 1:
                if predictions[indice] >= healthy_threshold:
                    result_list.append(indice)
                else:
                    problem_list.append(indice)


        print(int(len(result_list))/(int(len(x_test))))


        # Plot And Test Choosen Data
        """ i = 13
        print(model.predict(x_test[i].reshape(1,self.WINDOW_SIZE,self.NUM_FEATURE)),"==>",y_test[i])
        plt.plot(x_test[0][:,0])
        plt.show() """


TrainModel().train()