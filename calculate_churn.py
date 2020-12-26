import numpy as np
from numpy import random
import scipy.stats as ss
import matplotlib.pyplot as plt


class ReadData():
    def __init__(self, file_name, target_file, data_size = 667, target_size = 201):
        self._data_size = data_size
        self._target_size = target_size
        self._account_length = np.zeros(self._data_size, dtype = float)
        self._total_day_minutes = np.zeros(self._data_size, dtype = float)
        self._total_day_charge = np.zeros(self._data_size, dtype = float)
        self._total_eve_minutes = np.zeros(self._data_size, dtype = float)
        self._total_eve_charge = np.zeros(self._data_size, dtype = float)
        self._total_intl_minutes = np.zeros(self._data_size, dtype = float)
        self._total_intl_calls = np.zeros(self._data_size, dtype = float)
        self._total_intl_charge = np.zeros(self._data_size, dtype = float)
        self._customer_service_calls = np.zeros(self._data_size, dtype = float)
        self._churn = np.zeros(self._data_size, dtype = float)
        self._data = np.array([self._account_length, self._total_day_minutes, self._total_day_charge, self._total_eve_minutes, self._total_eve_charge, \
            self._total_intl_minutes, self._total_intl_calls,  self._total_intl_charge, self._customer_service_calls, self._churn])


        self._target_account_length = np.zeros(self._target_size, dtype = float)
        self._target_total_day_minutes = np.zeros(self._target_size, dtype = float)
        self._target_total_day_charge = np.zeros(self._target_size, dtype = float)
        self._target_total_eve_minutes = np.zeros(self._target_size, dtype = float)
        self._target_total_eve_charge = np.zeros(self._target_size, dtype = float)
        self._target_total_intl_minutes = np.zeros(self._target_size, dtype = float)
        self._target_total_intl_calls = np.zeros(self._target_size, dtype = float)
        self._target_total_intl_charge = np.zeros(self._target_size, dtype = float)
        self._target_customer_service_calls = np.zeros(self._target_size, dtype = float)
        self._target_churn = np.zeros(self._target_size, dtype = float)
        self._target_data = np.array([self._target_account_length, self._target_total_day_minutes, self._target_total_day_charge, \
            self._target_total_eve_minutes, self._target_total_eve_charge, self._target_total_intl_minutes, \
                self._target_total_intl_calls, self._target_total_intl_charge, self._target_customer_service_calls, self._target_churn])
        with open(file_name, 'r') as file:
            first = True
            line = 0
            for lines in file:
                if not first:
                    lines = lines.split(';')
                    self._account_length[line] = float(lines[0])
                    self._total_day_minutes[line] =float(lines[1])
                    self._total_day_charge[line] = float(lines[2])
                    self._total_eve_minutes[line] = float(lines[3])
                    self._total_eve_charge[line] = float(lines[4])
                    self._total_intl_minutes[line] = float(lines[5])
                    self._total_intl_calls[line] = float(lines[6])
                    self._total_intl_charge[line] = float(lines[7])
                    self._customer_service_calls[line] = float(lines[8])
                    self._churn[line] = float(lines[9])
                    line += 1
                else:
                    first = False
        
        with open(target_file, 'r') as target:
            line = 0
            first = True
            for lines in target:
                if not first:
                    lines = lines.split(';')
                    self._target_account_length[line] = float(lines[0])
                    self._target_total_day_minutes[line] =float(lines[1])
                    self._target_total_day_charge[line] = float(lines[2])
                    self._target_total_eve_minutes[line] = float(lines[3])
                    self._target_total_eve_charge[line] = float(lines[4])
                    self._target_total_intl_minutes[line] = float(lines[5])
                    self._target_total_intl_calls[line] = float(lines[6])
                    self._target_total_intl_charge[line] = float(lines[7])
                    self._target_customer_service_calls[line] = float(lines[8])
                    self._target_churn[line] = float(lines[9])
                    line += 1
                else:
                    first = False
                

    def get_best_points(self):
        
        
        predictions = self.predict_churn(self._target_total_day_minutes, self._target_total_eve_minutes,\
             self._total_day_minutes,self._total_eve_minutes, 10)   
        
        correctness = np.mean(predictions == self._target_churn)*100
        print(correctness)




    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt(np.square(x1-x2) + np.square(y1-y2))


   


    def predict_churn(self, target_1, target_2, training_1, training_2, number_neighbors):
        '''
        given the target set and training set, this function returns a list of the classifications
        for each value pair in the target set. 
        '''
        predictions = np.zeros(len(target_1), dtype = bool)
        distances = np.zeros(self._data_size)
        for target in range(len(target_1)):
            point1 = (target_1[target], target_2[target])
            for customer in range(len(training_1)):
                point2 = (training_1[customer], training_2[customer])
                distances[customer] = self.distance(point1, point2)

            indices = np.argsort(distances)
            indices = indices[:number_neighbors]    #10 is the number of nearest elements we are looking at
            churn = self._churn[indices]
            if np.count_nonzero(churn) == number_neighbors/2:
                choice = random.choice([0,1])
                predictions[target] = choice
            elif np.count_nonzero(churn)> number_neighbors/2:
                predictions[target] = True
            else:
                predictions[target] = False
        
        return predictions
        

    def graph_data(self, training_1, training_2, point_target):
        churn_1 = []
        churn_2 =[]
        not_churn_1 = []
        not_churn_2 = []

        for data in range(len(training_1)):
            if self._churn[data]:
                churn_1.append(training_1[data])
                churn_2.append(training_2[data])
            else:
                not_churn_1.append(training_1[data])
                not_churn_2.append(training_2[data])


        plt.plot(churn_1, churn_2, 'ro')
        plt.plot(not_churn_1, not_churn_2, 'go')
        plt.plot(point_target[0], point_target[1], 'bo')
        plt.axis([min(training_1), max(training_1), min(training_2), max(training_2)])
        plt.show()

    


if __name__ == '__main__':
    file = 'C:\Churn\src\churn.csv'
    target_file = 'C:\Churn\src\churn_target.csv'
    data = ReadData(file, target_file)
    
    data.get_best_points()