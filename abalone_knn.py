import pandas as pd
import numpy as np
import scipy.stats as ss
import math


class ReadData():
    def __init__(self, training_set, target_set):
        self._abalone_data = pd.read_csv(training_set)
        self._target_data = pd.read_csv(target_set)
        self._rows, self._cols = self._abalone_data.shape
        self._target_rows, self._target_cols = self._target_data.shape

    def get_r_squared(self, x_values:[int], y_values:[int]) -> float:
        '''
        returns the r squared value for the given data
        '''
        slope, intercept, r_value, p_value, std_err = ss.linregress(x_values[:, 0], y_values[:, 0])
        return np.square(r_value)

    def get_best_data(self) -> (int, int):
        '''
        iterates through all the attributes and returns a tuple of the data that has the highest 
        r squared value 
        '''
        data_quality = {}
        for i in range(1, self._cols-1):
            for j in range(i+1, self._cols-1):
                x_values = np.array(self._abalone_data.iloc[0:self._rows, i:i+1], dtype = float)
                y_values = np.array(self._abalone_data.iloc[0:self._rows, j:j+1], dtype = float)
                r_squared = self.get_r_squared(x_values, y_values)
                pair = (i, j)
                data_quality[pair] = r_squared
        
        best = max(data_quality, key = lambda x: data_quality[x])
        return best
        
        

    def get_best_predictions(self):
        '''
        predicts how many rings an abalone has using the attributes with the highest r squared value 
        prints out how accurate it was
        '''
        x, y = self.get_best_data()
        
        predictions = self.predict_churn(self._target_data.iloc[0:self._rows, x:x+1], self._target_data.iloc[0:self._rows, y:y+1], \
            self._abalone_data.iloc[0:self._rows, x:x+1], self._abalone_data.iloc[0:self._rows, y:y+1], self._abalone_data['Rings'], 100)
        print(f"This knn classifier has a {self.get_accuracy(predictions)}% accuracy")
        

    def get_accuracy(self, predictions:[float]) -> float:
        '''
        returns the average accuracy of each prediction
        '''
        target = np.array(self._target_data['Rings'])
        differences = np.zeros(self._target_rows)
        for prediction in range(self._target_rows):
            differences[prediction] = abs(target[prediction] - predictions[prediction])/target[prediction]

        return (1 - np.mean(differences))*100


    def distance(self, point1, point2):
        '''
        calculates the distance between 2 points 
        '''
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt(np.square(x1-x2) + np.square(y1-y2))

    def predict_churn(self, target_1, target_2, training_1, training_2, classification, number_neighbors):
        '''
        given the target set and training set, this function returns a list of the classifications
        for each value pair in the target set. 
        '''
        target_1 = np.array(target_1, dtype = float)[:, 0]
        target_2 = np.array(target_2, dtype = float)[:, 0]
        training_1 = np.array(training_1, dtype = float)[:, 0]
        training_2 = np.array(training_2, dtype = float)[:, 0]
        classification = np.array(classification, dtype = float)
        

        predictions = np.zeros(len(target_1), dtype = float)
        distances = np.zeros(self._rows)
        
        for target in range(self._target_rows):
            point1 = (target_1[target], target_2[target])
            for customer in range(self._rows):
                point2 = (training_1[customer], training_2[customer])
                distances[customer] = self.distance(point1, point2)

            indices = np.argsort(distances)
            indices = indices[:number_neighbors]
            sum = 0
            for abalone in indices:
                sum += classification[abalone]
            sum = sum/number_neighbors
            predictions[target] = sum
            
        return predictions






def run():
    file_name = 'abalone.csv'
    target_name = 'abalone_target.csv'
    data = ReadData(file_name, target_name)
    
    data.get_best_predictions()
run()
