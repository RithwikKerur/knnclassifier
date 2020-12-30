# knnclassifier

This is my first attempt at creating a knn classifier to classify two types of data(quantitative and qualitative)

Qualitative Data:

The knn classifier predicts whether a customer is likely to churn their credit card. Currently it has an accuracy of 92%.

To use this program, the user needs to have two csv files(churn.csv and churn_target.csv). The files also need to be formatted to have the correct headers for this program to run properly. Furthermore the user should make sure that the .csv files are in the same directory as the python file and the working directory for the interpreter is the location of the file. Alternately, the user can use the full file name path for the two .csv files. 

Quantitative Data:

The knn classifier predicts how many rings an abalone will have given certain attributes about the abalone such as weight, diameter, length. 

To use this program, the user needs to have two csv files(abalone.csv and abalone_target.csv). The files have a "Rings" header and two other headers. urthermore the user should make sure that the .csv files are in the same directory as the python file and the working directory for the interpreter is the location of the file. Alternately, the user can use the full file name path for the two .csv file

Currently this program has an accuracy rate of 83.61%. To increase this accuracy, I plan to first group the abalone into the different sexes before using the knn classifier. 
