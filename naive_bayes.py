#-------------------------------------------------------------------------
# AUTHOR: Andy Wu
# FILENAME: naive_bayes.py
# SPECIFICATION: Reads weather training and test files
# Trains Naive Bayes classifier and outputs 
# classification of each test instance where confidence
# is at least 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: Too Long
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array
#X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2,
#2], ...]]
#--> add your Python code here
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}
label_map = {'Yes': 1, 'No': 2}
inverse_label_map = {1: 'Yes', 2: 'No'}

X = []
for data in dbTraining:
    X.append([outlook_map[data[1]], temperature_map[data[2]], humidity_map[data[3]], wind_map[data[4]] ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for data in dbTraining:
    Y.append(label_map[data[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())
    
#Printing the header os the solution
#--> add your Python code here
print("Day | Outlook | Temperature | Humidity | Wind | PlayTennis | Confidence")

#Use your test samples to make probabilistic predictions. For instance:
#clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for data in dbTest:
    test_instance = [[outlook_map[data[1]], temperature_map[data[2]], humidity_map[data[3]], wind_map[data[4]]]]

    probs = clf.predict_proba(test_instance)[0]
    predicted_class = clf.predict(test_instance)[0]

    confidence = max(probs)

    if confidence >= 0.75:
        print(
            f"{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} "
            f"{inverse_label_map[predicted_class]} {confidence:.2f}"
        )
