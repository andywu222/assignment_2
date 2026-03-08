#-------------------------------------------------------------------------
# AUTHOR: Andy Wu
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and test decision tree classifiers using three
# training datasets and one test dataset through
# entropy and max_depth = 5
# Repeat training 10 times and print average accuracy for each file
# FOR: CS 4210- Assignment #2
# TIME SPENT: Too Long
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv',
'contact_lens_training_3.csv']

age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'No': 1, 'Yes': 2}
tear_map = {'Reduced': 1, 'Normal': 2}
label_map = {'Yes': 1, 'No': 2}

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())
    
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    df_training = pd.read_csv(ds)
    for _, row in df_training.iterrows():
        dbTraining.append(row.tolist())
    #Transform the original categorical training features to numbers and add to the
#4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1],
#[2, 2, 2, 2], ...]]
    #--> add your Python code here
    for _, row in df_training.iterrows():
        X.append([age_map[row['Age']], spectacle_map[row['Spectacle Prescription']], astigmatism_map[row['Astigmatism']], tear_map[row['Tear Production Rate']]])
    
    #Transform the original categorical training classes to numbers and add to the
    #vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for _, row in df_training.iterrows():
        Y.append(label_map[row['Recommended Lenses']])

    accuracies = []
    
    #Loop your training and test tasks 10 times here
    for i in range (10):
        
        # fitting the decision tree to the data using entropy as your impurity
        #measure and maximum depth = 5
        # --> addd your Python code here
        # clf =
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=i)
        clf = clf.fit(X, Y)

        correct = 0
        #Read the test data and add this data to dbTest
        #--> add your Python code here
        
        for data in dbTest:
            #Transform the features of the test instances to numbers following the
            #same strategy done during training,
            #and then use the decision tree to make the class prediction. For
            #instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so
            #that you can compare it with the true label
            #--> add your Python code here
            test_features = [age_map[data[0]], spectacle_map[data[1]], astigmatism_map[data[2]], tear_map[data[3]]]

            #Compare the prediction with the true label (located at data[4]) of the
            #test instance to start calculating the accuracy.
            #--> add your Python code here
            class_predicted = clf.predict([test_features])[0]
            true_label = label_map[data[4]]

            if class_predicted == true_label:
                correct += 1
            
        #Find the average of this model during the 10 runs (training and test set)
        #--> add your Python code here
        accuracy = correct / len(dbTest)
        accuracies.append(accuracy)
    #Print the average accuracy of this model during the 10 runs (training and test
    #set).    
    #Your output should be something like that: final accuracy when training on
    #contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f'Final accuracy when training on {ds}: {average_accuracy:.4f}')
