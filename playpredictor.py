'''
Classifier :- K Nearest Neighbour 
Dataset    :- Play Predictor Dataset
Features   :- Whether , Temperature
Labels     :-Yes , No
Training Dataset :- 30 Entries
Testing dataset :- 1 Entry
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def SpPlayPredictor(data_path):
    #step 1:- Load data
    data=pd.read_csv(data_path,index_col=0)

    print("Size of actual dataset :",len(data))

    #step 2:- Clean and prepare data
    features_names=['Whether','Temperature']

    print("Names Of Features :",features_names)

    whether=data.Whether
    Temperature=data.Temperature
    play=data.Play

    #Creating LabelEncoder
    le=preprocessing.LabelEncoder()

    #Converting string labels into numbers
    weather_encoded=le.fit_transform(whether)
    print(weather_encoded)

    #Converting string labels into numbers
    temp_encoded=le.fit_transform(Temperature)
    label=le.fit_transform(play)
    print(temp_encoded)
    print(label)

    #combining Weather and temperature into single listof tupples
    features=list(zip(weather_encoded,temp_encoded))

    #step 3:- Train the data 
    model=KNeighborsClassifier(n_neighbors=3)

    #Train the model
    model.fit(features,label)

    #step 4:- Test the data
    predicted=model.predict([[1,2]])    #0:-Overcast  , 2:- Mild
    print(predicted)


def main():
    print("Machine Learning algorithm")
    print("Play predictor case study using knn")
    SpPlayPredictor("PlayPredictor.csv")


if __name__ =="__main__":
    main()
