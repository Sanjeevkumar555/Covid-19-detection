For the data collection, the researchers used a dataset available from Kaggle entitled
“COVID-19 Symptoms and Presence”. This dataset has 20 attributes that are possible
factors related to acquiring the virus, and 1 class attribute that determines the presence of
COVID-19.
DATA Pre-Processing:
It being a  class imbalanced data , I use SMOTE technique 
which creates equal classes based on oversampling.
Feature Extraction:
I find the correlation between the features and my output classes
and finally discard the  <= 0 features.
MODEL BUILDING:
I ran 5  algorithms on the train model namely-Decision Tree, Random  Forest, 
SVM with  Linear Kernel and SVM  with RBF kernel and lastly KNN.
I  got the highest accuracy with SVM (RBF) also with the most fast training.
Hence , I build the model according to  the SVM algo.
