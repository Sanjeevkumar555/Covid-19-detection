## Covid-19-detection

**Objective** : To correctly label a person as Covid+ based on their symptomps and demographics.

* **Data collection** : Dataset available from Kaggle entitled
                        “COVID-19 Symptoms and Presence”. This dataset has 20 attributes that are possible
                        factors related to acquiring the virus, and 1 class attribute that determines the presence of
                        COVID-19.
* **DATA Pre-Processing** : It being a highly imbalanced Data, sampling techniques were utlized to arrive at a reasonalble accuracy.

* **Algorithm applied** : Ensemble Techniques , SVM

* **Deployment** : use Flask to build the API which will call the model and serve in real time based on the feature provided .
