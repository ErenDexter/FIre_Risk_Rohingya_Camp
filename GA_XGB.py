import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pygenetics import Population

import warnings
warnings.filterwarnings('ignore')

XGB = XGBClassifier()

training_dataset_path = 'XGB_dIndices_dataset/08_11_23_dNBR2_Dataset.csv'
parameter_to_train = 'dNBR2'
parameter_cat = 'dNBR2_cat'

def fitness_function(solution):
    if (solution[2] > solution[1] and solution[1] > solution[0]):
        print(solution)
        training_dataset = pd.read_csv(training_dataset_path)

        conditions = [
            training_dataset[parameter_to_train] < solution[0],
            (training_dataset[parameter_to_train] >= solution[0]) & (training_dataset[parameter_to_train] < solution[1]),
            (training_dataset[parameter_to_train] >= solution[1]) & (training_dataset[parameter_to_train] < solution[2]),
            training_dataset[parameter_to_train] >= solution[2]
        ]

        values = [0, 1, 2, 3]

        training_dataset[parameter_cat] = np.select(conditions, values, default=3)

        training_dataset.to_csv(training_dataset_path, index=False)

        y = training_dataset[parameter_cat]
        print(np.unique(y))
        if len(np.unique(y)) != 4:
            return 1000
        X = training_dataset.drop([parameter_cat, 'OID_', 'pointid', 'grid_code', parameter_to_train], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2)

        # Training and Prediction
        model = XGB.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # importance = rf_classifier.feature_importances_

        # feature_names = ['NDVI', 'NDVIre1n', 'NDVIre2n', 'NDVIre3n', 'Elevation', 'Slope', 'Aspect']

        # importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

        # importance_df = importance_df.sort_values(by='Importance', ascending=False)
        # print(f"Training Accuracy: {rf_classifier.score(X_train, y_train)}")
        # # print(confusion_matrix(y_test, prediction))
        # print(f"Testing Accuracy: {accuracy_score(y_test, prediction)}")
        # print('\n')

    
        return accuracy_score(y_test, y_pred)
    else:
        return 1000


pop = Population(10, fitness_function)

# Parameter Boundary
dataframe = pd.read_csv(training_dataset_path)
min_value = dataframe[parameter_to_train].min()
max_value = dataframe[parameter_to_train].max()
print(min_value, max_value)
pop.add_param(float(min_value), float(max_value)) 
pop.add_param(float(min_value), float(max_value)) 
pop.add_param(float(min_value), float(max_value)) 

pop.initialize()
i = 1;
for _ in range(25):
    pop.next_generation(p_crossover=0.75, p_mutation=0.05)
    print('Average fitness: {}'.format(pop.average_fitness))
    print('Average obj. fn. return value: {}'.format(pop.average_ret_val))
    print('Best fitness score: {}'.format(pop.best_fitness))
    print('Best obj. fn. return value: {}'.format(pop.best_ret_val))
    print('Best parameters: {}\n'.format(pop.best_params))
    print(i)
    i += 1





