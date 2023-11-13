import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')


def fitness_function(ga_instance, solution, solution_idx):
    print(solution)
    training_dataset = pd.read_csv('training_dataset.csv')

    conditions = [
        training_dataset['dCSI'] < solution[0],
        (training_dataset['dCSI'] >= solution[0]) & (training_dataset['dCSI'] < solution[1]),
        (training_dataset['dCSI'] >= solution[1]) & (training_dataset['dCSI'] < solution[2]),
        training_dataset['dCSI'] >= solution[2]
    ]

    values = [0, 1, 2, 3]

    training_dataset['dCSI_cat'] = np.select(conditions, values, default=3)

    training_dataset.to_csv('training_dataset.csv', index=False)

    y = training_dataset['dCSI_cat']
    X = training_dataset.drop(['dCSI_cat', 'OID_', 'pointid', 'grid_code', 'dCSI'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 2)

    # Training and Prediction
    rf_classifier = RandomForestClassifier(n_estimators = 50).fit(X_train, y_train)
    prediction = rf_classifier.predict(X_test)

    # importance = rf_classifier.feature_importances_

    # feature_names = ['NDVI', 'NDVIre1n', 'NDVIre2n', 'NDVIre3n', 'Elevation', 'Slope', 'Aspect']

    # importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(f'Iteration {solution_idx}:')
    print(f"Training Accuracy: {rf_classifier.score(X_train, y_train)}")
    # print(confusion_matrix(y_test, prediction))
    print(f"Testing Accuracy: {accuracy_score(y_test, prediction)}")
    print('\n')

  
    return accuracy_score(y_test, prediction)


num_generations = 10
num_parents_mating = 4
sol_per_pop = 8
num_genes = 3  
mutation_type = "random"
mutation_percent_genes = 10


# var_bounds = [(-0.05, 0.1), (0.15, 0.3), (0.35, 0.65)]

init_range_low = 0
init_range_high = 1

# initial_population = []
# for _ in range(sol_per_pop):
#     initial_population.append(np.random.uniform(var_bounds[0][0], var_bounds[0][1], num_genes))

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes
                      )


ga_instance.run()

solution, fitness = ga_instance.best_solution()[0], ga_instance.best_solution()[1]

print("\n\nBest Solution:", solution)
print("Best Fitness:", fitness)
