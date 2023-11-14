import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

XGB = XGBClassifier()

dIndices_ranges = [-0.0449, 0.0559, 0.0881]

# mother_dataset = pd.read_csv('new_dataset.csv')

# training_dataset = mother_dataset[['OID_', 'pointid', 'grid_code', 'NDVI', 'NDVIre1n', 'NDVIre2n', 'NDVIre3n', 'NDBI', 'NBR', 'NBR2', 'CSI', 'BSI', 'Elevation', 'Slope', 'Aspect', 'dCSI']]

training_dataset_path = 'XGB_dIndices_dataset/08_11_23_dNDVI_dNDBI_Avg_Dataset.csv'
parameter_to_train = 'dNDVI_dNDBI_divided_by_2'
parameter_cat = 'dNDVI_dNDBI_divided_by_2_cat'

training_dataset = pd.read_csv(training_dataset_path)


conditions = [
    training_dataset[parameter_to_train] < dIndices_ranges[0],
    (training_dataset[parameter_to_train] >= dIndices_ranges[0]) & (training_dataset[parameter_to_train] < dIndices_ranges[1]),
    (training_dataset[parameter_to_train] >= dIndices_ranges[1]) & (training_dataset[parameter_to_train] < dIndices_ranges[2]),
    training_dataset[parameter_to_train] >= dIndices_ranges[2]
]

values = [0, 1, 2, 3]

training_dataset[parameter_cat] = np.select(conditions, values, default=3)

training_dataset.to_csv(training_dataset_path, index=False)

y = training_dataset[parameter_cat]
X = training_dataset.drop([parameter_cat, 'OID_', 'pointid', 'grid_code', parameter_to_train], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2)

model = XGB.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nTesting Accuracy: {accuracy_score(y_test, y_pred)}")

report = classification_report(y_test, y_pred, output_dict=False)
print(report)


plot_importance(model)
pyplot.show()

#sns.pairplot(data=X, hue='dCSI', palette='Set2')

