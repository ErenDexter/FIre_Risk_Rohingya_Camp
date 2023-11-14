import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import seaborn as sns


SVM = SVC(kernel='linear')

dIndices_ranges = [-0.0859, 0.153, 0.2529]

# mother_dataset = pd.read_csv('new_dataset.csv')

# training_dataset = mother_dataset[['OID_', 'pointid', 'grid_code', 'NDVI', 'NDVIre1n', 'NDVIre2n', 'NDVIre3n', 'NDBI', 'NBR', 'NBR2', 'CSI', 'BSI', 'Elevation', 'Slope', 'Aspect', 'dCSI']]

training_dataset_path = 'SVM_dIndices_dataset/08_11_23_dCSI_Dataset.csv'
parameter_to_train = 'dCSI'
parameter_cat = 'dCSI_cat'

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

# sns.pairplot(data=training_dataset, hue='dCSI_cat', palette='Set2')

y = training_dataset[parameter_cat]
X = training_dataset.drop([parameter_cat, 'OID_', 'pointid', 'grid_code', parameter_to_train], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2)

model = SVM.fit(X_train, y_train)

pred = model.predict(X_test)

report = classification_report(y_test, pred, output_dict=False)
print(report)

#print(report['weighted avg']['precision'])
