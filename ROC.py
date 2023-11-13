import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')


dIndices_ranges = [-0.4, 0.09, 0.338]

# mother_dataset = pd.read_csv('new_dataset.csv')

# training_dataset = mother_dataset[['OID_', 'pointid', 'grid_code', 'NDVI', 'NDVIre1n', 'NDVIre2n', 'NDVIre3n', 'NDBI', 'NBR', 'NBR2', 'CSI', 'BSI', 'Elevation', 'Slope', 'Aspect', 'dCSI']]

training_dataset_path = 'RF_dIndices_dataset/08_11_23_dCSI_Dataset.csv'
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

y = training_dataset[parameter_cat]
X = training_dataset.drop([parameter_cat, 'OID_', 'pointid', 'grid_code', parameter_to_train], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2)

# Training and Prediction
rf_classifier = RandomForestClassifier(n_estimators = 50).fit(X_train, y_train)
prediction = rf_classifier.predict(X_test)

importance = rf_classifier.feature_importances_

feature_names = ['NDVIre1n', 'NDVIre2n', 'NDVIre3n', 'CSI', 'BSI', 'NBR', 'NBR2', 'Elevation', 'Aspect', 'Slope', 'NDVI_NDBI']

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

importance_df = importance_df.sort_values(by='Importance', ascending=False)


print(f"Training Accuracy: {rf_classifier.score(X_train, y_train)}")
# print(confusion_matrix(y_test, prediction))
print(f"\nTesting Accuracy: {accuracy_score(y_test, prediction)}")
print(classification_report(y_test, prediction))

print('\n')
print(importance_df)

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

print(y_onehot_test)

RocCurveDisplay.from_predictions(
    y_onehot_test[:, 2],
    prediction,
    name="Moderate vs the rest",
    color="darkorange",
    
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nModerate vs (Very High, High and Low)")
plt.legend()
plt.show()


from sklearn.metrics import roc_auc_score

micro_roc_auc_ovr = roc_auc_score(
    y_test,
    prediction,
    multi_class="ovr",
    average="micro",
)

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")