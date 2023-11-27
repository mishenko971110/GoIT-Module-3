import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def from_csv_to_dataframe(folder_name):
    file_paths = glob.glob('data/' + folder_name + '/*.csv')
    combined_data = pd.DataFrame()
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

idle_data = from_csv_to_dataframe('idle')
running_data = from_csv_to_dataframe('running')
stairs_data = from_csv_to_dataframe('stairs')
walking_data = from_csv_to_dataframe('walking')

print('Датасет idle:\n', idle_data.head())
print('Датасет running:\n', running_data.head())
print('Датасет stairs:\n', stairs_data.head())
print('Датасет walking:\n', walking_data.head())

def time_domain_features(data):
    time_features = pd.DataFrame()
    time_features['mean'] = np.round(data.mean(), 4)
    time_features['std'] = np.round(data.std(), 4)
    time_features['max'] = np.round(data.max(), 4)
    time_features['min'] = np.round(data.min(), 4)
    return time_features

time_features_idle = time_domain_features(idle_data)
time_features_running = time_domain_features(running_data)
time_features_stairs = time_domain_features(stairs_data)
time_features_walking = time_domain_features(walking_data)

print('Часові ознаки idle: \n', time_features_idle)
print('Часові ознаки running: \n', time_features_running)
print('Часові ознаки stairs: \n', time_features_stairs)
print('Часові ознаки walking: \n', time_features_walking)

def plot_time_domain_features(time_features, activity_name):
    plt.figure(figsize=(10, 6))
    plt.title(f'Time Domain Features for {activity_name}')
    plt.bar(time_features.columns, time_features.iloc[0], color='skyblue')
    plt.xlabel('Time Features')
    plt.ylabel('Values')
    plt.grid(axis='y')
    plt.show()

plot_time_domain_features(time_features_idle, 'Idle')
plot_time_domain_features(time_features_running, 'Running')
plot_time_domain_features(time_features_stairs, 'Stairs')
plot_time_domain_features(time_features_walking, 'Walking')

X_idle = time_features_idle
X_running = time_features_running
X_stairs = time_features_stairs
X_walking = time_features_walking

y_idle = np.zeros(X_idle.shape[0])
y_running = np.ones(X_running.shape[0])
y_stairs = np.full(X_stairs.shape[0], 2)
y_walking = np.full(X_walking.shape[0], 3)

X = np.vstack([X_idle, X_running, X_stairs, X_walking])
y = np.hstack([y_idle, y_running, y_stairs, y_walking])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('Розмір навчального набору: ', X_train.shape, 'Мітки: ', y_train.shape)
print('Розмір тестувального набору: ', X_test.shape, 'Мітки: ', y_test.shape)

svm_model = SVC(kernel='rbf', C=10)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

svm_accuracy = svm_model.score(X_test, y_test)
rf_accuracy = rf_model.score(X_test, y_test)

print('Точність моделі SVM: ', svm_accuracy)
print('Точність моделі Random Forest: ', rf_accuracy)

svm_model = SVC(kernel='linear')
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

svm_scores = precision_recall_fscore_support(y_test, svm_predictions, average='weighted', zero_division=0)
rf_scores = precision_recall_fscore_support(y_test, rf_predictions, average='weighted', zero_division=0)

print('Точність моделі SVM: ', np.round(svm_accuracy, 4))
print('Точність моделі Random Forest: ', np.round(rf_accuracy, 4))
print('Precision, Recall, F1-score для SVM: ', svm_scores)
print('Precision, Recall, F1-score для Random Forest: ', rf_scores)