import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

%matplotlib inline
plt.rcParams['figure.figsize'] = (14.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

allpatients = pd.read_csv('./data/male_a_1.csv')
allpatients.head()

features = allpatients.drop(labels='Status',axis=1).columns
for i in features:
    allpatients[i] = (allpatients[i] - np.mean(allpatients[i]))/np.std(allpatients[i])
allpatients.head()

status = pd.get_dummies(allpatients['Status'],drop_first=True)

allpatients.drop(['Status'],axis=1,inplace=True)

allpatients['Pathology'] = status
allpatients.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(allpatients.drop(['Pathology'], axis=1),
                                                    allpatients['Pathology'],
                                                    test_size=0.25,
                                                    random_state=42)

num_folds = 5

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)


from sklearn.linear_model import LogisticRegression

accuracies = {}
C = [1e-2, 4e-2, 8e-2, 1e-1, 2e-1, 4e-1, 6e-1, 8e-1, 1, 1.5, 2]

for c in C:
    for i in range(num_folds):
        logModel = LogisticRegression(penalty='l1', C=c, solver='liblinear')

        train_set = np.concatenate(X_train_folds[:i] + X_train_folds[i + 1:])
        labels_set = np.concatenate(y_train_folds[:i] + y_train_folds[i + 1:])

        logModel.fit(train_set, labels_set)

        y_val_pred = logModel.predict(X_train_folds[i])
        val_acc = np.mean(y_val_pred == y_train_folds[i])

        if c in accuracies:
            accuracies[c].extend([val_acc])
        else:
            accuracies[c] = [val_acc]

for c in C:
    acc = accuracies[c]
    plt.scatter([c] * len(acc), acc)

accuracies_mean = np.array([np.mean(v) for k, v in sorted(accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(accuracies.items())])
plt.errorbar(C, accuracies_mean, yerr=accuracies_std)
plt.xlabel('Regularization', fontsize=14)
plt.ylabel('Accuracy of the validation set', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

log_model = LogisticRegression(penalty='l1', C=0.6, solver='liblinear')
log_model.fit(X_train, y_train)
prediction = log_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,prediction))

