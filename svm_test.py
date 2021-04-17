import csv
import math
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics


VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']
FEATURES = ['Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate',
            'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2',
            'Glucose', 'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2',
            'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']


# average 12-row features
def generate_average(csv_read, csv_write):
    rfile = open(csv_read, 'r')
    wfile = open(csv_write, 'w')
    with rfile:
        reader = csv.reader(rfile)
        writer = csv.writer(wfile)
        count = -1
        for row in reader:
            # skip first row
            if count == -1:
                count = 0
                group = []
                writer.writerow(row)
                continue
            # start with 1 to 12
            count += 1
            if count < 13:
                temp = []
                for col in row:
                    temp.append(float(col))
                group.append(temp)
                if count == 12:
                    output = []
                    for i in range(len(row)):
                        num_nan = 0
                        temp_sum = 0
                        # check number of nan
                        for j in range(12):
                            if math.isnan(group[j][i]):
                                num_nan += 1
                            else:
                                temp_sum += group[j][i]
                        # keep nan for all nan column
                        if num_nan == 12:
                            output.append(float('nan'))
                        else:
                            output.append(round(temp_sum/(12 - num_nan), 2))
                    writer.writerow(output)
                    count = 0
                    group = []


# drop rows in feature_label that has too many nan
def drop_nan(features, labels, percent, write):
    # concat feature and label
    feature_labels = pd.concat([features, labels], axis=1)
    print(feature_labels.head())
    print('size before drop:', feature_labels.shape)
    feature_labels = feature_labels.dropna(axis=0, thresh=percent*features.shape[1]+labels.shape[1])
    print('size after drop:', feature_labels.shape, '\n')
    if write:
        feature_labels.to_csv(r'average_drop_features.csv', na_rep='nan', index=False, header=True)
    return feature_labels


# fill nan with column average
def fill_nan(average, write):
    average_mean = average.mean()
    average.fillna(round(average_mean, 2), inplace=True)
    if write:
        average.to_csv('average_drop_mean_train_data', index=False, header=True)
    return average


# svm training for binary classification
def binary_predict(patients, X_test):
    # get training features
    X_train = patients[FEATURES]
    # data normalization
    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(X_train)
    x_test_data = scaler.transform(X_test)
    # parameter initial
    predict = []
    flag = 0
    # training & testing for all binary labels
    for binary_label in TESTS:
        # get training labels
        y_train = patients[binary_label]
        # linear classifier
        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(x_train_data, y_train)
        # predict with probability
        y_pred = clf.predict_proba(x_test_data)
        y_pred = y_pred[:, 1]
        # append y_pred
        if flag == 0:
            np.reshape(y_pred, (len(y_pred), 1))
            predict = y_pred
            flag = 1
            print(predict.shape)
        else:
            predict = np.c_[predict, y_pred]
            print(predict.shape)
    # write numpy array to csv
    DF = pd.DataFrame(np.round(predict, 3), columns=TESTS)
    DF.to_csv('predict_binary.csv', index=False, header=True)
    return DF


def regress_predict(patients, X_test):
    # get training features
    X_train = patients[FEATURES]
    # data normalization
    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(X_train)
    x_test_data = scaler.transform(X_test)
    # parameter initial
    predict = []
    flag = 0
    # training & testing for all regression labels
    for regress_label in VITALS:
        # get training labels
        y_train = patients[regress_label]
        # linear ridge regression
        clf = linear_model.Ridge(alpha=1.0)
        clf.fit(x_train_data, y_train)
        y_pred = clf.predict(x_test_data)
        # append y_pred
        if flag == 0:
            np.reshape(y_pred, (len(y_pred), 1))
            predict = y_pred
            flag = 1
            print(predict.shape)
        else:
            predict = np.c_[predict, y_pred]
            print(predict.shape)
            # write numpy array to csv
    DF = pd.DataFrame(np.round(predict, 3), columns=VITALS)
    DF.to_csv('predict_regress.csv', index=False, header=True)
    return DF


def get_score(df_true, df_submission):
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    print('final score:', task1, task2, task3, score)
    return score

# # one-time data pre-processing
# # average original 12-row train data, save to average_train_features.csv
# generate_average('train_features.csv', 'average_train_features.csv')
# # average original 12-row test data, save to average_test_features.csv
# generate_average('test_features.csv', 'average_test_features.csv')


# Local train_test_split test
# read total patients' train data
X_total = pd.read_csv('average_train_features.csv')
y_total = pd.read_csv('train_labels.csv')
X_total = X_total.drop(['pid', 'Time'], axis=1)
y_total = y_total.drop(['pid'], axis=1)

# train_test_split
X_total, Xt_total, y_total, yt_total = train_test_split(X_total, y_total, test_size=0.20)

# drop nan rows in training data, fill nan with mean column value
patients_train = fill_nan(drop_nan(X_total, y_total, 0.65, False), False)
# fill nan in testing feature with mean column value
feature_test = fill_nan(Xt_total, False)

print('training feature size:', patients_train.shape[0])
print('testing feature size:', feature_test.shape[0])

# # predictions
binary_result = binary_predict(patients_train, feature_test)
regress_result = regress_predict(patients_train, feature_test)
final_predict = pd.concat([binary_result, regress_result], axis=1)
print(final_predict.head())
print(yt_total.head())

get_score(yt_total, final_predict)

