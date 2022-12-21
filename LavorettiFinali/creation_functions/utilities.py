import glob
import pandas as pd
import numpy as np
from pyarrow import null
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score

from feature_extraction import extract_features


def create_time_series(labeled=True, mode="Collapsed", num_samples=150):
    complete_dataset = pd.DataFrame()

    if mode == "phone":
        all_files=glob.glob("honor20readings_complete/*", recursive=True)
        for file in all_files:
            print("Processing file: " + file)
            raw_data = pd.read_csv(file)
            label = raw_data.iloc[0]["class"]
            raw_data = raw_data.drop("class", axis=1)
            # re-ordering of the columns
            raw_data = raw_data[["gravity.x", "gravity.y", "gravity.z", "rotationRate.x", "rotationRate.y",
                                 "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]]
            data_collapsed = extract_features(raw_data, num_samples)
            data_collapsed["class"] = label
            complete_dataset = pd.concat((complete_dataset, data_collapsed), axis=0)
        return complete_dataset

    if mode == "phone_scaled":
        all_files=glob.glob("honor20readings_complete/*", recursive=True)
        for file in all_files:
            print("Processing file: " + file)
            raw_data = scale_readings(pd.read_csv(file))
            raw_data = raw_data[["gravity.x", "gravity.y", "gravity.z", "rotationRate.x", "rotationRate.y",
                                 "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "class"]]
            label = raw_data.iloc[0]["class"]
            raw_data = raw_data.drop("class", axis=1)
            data_collapsed = extract_features(raw_data, num_samples)
            data_collapsed["class"] = label
            complete_dataset = pd.concat((complete_dataset, data_collapsed), axis=0)
        return complete_dataset

    ACTIVITY_CODES = ["dws", "jog", "sit", "std", "ups", "wlk"]

    TRIAL_CODES = {
        ACTIVITY_CODES[0]: [1, 2, 11],
        ACTIVITY_CODES[1]: [9, 16],
        ACTIVITY_CODES[2]: [5, 13],
        ACTIVITY_CODES[3]: [6, 14],
        ACTIVITY_CODES[4]: [3, 4, 12],
        ACTIVITY_CODES[5]: [7, 8, 15]
    }

    ACTORS = np.linspace(1, 24, 24).astype(int)

    for activity_code in ACTIVITY_CODES:
        for trial_code in TRIAL_CODES[activity_code]:
            for subject in ACTORS:
                filename = 'A_DeviceMotion_data/' + activity_code + '_' + str(trial_code) + '/sub_' + str(
                    int(subject)) + '.csv'
                raw_data = pd.read_csv(filename)
                raw_data = raw_data.drop(['Unnamed: 0', "attitude.pitch", "attitude.roll", "attitude.yaw"], axis=1)
                if mode == "raw":
                    data_collapsed = raw_data
                else:
                    data_collapsed = extract_features(raw_data, num_samples)
                if labeled:
                    data_collapsed["class"] = activity_code
                    data_collapsed["subject"] = subject
                    data_collapsed["trial"] = trial_code
                complete_dataset = pd.concat((complete_dataset, data_collapsed), axis=0)

    return complete_dataset


def get_some_filter(complete_dataset, actors, act_labels):
    filtered_dataset = complete_dataset.loc[complete_dataset["subject"].isin(actors)]
    filtered_dataset = filtered_dataset.loc[filtered_dataset["class"].isin(act_labels)]
    return filtered_dataset


def preprocessing(dataframe, to_drop=None):
    dataframe = dataframe.fillna(dataframe.groupby('class').transform('mean'))
    if to_drop is None:
        dataframe = dataframe.loc[(dataframe["subject"] != 5) | (dataframe["trial"] != 13)]
        only_numeric_dataset = dataframe.drop(["trial", "subject"], axis=1)
        only_numeric_dataset = only_numeric_dataset.drop_duplicates()
        corr_matrix = only_numeric_dataset.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    preprocessed_dataset = dataframe.drop(to_drop, axis=1)
    return preprocessed_dataset, to_drop


def custom_cross_validation(dataframe, classifier):
    a = range(1, 25)
    permutation = np.random.permutation(a)
    y_pred = []
    y_true = []
    for i in permutation:
        X_train, X_test, y_train, y_test = get_a_split(dataframe, i)
        y_true.append(y_test)
        classifier.fit(X_train, y_train)
        y_pred.append(classifier.predict(X_test))
    return permutation, y_true, y_pred


def make_report(permutation, true_labels, predicted_labels):
    report = pd.DataFrame()
    for i in range(0, len(permutation)):
        report = report.append({
            "accuracy" : accuracy_score(true_labels[i], predicted_labels[i]),
            "precision" : precision_score(true_labels[i], predicted_labels[i], average="weighted"),
            "recall" : recall_score(true_labels[i], predicted_labels[i], average="weighted"),
            "f1_score" : f1_score(true_labels[i], predicted_labels[i], average="weighted"),
            "without_who" : permutation[i]
        }, ignore_index=True
        )
    return report


def get_a_split(dataframe, who_to_leave_out):
    train_data = dataframe.loc[dataframe["subject"] != who_to_leave_out]
    test_data = dataframe.loc[dataframe["subject"] == who_to_leave_out]
    train_labels = train_data["class"]
    test_labels = test_data["class"]
    train_data = train_data.drop(["class", "subject", "trial"], axis=1)
    test_data = test_data.drop(["class", "subject", "trial"], axis=1)
    return train_data, test_data, train_labels, test_labels


def scale_readings(df, phone = "honor"):
    if phone=="honor":
        df["rotationRate.z"]=df["rotationRate.z"].apply(lambda x: x*2)
        df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]]=df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].apply(lambda x: x/7)
        df[["gravity.x", "gravity.y", "gravity.z"]]=df[["gravity.x", "gravity.y", "gravity.z"]].apply(lambda x: x/10)
        df["gravity.y"]=df["gravity.y"].apply(lambda x:x*-1)
    return df

