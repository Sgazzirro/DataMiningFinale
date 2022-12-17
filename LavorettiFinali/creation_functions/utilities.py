import pandas as pd
import numpy as np
from feature_extraction import extract_features


def create_time_series(labeled=True, mode="Collapsed", num_samples=150):
    ACTIVITY_CODES = ["dws", "jog", "sit", "std", "ups", "wlk"]

    TRIAL_CODES = {
        ACTIVITY_CODES[0]:[1,2,11],
        ACTIVITY_CODES[1]:[9,16],
        ACTIVITY_CODES[2]:[5,13],
        ACTIVITY_CODES[3]:[6,14],
        ACTIVITY_CODES[4]:[3,4,12],
        ACTIVITY_CODES[5]:[7,8,15]
    }

    ACTORS = np.linspace(1, 24, 24).astype(int)

    complete_dataset = pd.DataFrame()
    for subject in ACTORS:
        for activity_code in ACTIVITY_CODES:
            for trial_code in TRIAL_CODES[activity_code]:
                filename = '../LavorettiProvaDiMarco/A_DeviceMotion_data/'+activity_code+'_'+str(trial_code)+'/sub_'+str(int(subject))+'.csv'
                raw_data = pd.read_csv(filename)
                raw_data = raw_data.drop(['Unnamed: 0', "attitude.pitch", "attitude.roll", "attitude.yaw", "gravity.x",
                                          "gravity.y", "gravity.z"], axis=1)
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


def preprocessing(dataframe):
    new_dataset = dataframe.loc[(dataframe["subject"] != 5) | (dataframe["trial"] != 13)]
    only_numeric_dataset = new_dataset.drop(["trial", "subject"], axis=1)
    corr_matrix = only_numeric_dataset.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    preprocessed_dataset = new_dataset.drop(to_drop, axis=1)
    return preprocessed_dataset



