import os
import pandas as pd
import numpy as np

from mxnet.gluon.data import Dataset
from sklearn import preprocessing


def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


def get_data_frame(root_dir='./data'):
    data = pd.read_csv(os.path.join(root_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    data.drop(['customerID'], axis=1, inplace=True)

    data['Churn'] = data['Churn'].map(lambda s: 1 if s == 'Yes' else 0)
    data['gender'] = data['gender'].map(lambda s: 1 if s == 'Male' else 0)
    data['Partner'] = data['Partner'].map(lambda s: 1 if s == 'Yes' else 0)
    data['Dependents'] = data['Dependents'].map(lambda s: 1 if s == 'Yes' else 0)
    data['PhoneService'] = data['PhoneService'].map(lambda s: 1 if s == 'Yes' else 0)
    data['PaperlessBilling'] = data['PaperlessBilling'].map(lambda s: 1 if s == 'Yes' else 0)

    data['MultipleLines'].replace('No phone service', 'No', inplace=True)
    data['MultipleLines'] = data['MultipleLines'].map(lambda s: 1 if s == 'Yes' else 0)

    data['Has_InternetService'] = data['InternetService'].map(lambda s: 0 if s == 'No' else 1)
    data['Fiber_optic'] = data['InternetService'].map(lambda s: 1 if s == 'Fiber optic' else 0)
    data['DSL'] = data['InternetService'].map(lambda s: 1 if s == 'DSL' else 0)
    data.drop(['InternetService'], axis=1, inplace=True)

    data['OnlineSecurity'] = data['OnlineSecurity'].map(lambda s: 1 if s == 'Yes' else 0)
    data['OnlineBackup'] = data['OnlineBackup'].map(lambda s: 1 if s == 'Yes' else 0)
    data['DeviceProtection'] = data['DeviceProtection'].map(lambda s: 1 if s == 'Yes' else 0)
    data['TechSupport'] = data['TechSupport'].map(lambda s: 1 if s == 'Yes' else 0)
    data['StreamingTV'] = data['StreamingTV'].map(lambda s: 1 if s == 'Yes' else 0)
    data['StreamingMovies'] = data['StreamingMovies'].map(lambda s: 1 if s == 'Yes' else 0)

    data = pd.get_dummies(data=data, columns=['PaymentMethod'])
    data = pd.get_dummies(data=data, columns=['Contract'])

    data['TotalCharges'] = data['TotalCharges'].replace(np.nan, 0.0)
    data['TotalCharges'] = data['TotalCharges'].map(
        lambda s: float(s.strip()) if s.strip() else float(0.0))

    min_max_scaler = preprocessing.MinMaxScaler()
    data['MonthlyCharges'] = min_max_scaler.fit_transform(
        data['MonthlyCharges'].values.reshape(-1, 1))
    data['TotalCharges'] = min_max_scaler.fit_transform(data['TotalCharges'].values.reshape(-1, 1))

    # rearrange columns, so Churn is the last column
    data = data[[c for c in data if c not in ['Churn']] + ['Churn']]

    return data


class TelcoDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def __getitem__(self, idx):
        return tuple(self._df.iloc[idx].tolist())

    def __len__(self):
        return self._df.shape[0]
