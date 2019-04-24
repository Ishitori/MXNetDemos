import os
import pandas as pd
import numpy as np
from mxnet.gluon.data import Dataset
from sklearn import preprocessing


def get_data_frame(root_dir='./data'):
    multiple_lines_mapping = {
        'Yes': 2,
        'No': 1,
        'No phone service': 0
    }

    internet_service_mapping = {
        'Fiber optic': 2,
        'DSL': 1,
        'No': 0
    }

    online_security_mapping = {
        'Yes': 2,
        'No': 1,
        'No internet service': 0
    }

    online_backup_mapping = {
        'Yes': 2,
        'No': 1,
        'No internet service': 0
    }

    device_protection_mapping = {
        'Yes': 2,
        'No': 1,
        'No internet service': 0
    }

    tech_support_mapping = {
        'Yes': 2,
        'No': 1,
        'No internet service': 0
    }

    streaming_tv_mapping = {
        'Yes': 2,
        'No': 1,
        'No internet service': 0
    }
    streaming_movies_mapping = {
        'Yes': 2,
        'No': 1,
        'No internet service': 0
    }
    contract_mapping = {
        'Two year': 2,
        'One year': 1,
        'Month-to-month': 0
    }
    payment_method_mapping = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 0
    }

    data = pd.read_csv(os.path.join(root_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    data.drop(['customerID'], axis=1, inplace=True)

    data['Churn'] = data['Churn'].map(lambda s: 1 if s == 'Yes' else 0)
    data['gender'] = data['gender'].map(lambda s: 1 if s == 'Male' else 0)
    data['Partner'] = data['Partner'].map(lambda s: 1 if s == 'Yes' else 0)
    data['Dependents'] = data['Dependents'].map(lambda s: 1 if s == 'Yes' else 0)
    data['PhoneService'] = data['PhoneService'].map(lambda s: 1 if s == 'Yes' else 0)
    data['PaperlessBilling'] = data['PaperlessBilling'].map(lambda s: 1 if s == 'Yes' else 0)
    data['MultipleLines'] = data['MultipleLines'].map(lambda s: multiple_lines_mapping[s])
    data['InternetService'] = data['InternetService'].map(lambda s: internet_service_mapping[s])
    data['OnlineSecurity'] = data['OnlineSecurity'].map(lambda s: online_security_mapping[s])
    data['OnlineBackup'] = data['OnlineBackup'].map(lambda s: online_backup_mapping[s])
    data['DeviceProtection'] = data['DeviceProtection'].map(lambda s: device_protection_mapping[s])
    data['TechSupport'] = data['TechSupport'].map(lambda s: tech_support_mapping[s])
    data['StreamingTV'] = data['StreamingTV'].map(lambda s: streaming_tv_mapping[s])
    data['StreamingMovies'] = data['StreamingMovies'].map(lambda s: streaming_movies_mapping[s])
    data['Contract'] = data['Contract'].map(lambda s: contract_mapping[s])
    data['PaymentMethod'] = data['PaymentMethod'].map(lambda s: payment_method_mapping[s])

    data['TotalCharges'] = data['TotalCharges'].replace(np.nan, 0.0)
    data['TotalCharges'] = data['TotalCharges'].map(
        lambda s: float(s.strip()) if s.strip() else float(0.0))

    min_max_scaler = preprocessing.MinMaxScaler()
    data['MonthlyCharges'] = min_max_scaler.fit_transform(
        data['MonthlyCharges'].values.reshape(-1, 1))
    data['TotalCharges'] = min_max_scaler.fit_transform(data['TotalCharges'].values.reshape(-1, 1))

    return data


class TelcoDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def __getitem__(self, idx):
        return tuple(self._df.iloc[idx].tolist())

    def __len__(self):
        return self._df.shape[0]
