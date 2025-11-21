import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_export_models():
    X_train = pd.read_csv('./data/X_train.csv')
    X_test = pd.read_csv('./data/X_test.csv')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')

    lgbm = LGBMClassifier(class_weight='balanced', n_estimators=250, random_state=42)
    lgbm.fit(X_train, y_train)
    joblib.dump(lgbm, './deployment/model_lgbm.pkl')

    xgb = XGBClassifier(n_estimators=250, random_state=42)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, './deployment/model_xgb.pkl')

    print('LightGBM Classification Report:')
    print(classification_report(y_test, lgbm.predict(X_test)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, lgbm.predict(X_test)))

    print('XGBoost Classification Report:')
    print(classification_report(y_test, xgb.predict(X_test)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, xgb.predict(X_test)))

if __name__ == "__main__":
    train_and_export_models()
