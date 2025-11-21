import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

# Load preprocessed data
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = np.load('../data/y_train.npy')
y_test = np.load('../data/y_test.npy')

# Train a LightGBM Classifier (tabular anomaly detection)
model = LGBMClassifier(class_weight='balanced', n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, preds))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, preds))

# Save classic ML model
joblib.dump(model, '../data/model_lgbm.pkl')

# Feature importances simple report
importances = model.feature_importances_
feat_report = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
feat_report = feat_report.sort_values('importance', ascending=False)
feat_report.to_csv('../data/feature_importances.csv', index=False)

print('✅ LightGBM model trained and saved as model_lgbm.pkl. Feature importances in feature_importances.csv.')
