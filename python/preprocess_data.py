import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load data
df = pd.read_csv('../data/tourism_anomaly_dataset.csv')

# Feature engineering: extract hour from timestamp
if 'timestamp' in df.columns:
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
else:
    df['hour'] = 0  # fallback

# Encode categorical
df['user_id'] = LabelEncoder().fit_transform(df['user_id'])
df['planned_route_id'] = df['planned_route_id'].astype(str)
df['planned_route_id'] = LabelEncoder().fit_transform(df['planned_route_id'])

# Encode bool/flag
for col in ['is_checkin', 'is_emergency_action']:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# Encode target
y = LabelEncoder().fit_transform(df['anomaly_type'])
X = df.drop(['timestamp', 'anomaly_type'], axis=1)

# Scale numeric columns
num_cols = ['latitude', 'longitude', 'speed_mps', 'distance_from_route_m', 'battery_percent',
            'signal_strength', 'inactivity_minutes', 'hour']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Save splits for re-use in notebooks/scripts
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)

print("✅ Preprocessing complete! - X_train, X_test, y_train, y_test ready for ML/DL pipelines.")
