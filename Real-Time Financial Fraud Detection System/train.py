import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

# -------- LOAD DATA --------
df = pd.read_csv("data/fraudTrain.csv")

# -------- TARGET --------
target_col = 'is_fraud'  # adjust if needed

# -------- FEATURE ENGINEERING --------
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day

df['amt_log'] = np.log1p(df['amt'])

df = df.drop(columns=['trans_date_trans_time'])

# -------- ENCODING --------
encoders = {}

for col in df.select_dtypes(include=['object']).columns:
    if col != target_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# -------- SPLIT --------
X = df.drop(target_col, axis=1)
y = df[target_col]

# -------- CLASS WEIGHT --------
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
scale_pos_weight = weights[1]

# -------- TRAIN TEST --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- MODEL --------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -------- EVALUATION --------
probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > 0.3).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, probs))

# -------- SAVE --------
pickle.dump(model, open("model/fraud_model.pkl", "wb"))
pickle.dump(encoders, open("model/encoders.pkl", "wb"))
pickle.dump(list(X.columns), open("model/columns.pkl", "wb"))

print(" Training Complete")
