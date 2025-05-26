import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

st.set_page_config(layout="wide")
st.title("Artist Classification: Solo vs. Collaborative Styles Using Machine Learning")

# Tambahkan gambar di sini
st.image("headermcl.png", caption="Logo Project", use_column_width=True)
# ...existing code...

# 1. Load Data
st.header("1. Data Preprocessing & EDA")
df = pd.read_csv('TUBES MCL TERBARU/artists.csv')
st.subheader("Preview Data")
st.dataframe(df.head())

# Cleaning
df['Daily'] = df['Daily'].fillna(0)
df['As lead'] = df['As lead'].fillna(0)
df['Solo'] = df['Solo'].fillna(0)
df['As feature'] = df['As feature'].fillna(0)
def clean_numeric_column(series):
    return series.astype(str).str.replace(',', '', regex=False).astype(float)
for col in ['Streams', 'Daily', 'As lead', 'Solo', 'As feature']:
    df[col] = clean_numeric_column(df[col])
df['artist_type'] = df.apply(lambda x: 'Solo-oriented' if x['Solo'] >= x['As feature'] else 'Collaborative', axis=1)

# 2. Data Splitting
st.header("2. Data Splitting & Scaling")
X = df.drop(columns=["Artist", "artist_type"])
y = df["artist_type"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

col_ds1, col_ds2 = st.columns(2)
with col_ds1:
    st.metric("Train shape", f"{X_train.shape}")
with col_ds2:
    st.metric("Test shape", f"{X_test.shape}")

# EDA
st.subheader("Distribusi Label")
st.bar_chart(df['artist_type'].value_counts())

st.subheader("Distribusi Fitur Numerik")
features = ['Streams', 'Daily', 'As lead', 'Solo', 'As feature']
fig, axes = plt.subplots(1, len(features), figsize=(15, 3))
for i, col in enumerate(features):
    axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].tick_params(axis='y', labelsize=8)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.subheader("Matriks Korelasi")
col_corr = st.columns([1, 2, 1])  # Tengah lebih besar
with col_corr[1]:
    fig, ax = plt.subplots(figsize=(4, 3))  # Ukuran lebih kecil
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax, cbar=False)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

# Fungsi untuk membuat classification report dalam bentuk DataFrame
def classification_report_df(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(2)
    # Hanya tampilkan kolom penting
    return df_report[['precision', 'recall', 'f1-score', 'support']]

# 3. Baseline Models
st.header("3. Baseline Model Training & Evaluation")
def show_metrics(name, y_true, y_pred):
    st.markdown(f"**{name}**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Accuracy:", accuracy_score(y_true, y_pred))
        st.write("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
        st.write("Precision:", precision_score(y_true, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_true, y_pred, average='weighted'))
        df_report = classification_report_df(y_true, y_pred, label_encoder.classes_)
        st.dataframe(df_report)
    with col2:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
show_metrics("Random Forest (Baseline)", y_test, y_pred_rf)

# XGBoost
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
show_metrics("XGBoost (Baseline)", y_test, y_pred_xgb)

# SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
show_metrics("SVM (Baseline)", y_test, y_pred_svm)

# 4. Hyperparameter Tuning
st.header("4. Hyperparameter Tuning (Grid Search)")
with st.spinner("Tuning Random Forest..."):
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf_tuned = best_rf.predict(X_test)
    st.write("Best RF Params:", rf_grid.best_params_)

with st.spinner("Tuning XGBoost..."):
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), xgb_param_grid, cv=3, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    y_pred_xgb_tuned = best_xgb.predict(X_test)
    st.write("Best XGB Params:", xgb_grid.best_params_)

with st.spinner("Tuning SVM..."):
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42))
    ])
    svm_param_grid = {
        'classifier__C': [1, 10],
        'classifier__gamma': ['scale', 0.1],
        'classifier__kernel': ['rbf']
    }
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=3, n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    y_pred_svm_tuned = best_svm.predict(X_test)
    st.write("Best SVM Params:", svm_grid.best_params_)

# 5. Evaluation After Tuning
st.header("5. Evaluation After Tuning")
show_metrics("Random Forest (Tuned)", y_test, y_pred_rf_tuned)
show_metrics("XGBoost (Tuned)", y_test, y_pred_xgb_tuned)
show_metrics("SVM (Tuned)", y_test, y_pred_svm_tuned)

# 6. Summary Table
st.header("6. Ringkasan Metrik Model")
summary = pd.DataFrame({
    "Model": [
        "Random Forest (Baseline)", "Random Forest (Tuned)",
        "XGBoost (Baseline)", "XGBoost (Tuned)",
        "SVM (Baseline)", "SVM (Tuned)"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_rf_tuned),
        accuracy_score(y_test, y_pred_xgb), accuracy_score(y_test, y_pred_xgb_tuned),
        accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_svm_tuned)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_rf, average='weighted'), f1_score(y_test, y_pred_rf_tuned, average='weighted'),
        f1_score(y_test, y_pred_xgb, average='weighted'), f1_score(y_test, y_pred_xgb_tuned, average='weighted'),
        f1_score(y_test, y_pred_svm, average='weighted'), f1_score(y_test, y_pred_svm_tuned, average='weighted')
    ],
    "Precision": [
        precision_score(y_test, y_pred_rf, average='weighted'), precision_score(y_test, y_pred_rf_tuned, average='weighted'),
        precision_score(y_test, y_pred_xgb, average='weighted'), precision_score(y_test, y_pred_xgb_tuned, average='weighted'),
        precision_score(y_test, y_pred_svm, average='weighted'), precision_score(y_test, y_pred_svm_tuned, average='weighted')
    ],
    "Recall": [
        recall_score(y_test, y_pred_rf, average='weighted'), recall_score(y_test, y_pred_rf_tuned, average='weighted'),
        recall_score(y_test, y_pred_xgb, average='weighted'), recall_score(y_test, y_pred_xgb_tuned, average='weighted'),
        recall_score(y_test, y_pred_svm, average='weighted'), recall_score(y_test, y_pred_svm_tuned, average='weighted')
    ]
})
st.dataframe(summary)

# 7. Analisis & Insight
st.header("7. Analisis & Insight")
st.markdown("""
- Semua model menunjukkan akurasi dan F1 Score sangat tinggi.
- SVM Tuned umumnya memberikan performa terbaik.
- Tuning hyperparameter tidak selalu meningkatkan performa secara signifikan jika baseline sudah optimal.
- Penting untuk validasi lebih lanjut pada data baru.
""")