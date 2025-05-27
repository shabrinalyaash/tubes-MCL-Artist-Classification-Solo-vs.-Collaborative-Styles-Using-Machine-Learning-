import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")
st.title("Artist Classification: Solo vs. Collaborative Styles Using Machine Learning")

# Header Image
st.image("TUBES MCL TERBARU/headermcl.png", caption="Logo Project", use_column_width=True)

# 1. Data Preprocessing dan Eksplorasi Data
st.header("1. Data Preprocessing dan Eksplorasi Data")
df = pd.read_csv('TUBES MCL TERBARU/artists.csv')

st.subheader("Preview Dataset")
st.dataframe(df.head())

# Cleaning missing values sama persis
df['Daily'] = df['Daily'].fillna(0)
df['As lead'] = df['As lead'].fillna(0)
df['Solo'] = df['Solo'].fillna(0)
df['As feature'] = df['As feature'].fillna(0)

def clean_numeric_column(series):
    return series.astype(str).str.replace(',', '', regex=False).astype(float)

for col in ['Streams', 'Daily', 'As lead', 'Solo', 'As feature']:
    df[col] = clean_numeric_column(df[col])

df['artist_type'] = df.apply(lambda x: 'Solo-oriented' if x['Solo'] >= x['As feature'] else 'Collaborative', axis=1)

st.subheader("Distribusi Label Artist Type")
st.bar_chart(df['artist_type'].value_counts())

# Distribusi fitur numerik
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

# Korelasi antar fitur
st.subheader("Matriks Korelasi")
col_corr = st.columns([1, 2, 1])  # Tengah lebih besar
with col_corr[1]:
    fig, ax = plt.subplots(figsize=(4, 3))  # Ukuran lebih kecil
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax, cbar=False)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

# 2. Data Splitting dan Scaling
st.header("2. Data Splitting dan Scaling")
X = df.drop(columns=["Artist", "artist_type"])
y = df["artist_type"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

col1, col2 = st.columns(2)
col1.metric("Train shape", str(X_train.shape))
col2.metric("Test shape", str(X_test.shape))


# Fungsi buat classification report sebagai DataFrame
def classification_report_df(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(2)
    return df_report[['precision', 'recall', 'f1-score', 'support']]

# Fungsi evaluasi dan tampilkan hasil plus confusion matrix
def show_metrics(name, y_true, y_pred):
    st.markdown(f"### {name}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Akurasi   : {accuracy_score(y_true, y_pred):.4f}")
        st.write(f"F1 Score  : {f1_score(y_true, y_pred, average='weighted'):.4f}")
        st.write(f"Precision : {precision_score(y_true, y_pred, average='weighted'):.4f}")
        st.write(f"Recall    : {recall_score(y_true, y_pred, average='weighted'):.4f}")
        st.dataframe(classification_report_df(y_true, y_pred, label_encoder.classes_))
    with col2:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

# 3. Baseline Model Training & Evaluation
st.header("3. Baseline Model Training & Evaluation")

# Random Forest baseline
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train_encoded)
y_pred_rf = rf_model.predict(X_test)
show_metrics("Random Forest (Baseline)", y_test_encoded, y_pred_rf)

# XGBoost baseline
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train_encoded)
y_pred_xgb = xgb_model.predict(X_test)
show_metrics("XGBoost (Baseline)", y_test_encoded, y_pred_xgb)

# SVM baseline (scaled)
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train_encoded)
y_pred_svm = svm_model.predict(X_test_scaled)
show_metrics("SVM (Baseline)", y_test_encoded, y_pred_svm)

# 4. Hyperparameter Tuning
st.header("4. Hyperparameter Tuning (Grid Search)")

with st.spinner("Tuning Random Forest..."):
    rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train_encoded)
    best_rf_model = rf_grid_search.best_estimator_
    y_pred_rf_tuned = best_rf_model.predict(X_test)
    st.write("Best RF Params:", rf_grid_search.best_params_)

with st.spinner("Tuning XGBoost..."):
    xgb_param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    xgb_grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train_encoded)
    best_xgb_model = xgb_grid_search.best_estimator_
    y_pred_xgb_tuned = best_xgb_model.predict(X_test)
    st.write("Best XGB Params:", xgb_grid_search.best_params_)

with st.spinner("Tuning SVM..."):
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42, probability=True))
    ])
    svm_param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'classifier__kernel': ['rbf', 'linear']
    }
    svm_grid_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid_search.fit(X_train, y_train_encoded)
    best_svm_model = svm_grid_search.best_estimator_
    y_pred_svm_tuned = best_svm_model.predict(X_test)
    st.write("Best SVM Params:", svm_grid_search.best_params_)

# 5. Evaluation After Tuning
st.header("5. Evaluation After Tuning")
show_metrics("Random Forest (Tuned)", y_test_encoded, y_pred_rf_tuned)
show_metrics("XGBoost (Tuned)", y_test_encoded, y_pred_xgb_tuned)
show_metrics("SVM (Tuned)", y_test_encoded, y_pred_svm_tuned)

# 6. Summary Table
st.header("6. Ringkasan Metrik Model")
# Lengkapi DataFrame summary metrik (lanjutan dari kode sebelumnya)
summary = pd.DataFrame({
    "Model": [
        "Random Forest (Baseline)", "Random Forest (Tuned)",
        "XGBoost (Baseline)", "XGBoost (Tuned)",
        "SVM (Baseline)", "SVM (Tuned)"
    ],
    "Accuracy": [
        accuracy_score(y_test_encoded, y_pred_rf), accuracy_score(y_test_encoded, y_pred_rf_tuned),
        accuracy_score(y_test_encoded, y_pred_xgb), accuracy_score(y_test_encoded, y_pred_xgb_tuned),
        accuracy_score(y_test_encoded, y_pred_svm), accuracy_score(y_test_encoded, y_pred_svm_tuned)
    ],
    "F1 Score": [
        f1_score(y_test_encoded, y_pred_rf, average='weighted'), f1_score(y_test_encoded, y_pred_rf_tuned, average='weighted'),
        f1_score(y_test_encoded, y_pred_xgb, average='weighted'), f1_score(y_test_encoded, y_pred_xgb_tuned, average='weighted'),
        f1_score(y_test_encoded, y_pred_svm, average='weighted'), f1_score(y_test_encoded, y_pred_svm_tuned, average='weighted')
    ],
    "Precision": [
        precision_score(y_test_encoded, y_pred_rf, average='weighted'), precision_score(y_test_encoded, y_pred_rf_tuned, average='weighted'),
        precision_score(y_test_encoded, y_pred_xgb, average='weighted'), precision_score(y_test_encoded, y_pred_xgb_tuned, average='weighted'),
        precision_score(y_test_encoded, y_pred_svm, average='weighted'), precision_score(y_test_encoded, y_pred_svm_tuned, average='weighted')
    ],
    "Recall": [
        recall_score(y_test_encoded, y_pred_rf, average='weighted'), recall_score(y_test_encoded, y_pred_rf_tuned, average='weighted'),
        recall_score(y_test_encoded, y_pred_xgb, average='weighted'), recall_score(y_test_encoded, y_pred_xgb_tuned, average='weighted'),
        recall_score(y_test_encoded, y_pred_svm, average='weighted'), recall_score(y_test_encoded, y_pred_svm_tuned, average='weighted')
    ]
})
st.dataframe(summary)

# Tambahkan bagian Kesimpulan / Analisis
st.header("7. Kesimpulan dan Insight")

st.markdown("""
- Semua model yang diuji menunjukkan performa sangat baik dengan akurasi dan F1 Score di atas 97%.
- Model SVM dengan hyperparameter tuning memberikan performa terbaik, dengan akurasi dan F1 Score sekitar 99.18%.
- Hyperparameter tuning memberikan peningkatan kecil namun konsisten pada ketiga model.
- Fitur yang digunakan ('Streams', 'Daily', 'As lead', 'Solo', 'As feature') terbukti sangat informatif untuk membedakan artis solo dan kolaboratif.
- Penting untuk melakukan validasi lebih lanjut pada data eksternal untuk memastikan generalisasi model.
""")
