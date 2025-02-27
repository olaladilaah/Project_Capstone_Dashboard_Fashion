import numpy as np
import pandas as pd
from datetime import datetime

# Untuk Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from yellowbrick.cluster import KElbowVisualizer

# Untuk Machine Learning and Model Evaluation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, classification_report,
                             accuracy_score, confusion_matrix, mean_absolute_percentage_error,  silhouette_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, TruncatedSVD

# Untuk Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Untuk Time Series Analysis
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Untuk Recommendation Systems
from surprise import Dataset, Reader, SVD, accuracy

# Untuk Utilities
import joblib
import opendatasets as od
import streamlit as st
from io import BytesIO

@st.cache_data
def load_data():
    df = pd.read_csv("Fashion_Retail_Sales.csv").dropna()
    df["Date Purchase"] = pd.to_datetime(df["Date Purchase"])
    return df

df = load_data()

# Sidebar
st.sidebar.title("Dashboard UMKM Fashion")
st.sidebar.subheader("Filter Data")
df.sort_values(by="Date Purchase", inplace=True)
df.reset_index(inplace=True)
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])

min_date = df["Date Purchase"].min().date()
max_date = df["Date Purchase"].max().date()

with st.sidebar :
    start_date, end_date = st.date_input(
        label='Rentang Transaksi',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = df[(df["Date Purchase"] >= pd.to_datetime(start_date)) &
                 (df["Date Purchase"] <= pd.to_datetime(end_date))]

# Header
st.title("📊 Dashboard Penjualan Fashion")

# Ringkasan Data
st.subheader("Ringkasan Data")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transaksi", f"{main_df.shape[0]:,}")
col2.metric("Rata-rata Pembelian ($)", f"{main_df['Purchase Amount (USD)'].mean():,.2f}")
col3.metric("Total Produk Unik", f"{main_df['Item Purchased'].nunique()}")

# Grafik Tren Penjualan
st.subheader("📈 Tren Penjualan")
sales_trend = main_df.groupby("Date Purchase")["Purchase Amount (USD)"].sum().reset_index()
fig_trend = px.line(sales_trend, x="Date Purchase", y="Purchase Amount (USD)", title="Tren Penjualan dari Waktu ke Waktu")
st.plotly_chart(fig_trend)

# Produk Terlaris
value_counts = df["Item Purchased"].value_counts()

# Membuat grafik batang horizontal
fig = go.Figure()

fig.add_trace(go.Bar(
    x=value_counts.values,
    y=value_counts.index,
    orientation='h',
    marker=dict(
        color=value_counts.values,
        colorscale='blues',  # Menggunakan skema warna biru
        showscale=True
    )
))

# Update layout
fig.update_layout(
    title={
        'text': 'Jumlah Pembelian per Produk',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Jumlah Pembelian',
    yaxis_title='Produk',
    plot_bgcolor='white',
    bargap=0.2,
    showlegend=False
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=False)

# Tampilkan grafik di Streamlit
st.subheader("🔥 Produk Terlaris")
st.plotly_chart(fig)

# Distribusi Metode Pembayaran
st.subheader("💳 Distribusi Metode Pembayaran")
payment_counts = main_df["Payment Method"].value_counts().reset_index()
payment_counts.columns = ["Metode Pembayaran", "Jumlah"]  

fig_payment = px.pie(
    payment_counts, names="Metode Pembayaran", values="Jumlah", 
    title="Pembagian Metode Pembayaran"
)
st.plotly_chart(fig_payment)

# Analisis Rating
fig = px.histogram(df, x='Review Rating', nbins=5, title='Distribusi Review Rating')

# Update layout
fig.update_layout(
    xaxis_title='Rating',
    yaxis_title='Frekuensi',
    template='plotly_dark'
)

# Update warna
fig.update_traces(marker=dict(color='turquoise'))

# Tampilkan grafik di Streamlit
st.subheader("⭐ Distribusi Review Rating")
st.plotly_chart(fig)

st.subheader("🧑‍🤝‍🧑 Clustering Pelanggan Berdasarkan Perilaku Pembelian")
cluster_method = st.selectbox("Pilih Metode Clustering", ["KMeans", "Agglomerative Clustering"])

#CLUSTERING
# Preprocessing Data
data_encoder = df.copy().dropna()
LE = LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    data_encoder[col] = LE.fit_transform(df[col])

# Data untuk clustering
data_clustering = data_encoder[["Item Purchased", "Purchase Amount (USD)", "Review Rating", "Payment Method"]]

# Standarisasi Data
scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(data_clustering), columns=data_clustering.columns)

if cluster_method == "KMeans":
    #PCA
    pca = PCA()
    pca.fit(scaled_features)
    cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)

    # Plot PCA Explained Variance
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, label="Individual Explained Ratio")
    ax.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where="mid", label="Cumulative Explained Ratio")
    ax.set_xlabel("N Components")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend(loc="best")

    # Menampilkan di Streamlit
    st.subheader("🔎 PCA Explained Variance Ratio")
    st.pyplot(fig)

    cols = ["PCA1", "PCA2", "PCA3"]
    pca = PCA(n_components=len(cols))
    pca.fit(scaled_features)
    PCA_df = pd.DataFrame(pca.transform(scaled_features), columns=(cols))
    model = KMeans()
    # Buat visualisasi elbow method
    fig, ax = plt.subplots()
    visualizer = KElbowVisualizer(model, k=11, metric='silhouette', ax=ax)
    visualizer.fit(PCA_df)

    # Simpan plot ke dalam buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Tampilkan di Streamlit
    st.subheader("K-Means Elbow Method with Silhouette Score")
    st.image(buf, caption="Elbow Method Visualization", use_container_width=True)

    def visualize_silhouette_layer(data):
        clusters_range = range(2, 10)
        results = []

        for i in clusters_range:
            km = KMeans(n_clusters=i, random_state=42)
            cluster_labels = km.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            results.append([i, silhouette_avg])

        result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
        pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

        # Buat plot
        fig, ax = plt.subplots()
        sns.heatmap(pivot_km, annot=True, linewidths=1, fmt='.3f', cmap='RdYlGn', ax=ax)
        plt.tight_layout()

        return fig  # Mengembalikan figure untuk Streamlit

    # Streamlit App
    st.subheader("Silhouette Score Heatmap")

    fig = visualize_silhouette_layer(PCA_df)

    # Simpan plot ke dalam buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Tampilkan di Streamlit
    st.image(buf, caption="Silhouette Score Heatmap", use_container_width=True)

elif cluster_method == "Agglomerative Clustering": 
    # PCA untuk mengurangi dimensi ke 3 komponen
    pca = PCA(n_components=3)
    PCA_df = pd.DataFrame(pca.fit_transform(scaled_features), columns=["PCA1", "PCA2", "PCA3"])

    # Agglomerative Clustering
    AC = AgglomerativeClustering(n_clusters=5)
    PCA_df["Clusters"] = AC.fit_predict(PCA_df)

    # Streamlit App
    st.title("Clustering Analysis using PCA & Agglomerative Clustering")

    # --- 3D Scatter Plot ---
    st.subheader("3D Scatter Plot of Clusters")
    fig_3d = plt.figure(figsize=(10, 6))
    ax = fig_3d.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        PCA_df["PCA1"], PCA_df["PCA2"], PCA_df["PCA3"],
        c=PCA_df["Clusters"], cmap='cool_r', s=40, marker='o'
    )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.set_title("The Plot Of The Clusters")
    st.pyplot(fig_3d)

    # --- Pairplot ---
    st.subheader("Pairplot of PCA Components")
    sns.set(style="whitegrid")
    pairplot_fig = sns.pairplot(PCA_df, vars=['PCA1', 'PCA2', 'PCA3'], hue='Clusters', palette="cool_r")
    st.pyplot(pairplot_fig.fig)

    # --- Countplot (Distribusi Cluster) ---
    st.subheader("Distribution of the Clusters")
    fig_count, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=PCA_df["Clusters"], palette="cool_r", ax=ax)
    ax.set_title("Cluster Distribution")
    st.pyplot(fig_count)
# 2. Classification from Cluster: Predicting Customer Segments with RFM (Recency, Frequency, Monetary)
st.subheader("📉 Klasifikasi Segmen Pelanggan Berdasarkan RFM")

seg_map = {
    r'[1-2][1-2]': 'Inactive Customer',
    r'[1-2][3-4]': 'Declining Customer',
    r'[1-2]5': 'High Value Low Engagement Customer',
    r'3[1-2]': 'Transition to Inactive Customer',
    r'33': 'Needed Attention Customer',
    r'[3-4][4-5]': 'Loyal Customer',
    r'41': 'Promising Customer',
    r'51': 'New Customer',
    r'[4-5][2-3]': 'Potential Loyal Customer',
    r'5[4-5]': 'Top Spending Customer'
}

# Hitung RFM
current_date = pd.to_datetime(df['Date Purchase']).max()
rfm = df.groupby('Customer Reference ID').agg({
    'Date Purchase': lambda x: ((current_date - pd.to_datetime(x.max())).days) / 7,
    'Customer Reference ID': 'count',
    'Purchase Amount (USD)': 'sum'
})

# Normalisasi Monetary (pembelian rata-rata mingguan)
rfm['Purchase Amount (USD)'] = (rfm['Purchase Amount (USD)'] / 7).astype(int)

# Rename kolom ke RFM
rfm.rename(columns={
    'Date Purchase': 'Recency',
    'Customer Reference ID': 'Frequency',
    'Purchase Amount (USD)': 'Monetary'
}, inplace=True)

# Fungsi untuk mendapatkan skor RFM
def get_rfm_scores(data):
    data["R"] = pd.qcut(data["Recency"], 5, labels=[5, 4, 3, 2, 1])
    data["F"] = pd.qcut(data["Frequency"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    data["M"] = pd.qcut(data["Monetary"], 5, labels=[1, 2, 3, 4, 5])
    data["RFM_SCORE"] = data["R"].astype(str) + data["F"].astype(str)
    return data

# Apply RFM scoring
rfm = get_rfm_scores(rfm)
rfm.reset_index(inplace=True)

# Mapping ke segmen
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# Buat DataFrame untuk visualisasi
df_cluster = rfm.copy()
x = df_cluster.segment.value_counts().reset_index()
x.columns = ['segment', 'count']

# Buat Treemap
fig = px.treemap(
    x,
    path=['segment'],
    values='count',
    color='segment',
    title='Distribution of the RFM Segments',
    color_continuous_scale='blues'
)

fig.update_layout(title_x=0.5, title_font=dict(size=30))
fig.update_traces(textinfo="label+value+percent root")

# Streamlit App
st.title("RFM Segmentation Analysis")
st.plotly_chart(fig)   

X = rfm.drop(['segment'], axis=1)
y = rfm['segment']
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

classifier = st.selectbox("Pilih Model Klasifikasi", ["SVM", "Logistic Regression", "Naive Bayes", "Decision Tree"])

if classifier == "SVM":
    # Train SVM Model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm_svc = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

    # --- Streamlit UI ---
    st.title("Customer Segmentation using Support Vector Machine (SVC)")

    # --- Accuracy ---
    st.subheader("Model Accuracy")
    st.write(f"**Accuracy: {accuracy}**")

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix for SVC')
    st.pyplot(fig)

    # --- Classification Report ---
    st.subheader("Classification Report")
    st.text(classification_rep)

elif classifier == "Logistic Regression":
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)
    y_pred_logreg = logreg_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred_logreg)
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    classification_rep = classification_report(y_test, y_pred_logreg, target_names=le.classes_)

    # --- Streamlit UI ---
    st.title("Customer Segmentation using Logistic Regression")

    # --- Accuracy ---
    st.subheader("Model Accuracy")
    st.write(f"**Accuracy: {accuracy:.4f}**")

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix for Logistic Regression')
    st.pyplot(fig)

    # --- Classification Report ---
    st.subheader("Classification Report")
    st.text(classification_rep)

elif classifier == "Naive Bayes":
    # Train Naïve Bayes Model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    # Metrics
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    classification_rep_nb = classification_report(y_test, y_pred_nb, target_names=le.classes_)

    # --- Streamlit UI ---
    st.title("Customer Segmentation using Naïve Bayes (GaussianNB)")

    # --- Accuracy ---
    st.subheader("Model Accuracy")
    st.write(f"**Accuracy: {accuracy_nb:.4f}**")

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix for Naïve Bayes')
    st.pyplot(fig)

    # --- Classification Report ---
    st.subheader("Classification Report")
    st.text(classification_rep_nb)
    
    
elif classifier == "Decision Tree":
    # Train Decision Tree Model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # Metrics
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    classification_rep_dt = classification_report(y_test, y_pred_dt, target_names=le.classes_)

    # --- Streamlit UI ---
    st.title("Customer Segmentation using Decision Tree Classifier")

    # --- Accuracy ---
    st.subheader("Model Accuracy")
    st.write(f"**Accuracy: {accuracy_dt:.4f}**")

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix for Decision Tree')
    st.pyplot(fig)

    # --- Classification Report ---
    st.subheader("Classification Report")
    st.text(classification_rep_dt)

# 3. Time Series: Forecasting Future Sales with SARIMAX
st.subheader("🔮 Prediksi Tren Penjualan Masa Depan")

# --- Preprocessing Data ---
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
data_encoder = df.copy()
data_encoder = data_encoder.set_index('Date Purchase')

df_resampled = data_encoder.resample('D').agg({
    'Purchase Amount (USD)': 'sum',
    'Item Purchased': 'count',
    'Review Rating': 'mean',
    'Payment Method': lambda x: x.mode()[0] if not x.empty else None
})

# Train-Test Split
train_data, test_data = train_test_split(df_resampled, test_size=0.2, shuffle=False)

# --- Streamlit UI ---
st.subheader("Retail Sales Forecasting using SARIMAX")

# --- Plot Historical Data ---
st.subheader("Purchase Amount Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_resampled['Purchase Amount (USD)'], label='Purchase Amount')
ax.set_xlabel('Date')
ax.set_ylabel('USD')
ax.set_title('Purchase Amount vs. Date')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

model = SARIMAX(train_data['Purchase Amount (USD)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
results = model.fit()

# --- Model Diagnostics ---
st.subheader("SARIMAX Model Diagnostics")
fig = results.plot_diagnostics(figsize=(15, 12))
st.pyplot(fig)

# --- Forecasting ---
st.subheader("Forecasted vs. Actual Values")
predictions = results.get_forecast(steps=len(test_data))
predicted_mean = predictions.predicted_mean

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data['Purchase Amount (USD)'], predicted_mean))
st.write(f"**RMSE: {rmse:.4f}**")

# Plot predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_data['Purchase Amount (USD)'], label='Train Data')
ax.plot(test_data['Purchase Amount (USD)'], label='Test Data')
ax.plot(predicted_mean, label='Predicted', linestyle='dashed', color='red')
ax.legend()
ax.set_title('SARIMAX Predictions')
st.pyplot(fig)

# --- Forecast Function ---
def forecast(data, periods=2):
    train_data = data[-periods * 30:]

    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit()

    steps = periods * 30
    forecast_results = results.get_forecast(steps=steps)
    predicted_mean = forecast_results.predicted_mean
    confidence_intervals = forecast_results.conf_int()

    index_of_fc = pd.date_range(train_data.index[-1] + pd.DateOffset(days=1), periods=steps, freq='D')

    lower_series = pd.Series(confidence_intervals.iloc[:, 0], index=index_of_fc)
    upper_series = pd.Series(confidence_intervals.iloc[:, 1], index=index_of_fc)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data, color='#1f76b4', label='Actual Data')
    ax.plot(predicted_mean, color='red', label='Forecast')
    ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.15)
    ax.set_title("SARIMAX - Forecast of Purchase Amount (USD)")
    ax.legend()
    st.pyplot(fig)

# --- Forecast Future Data ---
st.subheader("Future Forecast")
forecast(df_resampled['Purchase Amount (USD)'], periods=1)

# 4. Collaborative Filtering: Product Recommendation with SVD
# Create matrix for SVD
pivot_table = df.pivot_table(index='Customer Reference ID', columns='Item Purchased', values='Review Rating', aggfunc='mean')

# Apply SVD
svd = TruncatedSVD(n_components=5, random_state=42)
svd_matrix = svd.fit_transform(pivot_table.fillna(0))

st.write("Rekomendasi produk untuk pelanggan tertentu berdasarkan SVD")
customer_id = st.selectbox("Pilih ID Pelanggan", df['Customer Reference ID'].unique())

# Get recommendation for customer
customer_index = pivot_table.index.get_loc(customer_id)
customer_ratings = svd_matrix[customer_index]

# Get top 5 recommended products
recommended_products = pivot_table.columns[customer_ratings.argsort()[-5:][::-1]]
st.write(f"Produk yang direkomendasikan untuk pelanggan {customer_id}: {', '.join(recommended_products)}")

# Footer
st.markdown("Dashboard ini dibuat untuk membantu UMKM dalam menganalisis tren penjualan dan meningkatkan strategi bisnis mereka.")
