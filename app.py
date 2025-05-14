from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename
import plotly.express as px
from plotly.offline import plot
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from hmmlearn.hmm import GaussianHMM




app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
EXPORT_FOLDER = 'exports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_uploaded_file(filepath):
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        data = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        data = pd.read_json(filepath)
    else:
        raise ValueError("Unsupported file type")
    return data

def plot_original_data(data):
    fig = px.line(data, x=data.index, y='Value', title='Original Data')
    return plot(fig, output_type='div')

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the data to get column names using the new function
        data = read_uploaded_file(filepath)
        columns = data.columns.tolist()
        return render_template('select_columns.html', columns=columns, filename=filename)
    else:
        return redirect(request.url)

@app.route('/process', methods=['POST'])
@limiter.limit("5 per minute")
def process_file():
    time_column = request.form.get('time_column')
    value_column = request.form.get('value_column')
    filename = request.form.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read the original data and get initial counts
    original_data = read_uploaded_file(filepath)
    original_count = len(original_data)
    print(f"Original data count: {original_count}")  # Debug print

    # Process the data
    data = original_data[[time_column, value_column]].copy()  # Use copy to avoid SettingWithCopyWarning
    count_before_index = len(data)
    print(f"Count before setting index: {count_before_index}")  # Debug print
    
    data.set_index(time_column, inplace=True)
    data.columns = ['Value']
    
    # Count before and after filling missing values
    count_before_fill = data['Value'].count()
    print(f"Count before filling missing values: {count_before_fill}")  # Debug print
    
    data.fillna(method='ffill', inplace=True)
    final_count = len(data)
    print(f"Final count after processing: {final_count}")  # Debug print

    # Use the final count for calculations
    count = final_count

    def plot_anomalies(data, anomalies, method_name):
        """Function to create plots with anomalies highlighted"""
        if isinstance(anomalies, np.ndarray):
            anomalies = pd.DataFrame(anomalies, columns=['Value'])
        fig = px.line(data, x=data.index, y='Value', title=f'Anomalies Detected using {method_name}')
        if not anomalies.empty:
            fig.add_scatter(x=anomalies.index, y=anomalies['Value'], mode='markers', marker=dict(color="red", size=5), name='Anomaly')
        return plot(fig, output_type='div')

    # def train_HMM(series, n_components=4, n_iter=1000):
    #     model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
    #     model.fit(series.reshape(-1, 1))
    #     return model
    
    # def detect_anomalies_HMM(model, values, indices, threshold=-10):
    #     log_likelihood = np.array([model.score(np.array([val]).reshape(-1, 1)) for val in values])
    #     anomaly_indices = np.where(log_likelihood < threshold)[0]
    #     anomalies = pd.DataFrame({'Value': values[anomaly_indices]}, index=indices[anomaly_indices])
    #     return anomalies


    # def train_holt_winters(series, seasonal_periods, trend='add', seasonal='add'):
    #     model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    #     hw_model = model.fit()
    #     return hw_model
    
    # def forecast_holt_winters(hw_model, steps=10):
    #     return hw_model.forecast(steps=steps)



    # K-means Clustering for Anomaly Detection
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[['Value']].values.reshape(-1, 1))
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data_normalized)
    distances = np.linalg.norm(data_normalized - kmeans.cluster_centers_[clusters], axis=1)
    threshold = np.percentile(distances, 95)
    data['anomaly_kmeans'] = (distances > threshold).astype(int)
    anomalies_kmeans = data[data['anomaly_kmeans'] == 1]
    plot_kmeans = plot_anomalies(data, anomalies_kmeans, 'K-means Clustering')

    # Feature Engineering for Isolation Forest and One-Class SVM
    data['rolling_mean'] = data['Value'].rolling(window=5).mean()
    data['rolling_std'] = data['Value'].rolling(window=5).std()
    data['z_score'] = (data['Value'] - data['rolling_mean']) / data['rolling_std']
    data = data.dropna()

    # Isolation Forest
    model_if = IsolationForest(contamination=0.01)
    data['anomaly_if'] = model_if.fit_predict(data[['Value', 'rolling_mean', 'rolling_std', 'z_score']])
    anomalies_if = data[data['anomaly_if'] == -1]
    plot_if = plot_anomalies(data, anomalies_if, 'Isolation Forest')

    # One-Class SVM
    model_svm = OneClassSVM(nu=0.03)
    data['anomaly_svm'] = model_svm.fit_predict(data[['Value']])
    anomalies_svm = data[data['anomaly_svm'] == -1]
    plot_svm = plot_anomalies(data, anomalies_svm, 'One-Class SVM')

    # Local Outlier Factor (LOF)
    # Simpler feature engineering for LOF
    data['rolling_mean'] = data['Value'].rolling(window=3).mean()  # Shorter window for more local sensitivity
    data['rolling_std'] = data['Value'].rolling(window=3).std()
    data['diff'] = data['Value'].diff()
    data['diff_abs'] = data['diff'].abs()
    
    # Fill NaN values created by rolling windows
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    # Prepare features for LOF - using simpler feature set
    lof_features = ['Value', 'rolling_mean', 'diff_abs']
    
    # Scale the features
    scaler = StandardScaler()
    lof_data = scaler.fit_transform(data[lof_features])
    
    # Optimize LOF parameters for better sensitivity
    model_lof = LocalOutlierFactor(
        n_neighbors=15,  # Reduced for more local sensitivity
        contamination=0.01,  # Reduced to be more selective
        metric='manhattan',  # Changed to manhattan distance for better handling of differences
        algorithm='auto',
        leaf_size=20,
        p=1,  # Changed to 1 for manhattan distance
        novelty=False
    )
    
    # Fit and predict
    data['anomaly_lof'] = model_lof.fit_predict(lof_data)
    anomalies_lof = data[data['anomaly_lof'] == -1]
    plot_lof = plot_anomalies(data, anomalies_lof, 'Local Outlier Factor')


    # # Normalize the data and apply DBSCAN
    # scaler = StandardScaler()
    # data_normalized = scaler.fit_transform(data[['Value']].values.reshape(-1, 1))
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # data['anomaly_dbscan'] = dbscan.fit_predict(data_normalized)
    # anomalies_dbscan = data[data['anomaly_dbscan'] == -1]
    # plot_dbscan = plot_anomalies(data, anomalies_dbscan, 'DBSCAN')


    # # Train HMM
    # hmm_model = train_HMM(data['Value'].values)

    # # Detect anomalies using HMM
    # anomalies_hmm = detect_anomalies_HMM(hmm_model, data['Value'].values, data['Value'].index)

    # plot_hmm = plot_anomalies(data, anomalies_hmm, 'HMM')

    # The data is good due to its weights toward newer information 


    #hw_model = train_holt_winters(data['Value'], seasonal_periods=12)  Blocked off due to error out of bound can't predict based 

    #hw_forecast = pd.DataFrame(hw_forecast, columns=['Value'])

    #plot_hw = plot_anomalies(data, hw_forecast, 'HW')


    # Additional Statistical Features and Descriptive Statistics
    data['diff'] = data['Value'].diff()
    data['cumsum'] = data['Value'].cumsum()
    data['cumprod'] = (1 + data['Value']).cumprod()
    mean = round(data['Value'].mean(), 2)
    median = round(data['Value'].median(), 2)
    std = round(data['Value'].std(), 2)
    min_val = round(data['Value'].min(), 2)
    max_val = round(data['Value'].max(), 2)
    q25 = round(data['Value'].quantile(0.25), 2)
    q75 = round(data['Value'].quantile(0.75), 2)

    # Performance Evaluation Metrics
    def calculate_silhouette_score(data, labels):
        from sklearn.metrics import silhouette_score
        try:
            return round(silhouette_score(data[['Value']], labels), 3)
        except:
            return None

    def calculate_calinski_harabasz_score(data, labels):
        from sklearn.metrics import calinski_harabasz_score
        try:
            return round(calinski_harabasz_score(data[['Value']], labels), 3)
        except:
            return None

    def calculate_davies_bouldin_score(data, labels):
        from sklearn.metrics import davies_bouldin_score
        try:
            return round(davies_bouldin_score(data[['Value']], labels), 3)
        except:
            return None

    # Calculate performance metrics for each algorithm
    performance_metrics = {
        'K-means': {
            'silhouette': calculate_silhouette_score(data, data['anomaly_kmeans']),
            'calinski_harabasz': calculate_calinski_harabasz_score(data, data['anomaly_kmeans']),
            'davies_bouldin': calculate_davies_bouldin_score(data, data['anomaly_kmeans'])
        },
        'Isolation Forest': {
            'silhouette': calculate_silhouette_score(data, data['anomaly_if']),
            'calinski_harabasz': calculate_calinski_harabasz_score(data, data['anomaly_if']),
            'davies_bouldin': calculate_davies_bouldin_score(data, data['anomaly_if'])
        },
        'SVM': {
            'silhouette': calculate_silhouette_score(data, data['anomaly_svm']),
            'calinski_harabasz': calculate_calinski_harabasz_score(data, data['anomaly_svm']),
            'davies_bouldin': calculate_davies_bouldin_score(data, data['anomaly_svm'])
        },
        'LOF': {
            'silhouette': calculate_silhouette_score(data, data['anomaly_lof']),
            'calinski_harabasz': calculate_calinski_harabasz_score(data, data['anomaly_lof']),
            'davies_bouldin': calculate_davies_bouldin_score(data, data['anomaly_lof'])
        }
    }

    # Create performance metrics table
    performance_table = pd.DataFrame(performance_metrics).T
    performance_table = performance_table.to_html(classes='table table-striped', index=True)

    # Anomaly Statistics
    total_anomalies_iso = len(data[data['anomaly_if'] == -1])
    percentage_anomalies_iso = round((total_anomalies_iso / count) * 100, 2)

    total_anomalies_svm = len(data[data['anomaly_svm'] == -1])
    percentage_anomalies_svm = round((total_anomalies_svm / count) * 100, 2)

    total_anomalies_kmeans = len(data[data['anomaly_kmeans'] == 1])
    percentage_anomalies_kmeans = round((total_anomalies_kmeans / count) * 100, 2)

    total_anomalies_lof = len(data[data['anomaly_lof'] == -1])
    percentage_anomalies_lof = round((total_anomalies_lof / count) * 100, 2)

    # Find common anomalies across all algorithms
    data['common_anomaly'] = (
        (data['anomaly_if'] == -1) & 
        (data['anomaly_svm'] == -1) & 
        (data['anomaly_kmeans'] == 1) & 
        (data['anomaly_lof'] == -1)
    )
    
    # Debug prints to check anomaly counts
    print("Isolation Forest anomalies:", len(data[data['anomaly_if'] == -1]))
    print("SVM anomalies:", len(data[data['anomaly_svm'] == -1]))
    print("K-means anomalies:", len(data[data['anomaly_kmeans'] == 1]))
    print("LOF anomalies:", len(data[data['anomaly_lof'] == -1]))
    
    # Create a more lenient common anomaly detection
    # Consider an anomaly if it's detected by at least 3 algorithms
    data['anomaly_count'] = (
        (data['anomaly_if'] == -1).astype(int) +
        (data['anomaly_svm'] == -1).astype(int) +
        (data['anomaly_kmeans'] == 1).astype(int) +
        (data['anomaly_lof'] == -1).astype(int)
    )
    
    # Mark as common anomaly if detected by at least 3 algorithms
    data['common_anomaly'] = data['anomaly_count'] >= 3
    
    common_anomalies = data[data['common_anomaly'] == True].sort_index()
    common_anomalies_count = len(common_anomalies)
    percentage_common_anomalies = round((common_anomalies_count / count) * 100, 2)

    # Create a table of common anomalies with more information
    common_anomalies_table = common_anomalies[['Value', 'anomaly_count']].reset_index()
    common_anomalies_table.columns = ['Date', 'Value', 'Number of Algorithms Detected']
    common_anomalies_table = common_anomalies_table.to_html(classes='table table-striped', index=False)

    # Plot original data
    original_plot = plot_original_data(data)

    def plot_anomalies(data, anomalies, column_name, title):
 
        fig = px.scatter(data, x=data.index, y='Value', color=data[column_name].apply(lambda x: 'Anomaly' if x == -1 else 'Normal'), 
                     title=title, color_discrete_map={'Anomaly':'red', 'Normal':'blue'})
        return plot(fig, output_type='div')

    def generate_isolation_forest_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_if', 'Isolation Forest Anomalies')

    def generate_one_class_svm_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_svm', 'One-Class SVM Anomalies')

    # def generate_dbscan_plot(data, anomalies):
    #     return plot_anomalies(data, anomalies, 'anomaly_dbscan', 'DBSCAN Anomalies')

    def generate_kmeans_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'cluster', 'KMeans Anomalies')

    def generate_lof_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_lof', 'Local Outlier Factor Anomalies')
    
    # def generate_hmm_plot(data, anomalies):
    #     return plot_anomalies(data, anomalies, 'anomalies_hmm', 'HMM Anomalies')
    
    # def generate_hw_plot(data, anomalies):
    #     return plot_anomalies(data, anomalies, 'hw_forecast', 'HW Forecast ')
    

# Returning the plotting functions for verification
    # generate_isolation_forest_plot, generate_one_class_svm_plot, generate_dbscan_plot, generate_kmeans_plot, generate_lof_plot,generate_hmm_plot,generate_hw_plot
    generate_isolation_forest_plot, generate_one_class_svm_plot, generate_kmeans_plot, generate_lof_plot

    # Exporting the processed data
    export_filename = f"processed_{filename}"
    export_filepath = os.path.join(app.config['EXPORT_FOLDER'], export_filename)
    data.to_csv(export_filepath)

    return render_template('dashboard.html', 
                           anomalies_kmeans=anomalies_kmeans,
                           original_plot=original_plot,
                           anomalies_if=anomalies_if,
                           anomalies_svm=anomalies_svm,
                           plot_kmeans=plot_kmeans,
                           plot_if=plot_if,
                           plot_svm=plot_svm,
                           anomalies_lof=anomalies_lof,
                           plot_lof=plot_lof,
                           export_filename=export_filename,
                           count=count,
                           original_count=original_count,
                           count_before_index=count_before_index,
                           count_before_fill=count_before_fill,
                           mean=mean,
                           median=median,
                           std=std,
                           min_val=min_val,
                           max_val=max_val,
                           q25=q25,
                           q75=q75,
                           total_anomalies_iso=total_anomalies_iso,
                           percentage_anomalies_iso=percentage_anomalies_iso,
                           total_anomalies_svm=total_anomalies_svm,
                           percentage_anomalies_svm=percentage_anomalies_svm,
                           total_anomalies_kmeans=total_anomalies_kmeans,
                           percentage_anomalies_kmeans=percentage_anomalies_kmeans,
                           total_anomalies_lof=total_anomalies_lof,
                           percentage_anomalies_lof=percentage_anomalies_lof,
                           common_anomalies_table=common_anomalies_table,
                           common_anomalies_count=common_anomalies_count,
                           percentage_common_anomalies=percentage_common_anomalies,
                           performance_table=performance_table
                           )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['EXPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
