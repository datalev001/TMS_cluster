import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Define the trend function including t and t^2 terms
def trend_function(t, a, b, c, d):
    return 1 / (np.exp(b * (t - a)) + np.sqrt(1 + (t - a)**2)) + c * t + d * t**2

# Fit the trend and extract coefficients before likelihood calculation
def fit_trend(data, t):
    trend_coeffs = []
    for i in range(data.shape[0]):
        y = data.iloc[i, :].values
        popt = np.polyfit(t, y, 2)  # Fit quadratic trend (t and t^2)
        trend_coeffs.append(popt)
    return np.array(trend_coeffs)

# Remove trend from data before applying AR-MA modeling
def detrend_data(data, trend_coeffs, t):
    detrended_data = data.copy()
    for i in range(data.shape[0]):
        trend = np.polyval(trend_coeffs[i], t)  # Use the trend with t and t^2
        detrended_data.iloc[i, :] -= trend
    return detrended_data

# Define a log-likelihood function for AR(1) + MA(1)
def full_log_likelihood(params, X, cluster_assignments, n_clusters, ar_order=1, ma_order=1, reg=1e-4):
    n, m = X.shape  # n: number of curves, m: number of time points
    likelihood = 0
    idx = 0

    for k in range(n_clusters):
        # Extract AR(1) and MA(1) parameters for cluster k
        ar_params = params[idx:idx+ar_order]
        ma_params = params[idx+ar_order:idx+ar_order+ma_order]
        sigma_root = np.abs(params[idx+ar_order+ma_order])  # Ensure sigma_root is positive
        sigma = sigma_root**2 + reg  # Square it to guarantee positive variance
        idx += ar_order + ma_order + 1  # ar_order + ma_order + sigma_root
        
        for i in range(n):  # Loop over each time series in X
            if cluster_assignments[i] == k:
                ar_component = np.zeros(m)
                ma_component = np.zeros(m)
                residuals = np.zeros(m)
                
                for j in range(1, m):  # AR(1) and MA(1)
                    ar_component[j] = ar_params[0] * X[i, j - 1]
                    residuals[j] = X[i, j] - ar_component[j]
                    ma_component[j] = ma_params[0] * residuals[j - 1]

                total_component = ar_component + ma_component
                # Gaussian log-likelihood for this time series in cluster k
                cov_matrix = np.eye(m) * sigma  # Simplified: AR(1) + MA(1)
                likelihood += np.sum(multivariate_normal.logpdf(X[i], mean=total_component, cov=cov_matrix))
    
    return -likelihood

# Fit the AR(1) + MA(1) model with fewer iterations and simpler settings
def fit_gmm_with_prior(X, n_clusters=3, ar_order=1, ma_order=1):
    n, m = X.shape
    
    # Use KMeans to initialize cluster assignments for better starting points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(X)
    
    # Initialize AR(1), MA(1), and sigma_root for each cluster, using distinct initial values
    initial_params = []
    for k in range(n_clusters):
        initial_params += [0.5 + np.random.uniform(-0.1, 0.1)] * ar_order  # Initial AR parameters
        initial_params += [0.3 + np.random.uniform(-0.05, 0.05)] * ma_order  # Initial weaker MA parameters
        initial_params += [1.0 + np.random.uniform(0, 0.1)]  # Initial sigma_root (positive square root)
    
    # Minimize the negative log-likelihood with fewer iterations
    result = minimize(full_log_likelihood, initial_params, args=(X, cluster_assignments, n_clusters, ar_order, ma_order), 
                      method='L-BFGS-B', options={'maxiter': 500})  # Reduced iteration count
    
    if not result.success:
        print("Optimization failed:", result.message)

    return result, cluster_assignments

# Adjust clusters with threshold for even distribution
def adjust_clusters_with_threshold(X, params, cluster_assignments, n_clusters, ar_order=1, ma_order=1):
    n, m = X.shape
    idx = 0
    log_likelihoods = np.zeros((n, n_clusters))  # To store log-likelihoods for each time series and cluster
    cluster_sizes = np.bincount(cluster_assignments, minlength=n_clusters)
    max_cluster_size = n // n_clusters  # The target number of samples per cluster

    # Compute log-likelihood for each time series and each cluster
    for k in range(n_clusters):
        ar_params = params[idx:idx+ar_order]
        ma_params = params[idx+ar_order:idx+ar_order+ma_order]
        sigma_root = np.abs(params[idx+ar_order+ma_order])  # Ensure positive sigma_root
        sigma = sigma_root**2  # Square it to ensure positive variance
        idx += ar_order + ma_order + 1  # ar_order + ma_order + sigma_root

        for i in range(n):  # Loop over each time series in X
            ar_component = np.zeros(m)
            ma_component = np.zeros(m)
            residuals = np.zeros(m)
            
            for j in range(1, m):  # AR(1) and MA(1)
                ar_component[j] = ar_params[0] * X[i, j - 1]
                residuals[j] = X[i, j] - ar_component[j]
                ma_component[j] = ma_params[0] * residuals[j - 1]

            total_component = ar_component + ma_component
            # Gaussian log-likelihood for this time series in cluster k
            cov_matrix = np.eye(m) * sigma  # Simplified: AR(1) + MA(1)
            log_likelihoods[i, k] = np.sum(multivariate_normal.logpdf(X[i], mean=total_component, cov=cov_matrix))

    # Reassign observations to balance clusters
    for k in range(n_clusters):
        while cluster_sizes[k] > max_cluster_size:
            # Find the observations in this cluster with the lowest likelihood
            cluster_members = np.where(cluster_assignments == k)[0]
            worst_members = cluster_members[np.argsort(log_likelihoods[cluster_members, k])]
            # Move the worst members to the smallest cluster
            for obs in worst_members:
                smallest_cluster = np.argmin(cluster_sizes)
                if cluster_sizes[smallest_cluster] < max_cluster_size:
                    cluster_assignments[obs] = smallest_cluster
                    cluster_sizes[k] -= 1
                    cluster_sizes[smallest_cluster] += 1
                    if cluster_sizes[smallest_cluster] >= max_cluster_size:
                        break

    return cluster_assignments


# get stock data
day_data = pd.read_csv('smb_price.csv')
tks = list(day_data.ticker)

result_df = pd.DataFrame([])  

for ticker in day_data['ticker'].unique():
    df_ticker = day_data[day_data['ticker'] == ticker].sort_values(by='Date')
    
    # Smooth the prices and volumes by averaging over 5 consecutive points
    df_ticker['Close_smoothed'] = df_ticker['Close'].rolling(window=5, min_periods=1).mean()
    df_ticker['Volume_smoothed'] = df_ticker['Volume'].rolling(window=5, min_periods=1).mean()

    # x values for interpolation (after smoothing)
    x_vals = np.arange(len(df_ticker))

    # Spline interpolation on the smoothed data
    spline_price = make_interp_spline(x_vals, df_ticker['Close_smoothed'], k=3)(np.linspace(x_vals.min(), x_vals.max(), 450))
    spline_volume = make_interp_spline(x_vals, df_ticker['Volume_smoothed'], k=3)(np.linspace(x_vals.min(), x_vals.max(), 450))

    # Select 30 evenly spaced points from the smoothed curves
    indices = np.linspace(0, 449, 100, dtype=int)  # Corrected the upper bound

    # Prepare the smoothed data DataFrame for this ticker
    df_smooth = pd.DataFrame({
        'ticker': ticker,
        'smooth_price': spline_price[indices],
        'smooth_volume': spline_volume[indices],
        'seq': np.arange(100, 0, -1)})

# The processed data for each stock is appended to the result_df DataFrame
result_df = pd.concat([result_df, df_smooth], ignore_index=True)

# Main Execution for Stock Data Clustering
def run_clustering(df, ticker_n=30, time_points=20, cluster_n=3, adjust_flag='N'):
    # Step 1: Preprocess the data to get a matrix form for clustering
    df_pivot = preprocess_data(df, ticker_n=ticker_n, time_points=time_points)

    # Step 2: Prepare the time points (seq) from 1 to `time_points`
    t = np.arange(1, time_points + 1)

    # Step 3: Fit the trend for each stock
    trend_coeffs = fit_trend(df_pivot, t)

    # Step 4: Detrend the data
    X_detrended = detrend_data(df_pivot, trend_coeffs, t)

    # Step 5: Perform GMM-based clustering
    result, cluster_assignments = fit_gmm_with_prior(X_detrended.values, n_clusters=cluster_n, ar_order=1, ma_order=1)

    # Step 6: Adjust or use original labels based on the adjust_flag
    if adjust_flag == 'Y':
        adjusted_labels = adjust_clusters_with_threshold(X_detrended.values, result.x, cluster_assignments, n_clusters=cluster_n, ar_order=1, ma_order=1)
    else:
        adjusted_labels = cluster_assignments  # No adjustment

    # Step 7 (a): Normalize the curves to [0,1] for each stock before plotting
    def normalize_data(df):
        df_normalized = df.copy()
        for i in range(df.shape[0]):
            min_val = df.iloc[i, :].min()
            max_val = df.iloc[i, :].max()
            df_normalized.iloc[i, :] = (df.iloc[i, :] - min_val) / (max_val - min_val)  # Normalize to [0,1]
        return df_normalized

    # Normalize the detrended data before plotting the clustered stock curves
    df_normalized = normalize_data(df_pivot)

    # Step 7 (b): Plot stock curves for each cluster in a single frame with subplots
    fig, axes = plt.subplots(cluster_n, 1, figsize=(10, cluster_n * 4))  # Create subplots: one row per cluster
    colors = ['r', 'g', 'b', 'c'][:cluster_n]

    for k in range(cluster_n):  # Plot each cluster in its subplot
        cluster_curves = df_normalized[adjusted_labels == k]  # Use normalized data for visualization
        ax = axes[k]  # Access the subplot for cluster k
        for i in range(len(cluster_curves)):
            ax.plot(cluster_curves.iloc[i], alpha=0.3, color=colors[k])
        ax.set_title(f"Normalized Stock Curves for Cluster {k+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized Price")

    plt.tight_layout()
    plt.show()

    # Step 8: Proportion of units in each cluster
    proportion_of_clusters = np.bincount(adjusted_labels) / len(adjusted_labels)

    # Step 9: Extract AR, MA, and sigma parameters for each cluster
    cluster_coefficients = extract_coefficients(result.x, n_clusters=cluster_n, ar_order=1, ma_order=1)

    # Step 10: Create a DataFrame to map tickers to their cluster labels
    ticker_cluster_df = pd.DataFrame({
        'ticker': df_pivot.index,
        'cluster': adjusted_labels
    })

    # Step 11: Print the estimated parameters for each cluster
    print("\nEstimated Parameters for Each Cluster:")
    for k, params in enumerate(cluster_coefficients):
        ar_params, ma_params, sigma = params
        print(f"Cluster {k} AR(1) Parameter: {ar_params}")
        print(f"Cluster {k} MA(1) Parameter: {ma_params}")
        print(f"Cluster {k} Variance (sigma^2): {sigma}")

    # Step 12: Print the proportion of units in each cluster
    print("\nProportion of Units in Each Cluster:")
    for k in range(cluster_n):
        print(f"Cluster {k} Proportion: {proportion_of_clusters[k]:.2%}")

    # Step 13: Plot the proportion of clusters
    plt.figure(figsize=(6, 4))
    plt.bar(range(cluster_n), proportion_of_clusters, color=['r', 'g', 'b', 'c'][:cluster_n])
    plt.xticks(range(cluster_n), [f'Cluster {k}' for k in range(cluster_n)])
    plt.title("Proportion of Units in Each Cluster")
    plt.ylabel("Proportion")
    plt.show()

    # Step 14: Return the results
    return {
        'ticker_cluster_df': ticker_cluster_df, 
        'cluster_coefficients': cluster_coefficients, 
        'proportion_of_clusters': proportion_of_clusters
    }


# Example:
    
    
df = result_df.copy()
result = run_clustering(df, ticker_n=40, time_points=20, cluster_n=3, adjust_flag='N')
result = run_clustering(df, ticker_n=30, time_points=20, cluster_n=3, adjust_flag='Y')
result = run_clustering(df, ticker_n=40, time_points=20, cluster_n=4, adjust_flag='N')
result = run_clustering(df, ticker_n=40, time_points=30, cluster_n=4, adjust_flag='Y')

# To access results:
result['ticker_cluster_df']  # Contains tickers and their corresponding clusters
result['cluster_coefficients']  # AR(1), MA(1), and sigma estimates for each cluster
result['proportion_of_clusters']  # Proportion of data points in each cluster

