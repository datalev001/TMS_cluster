import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Step 1: Define the trend function including t and t^2 terms
def trend_function(t, a, b, c, d):
    return 1 / (np.exp(b * (t - a)) + np.sqrt(1 + (t - a)**2)) + c * t + d * t**2

# Step 2: Fit the trend and extract coefficients before likelihood calculation
def fit_trend(data, t):
    trend_coeffs = []
    for i in range(data.shape[0]):
        y = data.iloc[i, :].values
        popt = np.polyfit(t, y, 2)  # Fit quadratic trend (t and t^2)
        trend_coeffs.append(popt)
    return np.array(trend_coeffs)

# Step 3: Remove trend from data before applying AR-MA modeling
def detrend_data(data, trend_coeffs, t):
    detrended_data = data.copy()
    for i in range(data.shape[0]):
        trend = np.polyval(trend_coeffs[i], t)  # Use the trend with t and t^2
        detrended_data.iloc[i, :] -= trend
    return detrended_data

# Step 4: Define a log-likelihood function for AR(1) + MA(1)
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

# Step 5: Fit the AR(1) + MA(1) model with fewer iterations and simpler settings
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

# Step 6: Forced Allocation for Even Cluster Distribution
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

# Step 7: Generate synthetic data for 150 curves and 3 clusters with AR(1) + MA(1)
def generate_group_data(n_curves=150, n_points=50, trend_params=None, ar_params=None, ma_params=None, noise_level=0.01):
    np.random.seed(42)
    data = []
    
    for i in range(n_curves):
        params = trend_params
        t = np.linspace(0, 10, n_points)
        trend = trend_function(t, params['a'], params['b'], params['c'], params['d'])  # Use the updated trend function
        
        # Generate autoregressive component with AR(1)
        ar_component = np.zeros(n_points)
        ma_component = np.zeros(n_points)
        residuals = np.zeros(n_points)
        
        for j in range(1, n_points):
            ar_component[j] = ar_params['p1'] * ar_component[j - 1] + np.random.normal(0, noise_level)
            residuals[j] = ar_component[j] + np.random.normal(0, noise_level)
            ma_component[j] = ma_params['q1'] * residuals[j - 1]
        
        noise = np.random.normal(0, noise_level, n_points)  # Reduced noise level
        y = trend + ar_component + ma_component + noise
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize to [0, 1]
        data.append(y)
    return pd.DataFrame(data, columns=[f'Time_{i}' for i in range(n_points)])

# Define more distinct parameters for each cluster
# Define more distinct parameters for each cluster
g1_params = {'a': 1, 'b': 1.5, 'c': 0.1, 'd': 0.02}
g2_params = {'a': 3.5, 'b': 0.7, 'c': -0.2, 'd': 0.01}
g3_params = {'a': 5.5, 'b': 0.3, 'c': 0.05, 'd': -0.01}

g1_ar_params = {'p1': 0.5}
g2_ar_params = {'p1': 0.8}
g3_ar_params = {'p1': 0.3}

g1_ma_params = {'q1': 0.2}  # Weaker MA parameter for cluster 1
g2_ma_params = {'q1': 0.15}  # Weaker MA parameter for cluster 2
g3_ma_params = {'q1': 0.1}  # Weaker MA parameter for cluster 3

# Step 8: Generate the data for the three clusters with AR(1) + MA(1)
DF_g1 = generate_group_data(n_curves=50, n_points=50, trend_params=g1_params, ar_params=g1_ar_params, ma_params=g1_ma_params, noise_level=0.01)
DF_g2 = generate_group_data(n_curves=50, n_points=50, trend_params=g2_params, ar_params=g2_ar_params, ma_params=g2_ma_params, noise_level=0.01)
DF_g3 = generate_group_data(n_curves=50, n_points=50, trend_params=g3_params, ar_params=g3_ar_params, ma_params=g3_ma_params, noise_level=0.01)

# Combine and shuffle the data
DF_combined = pd.concat([DF_g1, DF_g2, DF_g3], ignore_index=True)
DF_combined = DF_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 9: Prepare the time points and data matrix
t = np.linspace(0, 10, 50)
X = DF_combined.values

# Step 8: Plot the raw data before clustering with colors representing the true generated clusters
plt.figure(figsize=(10, 6))

# Define colors for the 3 clusters (generated)
colors = ['r', 'g', 'b']

# Plot each of the true generated clusters with different colors
for i in range(50):  # First 50 are Cluster 1
    plt.plot(DF_combined.iloc[i, :], color=colors[2], alpha=0.6)

for i in range(50, 100):  # Next 50 are Cluster 2
    plt.plot(DF_combined.iloc[i, :], color=colors[2], alpha=0.6)

for i in range(100, 150):  # Last 50 are Cluster 3
    plt.plot(DF_combined.iloc[i, :], color=colors[2], alpha=0.6)

plt.title("Generated Data Before Clustering (True Clusters)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()


# Step 10: Pre-fit the trend using curve fitting
trend_coeffs = fit_trend(DF_combined, t)

# Step 11: Detrend the data before applying AR-MA modeling
X_detrended = detrend_data(DF_combined, trend_coeffs, t).values

# Step 12: Fit the full AR(1) + MA(1) model with even cluster prior
result, cluster_assignments = fit_gmm_with_prior(X_detrended, n_clusters=3, ar_order=1, ma_order=1)
print(f"Optimization success: {result.success}")
print(f"Estimated parameters: {result.x}")

# Step 13: Apply forced allocation for even distribution
adjusted_labels = adjust_clusters_with_threshold(X_detrended, result.x, cluster_assignments, n_clusters=3, ar_order=1, ma_order=1)

# Step 14: Plot the clustering results after forced allocation
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']

for k in range(3):  # We have 3 clusters now
    cluster_curves = DF_combined[adjusted_labels == k]
    for i in range(len(cluster_curves)):
        plt.plot(cluster_curves.iloc[i], alpha=0.3, color=colors[k])

plt.title("AR(1) + MA(1) Model Clustering on Detrended Curves (With Forced Allocation)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# Step 18: Extract parameter estimates for each cluster
def extract_cluster_params(result, n_clusters, ar_order, ma_order):
    cluster_params = []
    idx = 0
    for k in range(n_clusters):
        ar_params = result.x[idx:idx+ar_order]
        ma_params = result.x[idx+ar_order:idx+ar_order+ma_order]
        sigma_root = result.x[idx+ar_order+ma_order]
        sigma = sigma_root**2  # Square sigma_root to get variance
        cluster_params.append((ar_params, ma_params, sigma))
        idx += ar_order + ma_order + 1  # Move to the next set of parameters
    return cluster_params

# Step 19: Extract AR(1), MA(1), and sigma parameters for each cluster
cluster_params = extract_cluster_params(result, n_clusters=3, ar_order=1, ma_order=1)

# Step 20: Print the estimated parameters for each cluster
for k, params in enumerate(cluster_params):
    ar_params, ma_params, sigma = params
    print(f"Cluster {k} AR(1) Parameter: {ar_params}")
    print(f"Cluster {k} MA(1) Parameter: {ma_params}")
    print(f"Cluster {k} Variance (sigma^2): {sigma}")

# Step 21: Calculate cluster unit proportion
cluster_counts = np.bincount(adjusted_labels, minlength=3)
cluster_proportion = cluster_counts / len(adjusted_labels)

# Step 22: Print the proportion of units in each cluster
for k in range(3):
    print(f"Cluster {k} Proportion: {cluster_proportion[k]:.2%}")

# Step 23: Plot the proportion of clusters
plt.figure(figsize=(6, 4))
plt.bar(range(3), cluster_proportion, color=['r', 'g', 'b'])
plt.xticks(range(3), ['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.title("Proportion of Units in Each Cluster (With Forced Allocation)")
plt.ylabel("Proportion")
plt.show()

# Step 24: Final Results Summary
print("\n### Final Summary ###")
print(f"Optimization Success: {result.success}")
print(f"Final Estimated Parameters (All Clusters): {result.x}")
print("Cluster Proportions:", cluster_proportion)

