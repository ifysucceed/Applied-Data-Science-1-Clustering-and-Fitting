"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit


def plot_relational_plot(df):
    """Scatter plot of alcohol vs quality."""
    fig, ax = plt.subplots(dpi=144)
    sns.scatterplot(data=df, x="Alcohol", y="Quality", ax=ax)
    ax.set_title("Relational Plot: Alcohol vs Quality")
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    """Bar plot of wine counts by quality rating."""
    fig, ax = plt.subplots(dpi=144)

    sns.countplot(
        data=df,
        x="Quality",
        hue='Quality',
        legend=False,
        ax=ax,
        palette="viridis"
    )
    ax.set_title("Categorical Plot: Wine Counts by Quality")
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    """Correlation heatmap of selected meaningful features."""
    fig, ax = plt.subplots(dpi=144)

    # Select only relevant columns
    selected_cols = [
        "Alcohol",
        "Sulphates",
        "Citric_Acid",
        "Volatile_Acidity",
        "Residual_Sugar",
        "Density",
        "Quality"
    ]
    corr = df[selected_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        fmt=".2f"
    )
    ax.set_title("Correlation Heatmap (Selected Features)")
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    """Compute mean, stddev, skewness,
    and kurtosis for a column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Basic preprocessing:
    drop NA, check correlations.
    """
    print(df.describe())
    print(df.head())
    print(df.corr())
    df = df.dropna()

    # Ensure correct data types
    print('\nData types:')
    print(df.dtypes)
    return df


def writing(moments, col):
    """Print statistical moments in human-readable form."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    return


def perform_clustering(df, col1, col2):
    """Perform K-means clustering on two chosen columns."""
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    norm = scaler.fit_transform(data)

    # Elbow method
    wcss = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, n_init=20)
        kmeans.fit(norm)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots(dpi=144)
    ax.plot(range(2, 11), wcss, 'kx-')
    ax.set_title("Elbow Method")
    plt.savefig('elbow_plot.png')

    # Silhouette score
    best_n, best_score = None, -np.inf
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, n_init=20)
        labels = kmeans.fit_predict(norm)
        score = silhouette_score(norm, labels)
        if score > best_score:
            best_n, best_score = n, score

    # Final clustering
    kmeans = KMeans(n_clusters=best_n, n_init=20)
    labels = kmeans.fit_predict(norm)
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans, ykmeans = cen[:, 0], cen[:, 1]
    cenlabels = kmeans.predict(kmeans.cluster_centers_)
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Scatter plot of clustered data with determined centres shown."""
    fig, ax = plt.subplots(dpi=144)

    # Plot clustered data points
    xz = data[:, 0]
    yz = data[:, 1]
    scatter = ax.scatter(xz, yz, c=labels, cmap="Set1", marker='o')
    ax.scatter(
       xkmeans,
       ykmeans,
       color='black',
       marker='X',
       s=50,
       label='cluster centres'
    )

    # Add discrete colorbar for cluster IDs
    unique_labels = sorted(set(labels))
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique_labels)
    cbar.set_ticklabels([f"Cluster {i}" for i in unique_labels])

    # Axis labels
    ax.legend(loc='upper left')
    ax.set_xlabel('Alcohol')
    ax.set_ylabel('Sulphates')
    ax.set_title("Clustered Data")

    plt.savefig('clustering.png')
    return


def perform_fitting(df, col1, col2):
    """Fit a linear regression line using curve_fit."""
    x = df[col1].values
    y = df[col2].values

    def linfunc(x, a, b):
        return a * x + b

    p, cov = curve_fit(linfunc, x, y)
    sigma = np.sqrt(np.diag(cov))
    print(f"a = {p[0]:.2f} +/- {sigma[0]:.2f}")
    print(f"b = {p[1]:.2f} +/- {sigma[1]:.2f}")

    xfit = np.linspace(np.min(x), np.max(x), 100)
    yfit = linfunc(xfit, *p)
    return df, xfit, yfit


def plot_fitted_data(data, x, y):
    """Scatter plot with fitted line."""
    fig, ax = plt.subplots(dpi=144)
    ax.scatter(data["Alcohol"], data["Quality"], label="data")
    ax.plot(x, y, 'r-', label="Fitted Line")
    ax.set_title("Fitting: Alcohol vs Quality")
    ax.legend()
    plt.savefig('fitting.png')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    # Example column for statistical analysis
    col = 'Alcohol'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)

    # Clustering on alcohol vs sulphates
    clustering_results = perform_clustering(df, 'Alcohol', 'Sulphates')
    plot_clustered_data(*clustering_results)

    # Fitting alcohol vs quality
    fitting_results = perform_fitting(df, 'Alcohol', 'Quality')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
