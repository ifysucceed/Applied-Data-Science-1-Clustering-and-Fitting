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
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
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
    """
    Perform K-means clustering on two chosen columns.
    """

    def plot_elbow_method():
        """Plot elbow method using WCSS and best_n from outer scope."""
        fig, ax = plt.subplots(dpi=144)
        ks = range(min_k, max_k + 1)
        ax.plot(ks, wcss, "kx-")
        ax.scatter(best_n, wcss[best_n - min_k],
                   marker="o", color="red", facecolors="none", s=50)
        ax.set_xlabel("k")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method")
        plt.tight_layout()
        plt.savefig("elbow_plot.png")
        return

    def one_silhouette_inertia():
        """Calculate silhouette score and WCSS for n clusters (using outer scope n, norm)."""
        kmeans_local = KMeans(n_clusters=n, n_init=20, random_state=42)
        kmeans_local.fit(norm)
        labels_local = kmeans_local.labels_
        _score = silhouette_score(norm, labels_local)
        _inertia = kmeans_local.inertia_
        return _score, _inertia

    # Gather data and scale
    data = df[[col1, col2]].to_numpy()
    scaler = StandardScaler()
    norm = scaler.fit_transform(data)

    # Find best number of clusters
    min_k, max_k = 2, 10
    best_n, best_score = None, -np.inf
    wcss = []
    for n in range(min_k, max_k + 1):
        score, inertia = one_silhouette_inertia()
        wcss.append(inertia)
        if score > best_score:
            best_n, best_score = n, score
        print(f"{n} clusters silhouette score = {score:.2f}")
    print(f"Best number of clusters = {best_n}")

    plot_elbow_method()

    # Get cluster centers
    kmeans = KMeans(n_clusters=best_n, n_init=20, random_state=42)
    kmeans.fit(norm)
    labels = kmeans.labels_
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans, ykmeans = centers[:, 0], centers[:, 1]
    cenlabels = kmeans.predict(kmeans.cluster_centers_)

    # Overwrite data with inverse-transformed values so return matches template
    data = scaler.inverse_transform(norm)

    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Creates a scatter plot of clustered data with centres shown as a black bold X.
    -Data points are colored by cluster label.
    """
    fig, ax = plt.subplots(dpi=144)

    # Plot clustered data
    scatter = ax.scatter(data[:, 0], data[:, 1],
                         c=labels, cmap="Set1",
                         marker="o")
    # Plot cluster centres
    ax.scatter(xkmeans, ykmeans,
               color="black", marker="X", s=50,
               label="Cluster Centres")

    # Add colorbar for cluster points
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks(np.unique(labels))

    ax.legend(loc="upper left")
    ax.set_xlabel("Alcohol")
    ax.set_ylabel("Sulphates")
    ax.set_title("Clustered Data")

    plt.tight_layout()
    plt.savefig("clustering.png")
    return


def perform_fitting(df, col1, col2):
    """Fitting a straight line between one feature = Alcohol
    and one target variable = Quality.
    """
    x = df[col1].to_numpy()
    y = df[col2].to_numpy()

    def linfunc(xvals, a, b):
        return a * xvals + b

    p, cov = curve_fit(linfunc, x, y)
    sigma = np.sqrt(np.diag(cov))
    a, b = p
    print(f"a = {a:.2f} +/- {sigma[0]:.2f}")
    print(f"b = {b:.2f} +/- {sigma[1]:.2f}")

    xfit = np.linspace(np.min(x), np.max(x), 100)
    yfit = linfunc(xfit, a, b)
    data = np.vstack([xfit, yfit])
    return data, x, y
    

def plot_fitted_data(data, x, y):
    """Scatter plot of data with fitted line."""
    fig, ax = plt.subplots(dpi=144)
    ax.scatter(x, y, label="Data")
    ax.plot(data[0], data[1], "r-", label="Fitted Line")
    ax.set_xlabel("Feature Var = Alcohol")
    ax.set_ylabel("Target Var = Quality")
    ax.set_title("Fitting: Alcohol vs Quality")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("fitting.png")
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    # statistical moments column selection
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
