#!/usr/bin/env python3
"""
main.py

This script performs clustering on the provided
public (4D) and private (6D) datasets and outputs submission files.
It automatically tries multiple algorithms (KMeans, GMM, Spectral, Agglomerative, DBSCAN)
and selects the best by Silhouette Score for higher quality.
Optionally reduces dimensionality via PCA.

Usage:
    python main.py \
      [--public public_data.csv] \
      [--private private_data.csv] \
      [--id ID] \
      [--n-init N] [--max-iter M] \
      [--pca PCA_COMPONENTS] [--eps EPS] [--min-samples MIN_SAMPLES] \
      [--evaluate]

Outputs:
    {id}_public.csv    # placed alongside input
    {id}_private.csv
"""
import argparse
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Default student ID
DEFAULT_ID = 'b12901163'
# Algorithms to evaluate
ALGORITHMS = ['kmeans', 'gmm', 'spectral', 'agg_ward', 'agg_complete', 'agg_average', 'dbscan']


def check_file_exists(path: str):
    if not os.path.isfile(path):
        print(f"Error: File not found: '{path}'")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in this directory:", os.listdir(os.getcwd()))
        sys.exit(1)


def fit_and_score(X, n_clusters, algo, n_init, max_iter, eps, min_samples):
    # Fit model based on algorithm
    if algo == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42,
                       n_init=n_init, max_iter=max_iter, algorithm='elkan')
        labels = model.fit_predict(X)
    elif algo == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42,
                                 n_init=n_init, covariance_type='full')
        labels = model.fit_predict(X)
    elif algo == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                   n_init=n_init, assign_labels='kmeans',
                                   affinity='nearest_neighbors')
        labels = model.fit_predict(X)
    elif algo.startswith('agg_'):
        linkage = algo.split('_')[1]
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
    elif algo == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    # Compute silhouette score if meaningful
    try:
        score = silhouette_score(X, labels)
    except Exception:
        score = -1.0
    return labels, score


def process_dataset(input_path, student_id, n_clusters, n_init, max_iter, pca_components, eps, min_samples, evaluate):
    check_file_exists(input_path)
    df = pd.read_csv(input_path)
    ids = df['id'] if 'id' in df.columns else pd.Series(range(1, len(df)+1), name='id')
    features = df.drop(columns=['id']) if 'id' in df.columns else df

    # Standardize data
    X = StandardScaler().fit_transform(features.values)

    # PCA dimensionality reduction if requested
    if pca_components:
        if 0 < pca_components < 1:
            pca = PCA(n_components=pca_components, svd_solver='full', random_state=42)
        else:
            pca = PCA(n_components=int(pca_components), random_state=42)
        X = pca.fit_transform(X)
        print(f"Reduced to {X.shape[1]} PCA components (requested={pca_components})")

    # Evaluate all algorithms and pick best
    best_score, best_algo, best_labels = -1.0, None, None
    for algo in ALGORITHMS:
        labels, score = fit_and_score(X, n_clusters, algo, n_init, max_iter, eps, min_samples)
        print(f"Algorithm {algo} Silhouette: {score:.4f}")
        if score > best_score:
            best_score, best_algo, best_labels = score, algo, labels
    print(f"Selected {best_algo} with Silhouette {best_score:.4f}")

    # Save output next to input
    folder = os.path.dirname(os.path.abspath(input_path))
    prefix = os.path.basename(input_path).split('_')[0]
    output_file = f"{student_id}_{prefix}.csv"
    output_path = os.path.join(folder, output_file)
    pd.DataFrame({'id': ids, 'label': best_labels}).to_csv(output_path, index=False)
    print(f"Saved {output_path} using {best_algo}")
    if evaluate:
        print(f"Final Silhouette Score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Auto-cluster Big Data datasets')
    parser.add_argument('--public', default='public_data.csv', help='Public CSV')
    parser.add_argument('--private', default='private_data.csv', help='Private CSV')
    parser.add_argument('-i', '--id', dest='id', help='Student ID prefix')
    parser.add_argument('--n-init', type=int, default=20, help='n_init (default 20)')
    parser.add_argument('--max-iter', type=int, default=500, help='max_iter (default 500)')
    parser.add_argument('--pca', type=float, default=0,
                        help='PCA components: float<1 for variance ratio, or int count')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps (default 0.5)')
    parser.add_argument('--min-samples', type=int, default=5, help='DBSCAN min_samples (default 5)')
    parser.add_argument('--evaluate', action='store_true', help='print silhouette')
    args = parser.parse_args()

    student_id = args.id if args.id else DEFAULT_ID
    pub_clusters = 4*4 - 1
    priv_clusters = 4*6 - 1

    print("Processing public dataset...")
    process_dataset(args.public, student_id, pub_clusters,
                    args.n_init, args.max_iter, args.pca, args.eps, args.min_samples, args.evaluate)
    print("Processing private dataset...")
    process_dataset(args.private, student_id, priv_clusters,
                    args.n_init, args.max_iter, args.pca, args.eps, args.min_samples, args.evaluate)

if __name__ == '__main__':
    main()
