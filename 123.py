#!/usr/bin/env python3
"""
main.py

This script performs clustering on the provided
public (4D) and private (6D) datasets and outputs submission files.
It automatically tries multiple algorithms (KMeans, GMM, Spectral,
Agglomerative, DBSCAN) and selects the best by Silhouette Score.
Optionally reduces dimensionality via PCA and compares to a reference submission.

Usage:
    python main.py \
      [--public public_data.csv] \
      [--private private_data.csv] \
      [--ref-public ref_public.csv] [--ref-private ref_private.csv] \
      [--id ID] [--n-init N] [--max-iter M] \
      [--pca PCA_COMPONENTS] [--eps EPS] [--min-samples MIN_SAMPLES] \
      [--evaluate]

Outputs:
    {id}_public.csv    # alongside input
    {id}_private.csv

If reference files provided, computes silhouette for both current and reference.
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
default_id = 'b12901163'
# Available clustering algorithms
algorithms = ['kmeans', 'gmm', 'spectral', 'agg_ward', 'agg_complete', 'agg_average', 'dbscan']


def load_features(path):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        return df['id'], df.drop(columns=['id']).values
    return pd.Series(range(1, len(df) + 1), name='id'), df.values


def check_file(path):
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)


def fit_model(X, n_clusters, algo, args):
    if algo == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42,
                       n_init=args.n_init, max_iter=args.max_iter, algorithm='elkan')
        labels = model.fit_predict(X)
    elif algo == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42,
                                 n_init=args.n_init, covariance_type='full')
        labels = model.fit_predict(X)
    elif algo == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                   assign_labels='kmeans', affinity='nearest_neighbors')
        labels = model.fit_predict(X)
    elif algo.startswith('agg_'):
        linkage = algo.split('_')[1]
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
    elif algo == 'dbscan':
        model = DBSCAN(eps=args.eps, min_samples=args.min_samples)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    try:
        score = silhouette_score(X, labels)
    except Exception:
        score = -1.0
    return labels, score


def cluster_and_compare(ids, X, n_clusters, args, prefix):
    best_score = -1.0
    best_algo = None
    best_labels = None
    for algo in algorithms:
        labels, score = fit_model(X, n_clusters, algo, args)
        print(f"{prefix} {algo} silhouette: {score:.4f}")
        if score > best_score:
            best_score, best_algo, best_labels = score, algo, labels
    print(f"Best for {prefix}: {best_algo} ({best_score:.4f})")

    out_file = f"{args.id or default_id}_{prefix}.csv"
    pd.DataFrame({'id': ids, 'label': best_labels}).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

    ref_attr = f"ref_{prefix}"
    ref_path = getattr(args, ref_attr)
    if ref_path:
        check_file(ref_path)
        ref_labels = pd.read_csv(ref_path)['label']
        ref_score = silhouette_score(X, ref_labels)
        print(f"Reference {prefix} silhouette: {ref_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Auto-cluster Big Data datasets')
    parser.add_argument('--public', default='public_data.csv', help='Public CSV path')
    parser.add_argument('--private', default='private_data.csv', help='Private CSV path')
    parser.add_argument('--ref-public', help='Reference public CSV')
    parser.add_argument('--ref-private', help='Reference private CSV')
    parser.add_argument('-i', '--id', dest='id', help='Student ID prefix')
    parser.add_argument('--n-init', type=int, default=20, help='KMeans/GMM n_init')
    parser.add_argument('--max-iter', type=int, default=500, help='KMeans max_iter')
    parser.add_argument('--pca', type=float, default=0,
                        help='PCA: float<1 variance ratio or int n components')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps')
    parser.add_argument('--min-samples', type=int, default=5, help='DBSCAN min_samples')
    parser.add_argument('--evaluate', action='store_true', help='Print silhouette results')
    args = parser.parse_args()

    for path in [args.public, args.private]:
        check_file(path)

    ids_pub, X_pub = load_features(args.public)
    X_pub = StandardScaler().fit_transform(X_pub)
    if args.pca > 0:
        if args.pca < 1:
            pca = PCA(n_components=args.pca, svd_solver='full', random_state=42)
        else:
            pca = PCA(n_components=int(args.pca), random_state=42)
        X_pub = pca.fit_transform(X_pub)
        print(f"Reduced public to {X_pub.shape[1]} PCA components")
    print("=== Public ===")
    cluster_and_compare(ids_pub, X_pub, 4 * 4 - 1, args, 'public')

    ids_pr, X_pr = load_features(args.private)
    X_pr = StandardScaler().fit_transform(X_pr)
    if args.pca > 0:
        if args.pca < 1:
            pca = PCA(n_components=args.pca, svd_solver='full', random_state=42)
        else:
            pca = PCA(n_components=int(args.pca), random_state=42)
        X_pr = pca.fit_transform(X_pr)
        print(f"Reduced private to {X_pr.shape[1]} PCA components")
    print("=== Private ===")
    cluster_and_compare(ids_pr, X_pr, 4 * 6 - 1, args, 'private')

if __name__ == '__main__':
    main()
