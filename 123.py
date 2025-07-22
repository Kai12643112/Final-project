#!/usr/bin/env python3
"""
main.py

This script performs clustering on the provided

public (4D) and private (6D) datasets and outputs submission files.
It automatically tries multiple algorithms (KMeans, GMM, Spectral,
Agglomerative, DBSCAN) and selects the best by Silhouette Score.
You can optionally try multiple random seeds for KMeans/GMM to find the best initialization,
and reduce dimensionality via PCA.

Usage:
    python main.py \
      --public public_data.csv \
      --private private_data.csv \
      [--id ID] [--seeds seed1,seed2,...] [--n-init N] [--max-iter M] \
      [--pca PCA_COMPONENTS] [--eps EPS] [--min-samples MIN_SAMPLES] \
      [--evaluate]

Outputs:
    public_submission.csv    # public dataset labels
    private_submission.csv   # private dataset labels
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

# Default student ID if not provided
default_id = 'b12901163'
# Supported algorithms
default_algorithms = ['kmeans', 'gmm', 'spectral', 'agg_ward', 'agg_complete', 'agg_average', 'dbscan']


def load_features(path):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        return df['id'], df.drop(columns=['id']).values
    return pd.Series(range(1, len(df)+1), name='id'), df.values


def check_file(path):
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)


def cluster_and_save(ids, X, n_clusters, args, prefix):
    print(f"* Evaluating algorithms for {prefix} dataset *")
    best_score = -1.0
    best_algo = None
    best_labels = None
    for algo in default_algorithms:
        if algo == 'kmeans':
            # Try multiple seeds for KMeans
            best_labels, score, seed = fit_with_seeds(
                lambda **kw: KMeans(**kw), X, n_clusters, args, args.seeds,
                n_init=args.n_init, max_iter=args.max_iter, algorithm='elkan'
            )
            print(f"  kmeans best seed={seed}")
        elif algo == 'gmm':
            best_labels, score, seed = fit_with_seeds(
                lambda **kw: GaussianMixture(**kw), X, n_clusters, args, args.seeds,
                n_components=n_clusters, n_init=args.n_init, covariance_type='full'
            )
            print(f"  gmm best seed={seed}")
        else:
            # single-run for other algorithms
            if algo == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', affinity='nearest_neighbors')
            elif algo.startswith('agg_'):
                linkage = algo.split('_')[1]
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            elif algo == 'dbscan':
                model = DBSCAN(eps=args.eps, min_samples=args.min_samples)
            else:
                continue
            labels = model.fit_predict(X)
            try:
                score = silhouette_score(X, labels)
            except:
                score = -1.0
        print(f"{prefix} {algo} silhouette: {score:.4f}")
        if score > best_score:
            best_score, best_algo, best_labels = score, algo, best_labels if algo in ['kmeans','gmm'] else labels
    print(f"Selected {best_algo} for {prefix} (Silhouette={best_score:.4f})")

    # Use fixed submission filenames
    out_file = 'public_submission.csv' if prefix == 'public' else 'private_submission.csv'
    pd.DataFrame({'id': ids, 'label': best_labels}).to_csv(out_file, index=False)
    print(f"Saved {out_file}\n")


def fit_with_seeds(model_cls, X, n_clusters, args, seeds, **kwargs):
    best_score = -1.0
    best_labels = None
    best_seed = None
    for seed in seeds:
        model = model_cls(random_state=seed, n_clusters=n_clusters, **kwargs)
        labels = model.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except:
            score = -1.0
        if score > best_score:
            best_score, best_labels, best_seed = score, labels, seed
    return best_labels, best_score, best_seed


def preprocess(path, args):
    check_file(path)
    ids, features = load_features(path)
    X = StandardScaler().fit_transform(features)
    if args.pca > 0:
        n = args.pca
        if n < 1:
            pca = PCA(n_components=n, svd_solver='full')
        else:
            pca = PCA(n_components=int(n))
        X = pca.fit_transform(X)
        print(f"Reduced {path} to {X.shape[1]} PCA components")
    return ids, X


def main():
    parser = argparse.ArgumentParser(description='Auto-cluster Big Data datasets')
    parser.add_argument('--public', default='public_data.csv', help='Public CSV path')
    parser.add_argument('--private', default='private_data.csv', help='Private CSV path')
    parser.add_argument('-i', '--id', dest='id', help='Student ID prefix')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated random seeds for KMeans/GMM')
    parser.add_argument('--n-init', type=int, default=20, help='n_init for KMeans/GMM')
    parser.add_argument('--max-iter', type=int, default=500, help='max_iter for KMeans')
    parser.add_argument('--pca', type=float, default=0, help='PCA components (float<1 or int)')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps')
    parser.add_argument('--min-samples', type=int, default=5, help='DBSCAN min_samples')
    parser.add_argument('--evaluate', action='store_true', help='Print silhouette results only')
    args = parser.parse_args()

    args.seeds = [int(s) for s in args.seeds.split(',') if s.strip().isdigit()]

    print("\n== Public dataset processing ==")
    ids_pub, X_pub = preprocess(args.public, args)
    cluster_and_save(ids_pub, X_pub, 4*4-1, args, 'public')

    print("\n== Private dataset processing ==")
    ids_pr, X_pr = preprocess(args.private, args)
    cluster_and_save(ids_pr, X_pr, 4*6-1, args, 'private')

if __name__ == '__main__':
    main()
