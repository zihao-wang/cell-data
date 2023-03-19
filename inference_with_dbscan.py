"""
Given a file of 3d points, this script will cluster the points using DBSCAN and then

"""


import argparse
import os
import pickle

import faiss
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from tqdm import tqdm

from data import Case3D
from model import LUNAR

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--pretrain', type=str, default=None)

parser.add_argument('--file', type=str, required=True)

parser.add_argument('--eps', type=float, default=370)
parser.add_argument('--min_samples', type=int, default=5)


def evaluate_on_case(model, case3d):
    model.eval()
    with torch.no_grad():
        out = model(case3d)
        out = out.cpu().numpy()

    return out


if __name__ == "__main__":
    args = parser.parse_args()

    # load the model
    print(">> prepare model")
    model = LUNAR(args.k)
    model.load_state_dict(torch.load(args.pretrain))
    model.to(args.device)


    # load the file
    print(">> prepare data")
    case = Case3D(args.file)
    graph = case.get_graph(k=args.k)
    graph.to(args.device)

    # evaluate the model on case
    print(">> inferencing")
    model_prediction = evaluate_on_case(model, graph)

    points, _ = case.get_feature_label()

    print(">> dbscan clustering")
    cluster_predictions = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(points)

    dbscan_noisy_index = (cluster_predictions == -1)
    print("noisy number by db scan", dbscan_noisy_index.sum())
    print("total number of data", len(cluster_predictions))
    noisy_answer_weight = model_prediction[dbscan_noisy_index]
    mean_noisy_score = np.mean(noisy_answer_weight) # we want to minimize this score
    max_noisy_score = np.max(noisy_answer_weight)
    print("mean_noisy_score", mean_noisy_score)
    print("max_noisy_score", max_noisy_score)

    noisy_rank = np.argsort(model_prediction)[dbscan_noisy_index]
    print(np.quantile(noisy_rank, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    mean_noisy_rank = noisy_rank.mean()
    max_noisy_rank = noisy_rank.max()
    print("max_noisy_rank", max_noisy_rank)
    print("mean_noisy_rank", mean_noisy_rank)



    print(">> visualize")
    # now we want to find the best cluster
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')



    for cluster_id in np.unique(cluster_predictions):
        if cluster_id == -1:
            color = 'red'
        else:
            color = 'blue'
        ax.scatter(points[cluster_predictions == cluster_id, 0],
                   points[cluster_predictions == cluster_id, 1],
                   points[cluster_predictions == cluster_id, 2],
                   alpha=0.5,
                   color=color,
                   s=0.1)
    fig.savefig(f"DBSCAN clustering with eps={args.eps}, min_samples={args.min_samples} and noisy_score={mean_noisy_score:.3f}.png")


    plt.figure()
    plt.hist(model_prediction, bins=100)
    plt.savefig(f"histogram of model prediction.png")