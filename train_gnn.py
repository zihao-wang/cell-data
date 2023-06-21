import argparse
from random import shuffle
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from utils.data import Case3D
from model import LUNAR

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--pretrain', type=str, default=None)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--save_every', type=int, default=5)

parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('--data_dir', type=str, default='data/synthetic')

parser.add_argument('--train_ratio', type=float, default=0.33)
parser.add_argument('--val_ratio', type=float, default=0.33)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--weight_decay', type=float, default=0.0001)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_multiple_roc_curves(predict_list, target_list, save_file=None):
    plt.figure()
    for plist, tlist in zip(predict_list, target_list):
        # sort the prediction
        fpr, tpr, _ = roc_curve(tlist,  plist)
        plt.plot(fpr, tpr, color='b', alpha=0.1)

    if save_file is not None:
        plt.savefig(save_file)


def train_step_on_case3d(case3d_graph, model, optimizer, criterion):
    model.train()
    case3d_graph.to(args.device)

    output = model(case3d_graph)
    loss = criterion(output, case3d_graph.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predict = output.cpu().detach().tolist()
    target = case3d_graph.y.cpu().tolist()

    return {"loss": loss.item(),
            "roc_auc": roc_auc_score(target, predict),
            "pred_list": predict, 'true_list': target}

def eval_on_case3d(case3d_graph, model, criterion):
    case3d_graph.to(args.device)

    model.eval()
    with torch.no_grad():
        output = model(case3d_graph)
        loss = criterion(output, case3d_graph.y)
        predict = output.cpu().detach().tolist()
        target = case3d_graph.y.cpu().tolist()
        return {"loss": loss.item(),
                "roc_auc": roc_auc_score(case3d_graph.y.cpu(), output.cpu()),
                "pred_list": predict, 'true_list': target}


def train_epoch(e, train_loader, model, optimizer, criterion):
    total_loss,  total_roc_auc = 0, 0
    roc_auc_score_list = []
    with tqdm(train_loader, desc=f"train epoch {e}") as t:
        for i, batch in enumerate(t):
            output = train_step_on_case3d(batch, model, optimizer, criterion)
            total_loss += output['loss']
            total_roc_auc += output['roc_auc']
            roc_auc_score_list.append(output['roc_auc'])
            postfix = {'ave_loss': total_loss / (i+1),
                       'ave_roc_auc': total_roc_auc / (i+1),
                       'roc_auc_95': np.quantile(roc_auc_score_list, 0.95),
                       'roc_auc_05': np.quantile(roc_auc_score_list, 0.05)}
            t.set_postfix(postfix)
            # print(compute_accuracy(target_list, predict_list))
    return {'ave_loss': total_loss / len(train_loader),
            'ave_roc_auc': total_roc_auc / len(train_loader)}


def evaluate(cases, model, criterion):
    total_loss,  total_roc_auc = 0, 0
    predict_list, target_list = [], []
    roc_auc_score_list = []
    with tqdm(cases) as t:
        for i, c in enumerate(t):
            output = eval_on_case3d(c.get_graph(model.k), model, criterion)
            total_loss += output['loss']
            total_roc_auc += output['roc_auc']
            roc_auc_score_list.append(output['roc_auc'])
            predict_list.append(output['pred_list'])
            target_list.append(output['true_list'])
            postfix = {'ave_loss': total_loss / (i+1),
                       'ave_roc_auc': total_roc_auc / (i+1),
                       'roc_auc_95': np.quantile(roc_auc_score_list, 0.95),
                       'roc_auc_05': np.quantile(roc_auc_score_list, 0.05)
                       }
            t.set_postfix(postfix)
    return predict_list, target_list


if __name__ == "__main__":

    args = parser.parse_args()
    seed_everything(args.seed)

    # prerare dataset
    # if exists, load from pickle
    if os.path.exists(os.path.join(args.data_dir, "collection.pkl")):
        with open(os.path.join(args.data_dir, "collection.pkl"), "rb") as f:
            collection_of_cases = pickle.load(f)
    # else, create from scratch
    else:
        collection_of_cases = []
        for file in tqdm(os.listdir(args.data_dir), desc="loading cases"):
            if file.endswith(".3d"):
                collection_of_cases.append(
                    Case3D(os.path.join(args.data_dir, file))
                )
        with open(os.path.join(args.data_dir, "collection.pkl"), "wb") as f:
            pickle.dump(collection_of_cases, f)

    shuffle(collection_of_cases)

    total_num = len(collection_of_cases)
    train_num = int(total_num * args.train_ratio) + 1
    valid_num = int(total_num * args.val_ratio)

    train_cases = collection_of_cases[:train_num]
    valid_cases = collection_of_cases[train_num: train_num+valid_num]
    test_cases  = collection_of_cases[train_num+valid_num:]

    print("number of training case", len(train_cases))
    print("number of validation case", len(valid_cases))
    print("number of testing case", len(test_cases))

    train_graphs = []
    for c in tqdm(train_cases):
        print(c.filename)
        train_graphs.append(c.get_graph(args.k).to(args.device))

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)

    print("prepare train loader")

    model = LUNAR(args.k)

    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))

    model.to(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction='mean')

    print("on valid cases")
    pred_list, targ_list = evaluate(valid_cases, model, criterion)
    plot_multiple_roc_curves(pred_list, targ_list,
                             save_file=os.path.join(args.save_dir,
                                                    f"valid_roc_curve_{0}.png"))

    print("on test cases")
    pred_list, targ_list = evaluate(test_cases, model, criterion)
    plot_multiple_roc_curves(pred_list, targ_list,
                             save_file=os.path.join(args.save_dir,
                                                    f"test_roc_curve_{0}.png"))


    for e in range(1, args.epochs + 1):
        print("epoch", e)
        train_epoch(e, train_loader, model, optimizer, criterion)
        if e % args.eval_every == 0:
            print("on valid cases")
            pred_list, targ_list = evaluate(valid_cases, model, criterion)
            plot_multiple_roc_curves(pred_list, targ_list,
                                     save_file=os.path.join(args.save_dir,
                                                            f"valid_roc_curve_{e}.png"))

            print("on test cases")
            pred_list, targ_list = evaluate(test_cases, model, criterion)
            plot_multiple_roc_curves(pred_list, targ_list,
                                    save_file=os.path.join(args.save_dir,
                                                            f"test_roc_curve_{e}.png"))

        if e % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{e}.pth"))
