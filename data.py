import os

import faiss
import numpy as np
import torch
from torch_geometric.data import Data


class Case3D():
    def __init__(self, filename=None):
        self.filename = filename
        if filename is None:
            self.coord_tensor = None
            self.label_tensor = None
            return

        # load file name
        coords = []
        labels = []
        with open(filename, 'rt') as f:
            for l in f.readlines():
                x, y, z, *_ = l.strip().split()
                coords.append(
                    [float(x), float(y), float(z)]
                )

        try:
            total_num, ratio, *_ = os.path.basename(filename).split('_')
            total_num = int(total_num)
            ratio = float(ratio)
            assert len(coords) == total_num
            for i in range(total_num):
                if i / total_num < ratio:
                    labels.append(0)
                else:
                    labels.append(1)
        except:
            # if one cannot proceed this file, we fill in dummy labels
            for i in range(len(coords)):
                labels.append(-1)

        self.coord_arr = np.asarray(coords, dtype=np.float32)
        self.label_arr = np.asarray(labels, dtype=np.float32)

    def get_feature_label(self):
        return self.coord_arr, self.label_arr

    def get_graph(self, k=10):
        faiss_index = faiss.IndexFlatL2(3)
        faiss_index.add(self.coord_arr)
        distance, target_idx = faiss_index.search(self.coord_arr, k=k+1)
        distance, target_idx = distance[:, 1:], target_idx[:, 1:]
        source_idx = np.repeat(np.arange(len(self.coord_arr)),
                                 target_idx.shape[-1])
        flat_dist, flat_srcidx, flat_tgtidx = distance.flatten(), source_idx.flatten(), target_idx.flatten()
        index = np.vstack((flat_srcidx, flat_tgtidx))

        x = torch.tensor(self.coord_arr, dtype=torch.float32)
        y = torch.tensor(self.label_arr, dtype=torch.float32)
        edge_index = torch.tensor(index, dtype=torch.long)
        edge_attr = torch.tensor(flat_dist.reshape(-1, 1))

        return Data(x, edge_index, edge_attr, y)
