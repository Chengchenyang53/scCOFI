from time import time
import math, os
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
from scCOFI import scMultiCluster
import numpy as np
import collections
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize, clr_normalize_each_cell
from utils import *
from dataload import load_data, preprocess_data
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
from time import time
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_model(input_size1, input_size2, args):
    encodeLayer = list(map(int, args.encodeLayer))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))

    model = scMultiCluster(input_dim1=input_size1, input_dim2=input_size2, tau=args.tau,
                           encodeLayer=encodeLayer, decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                           activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                           lamda=args.lamda, cutoff=args.cutoff, phi1=args.phi1, phi2=args.phi2,
                           device=args.device, temperature=args.temperature).to(args.device)

    return model


def pretrain_autoencoder(model, adata1, adata2, args):
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                   X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors,
                                   batch_size=args.batch_size,
                                   epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError


def fit_model(model, adata1, adata2, y, args):
    latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))

    if args.n_clusters == -1:
        n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
        print("n_cluster is defined as " + str(args.n_clusters))
        n_clusters = args.n_clusters

    y_pred, _, _, _, _ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                   X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y,
                                   n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter,
                                   update_interval=args.update_interval, tol=args.tol, lr=1, save_dir=args.save_dir)

    return y_pred
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=-1, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='../datasets/ATAC/GSE126074.h5')
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--pretrain_epochs', default=150, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float,
                        help='fuzziness of clustering loss')
    parser.add_argument('--lamda', default=1., type=float,
                        help='fuzziness of contrastive loss')
    parser.add_argument('--phi1', default=0.001, type=float,
                        help='coefficient of KL loss')
    parser.add_argument('--phi2', default=0.001, type=float,
                        help='coefficient of KL loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='../results/')
    parser.add_argument('--ae_weight_file', default='../AEWeights/AE_weights_1.pth.tar')
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', action='store_true', default=True)
    parser.add_argument('--prediction_file', action='store_true', default=True)
    parser.add_argument('-el', '--encodeLayer', nargs='+', default=[256, 64, 32, 16])
    parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[16, 64, 256])
    parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[16, 20])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--temperature', default=0.8, type=float)
    parser.add_argument('--f1', default=2000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--filter1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(args)
    print(args.save_dir)
    # Load and preprocess data
    x1, x2, y = load_data(args.data_file)
    adata1, adata2 = preprocess_data(x1, x2, y, args.filter1, args.filter2, args.f1, args.f2)

    # Initialize model
    model = initialize_model(adata1.n_vars, adata2.n_vars, args)

    # Pretrain the autoencoder
    t0 = time()
    pretrain_autoencoder(model, adata1, adata2, args)
    print('Pretraining time: %d seconds.' % int(time() - t0))

    # Train the model and get predictions
    y_pred = fit_model(model, adata1, adata2, y, args)

    # Save predictions and embeddings if required
    if args.prediction_file:
        y_pred_ = best_map(y, y_pred) - 1
        np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred_, delimiter=",")

    if args.embedding_file:
        final_latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))
        final_latent = final_latent.cpu().numpy()
        np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")

    # Evaluation
    y_pred_ = best_map(y, y_pred)
    acc = np.round(metrics.accuracy_score(y, y_pred_), 5)
    ami = np.round(metrics.adjusted_mutual_info_score(y, y_pred_), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred_), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred_), 5)
    print('Final: ACC= %.4f, AMI= %.4f, NMI= %.4f, ARI= %.4f' % (acc, ami, nmi, ari))
