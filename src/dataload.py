import scanpy as sc
import numpy as np
import h5py
from preprocess import read_dataset, normalize
from utils import *

def load_data(data_file):
    data_mat = h5py.File(data_file, 'r')
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X2'])
    y = np.array(data_mat['Y'])
    data_mat.close()

    return x1, x2, y


def preprocess_data(x1, x2, y, filter1, filter2, f1, f2):
    # gene filter
    if filter1:
        importantGenes = geneSelection(x1, n=f1, plot=False)
        x1 = x1[:, importantGenes]

    if filter2:
        importantGenes = geneSelection(x2, n=f2, plot=False)
        x2 = x2[:, importantGenes]


    adata1 = sc.AnnData(x1)
    adata1.obs['Group'] = y
    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y

    adata1.X = adata1.X.astype(float)
    adata1 = read_dataset(adata1, transpose=False, test_split=True, copy=True)
    adata1 = normalize(adata1, size_factors=True, normalize_input=True, logtrans_input=True)

    adata2.X = adata2.X.astype(float)
    adata2 = read_dataset(adata2, transpose=False, test_split=True, copy=True)
    adata2 = normalize(adata2, size_factors=False, normalize_input=True, logtrans_input=True)

    return adata1, adata2
