# scCOFI
Single Cell Cross-omics Feature Integration for Single-cell Multi-omics data clustering

# Dependencies
Python 3.8.1

Pytorch 1.6.0

Scanpy 1.6.0

SKlearn 0.22.1

Numpy 1.18.1

h5py 2.9.0

### Setting Up the Environment
1) **Create a virtual environment:**

`conda create -n scCOFI python=3.8` 

` conda activate scCOFI` 

2) **Install dependencies:**

`pip install -r requirements.txt` 

3) **Update transformer:**

Before run, please carefully read  'S_update_transformer.docs', and refer to the steps inside it to modify the code in order to get S.

# Run scCOFI
1) Prepare the input data in h5 format.
   
2) RNA and ATAC : 

`python -u run_scCOFI.py --n_clusters 'num_cluster' --ae_weight_file 'AE_weight_file' --data_file datafile
--save_dir ../result --embedding_file --prediction_file --filter1 --filter2 --f1 2000 --f2 2000 -el 256 128 64 -dl1 64 128 256 -dl2 64 128 256` 

3) RNA and ADT : 

`python -u run_scCOFI.py --n_clusters 'num_cluster' --ae_weight_file 'AE_weight_file' --data_file datafile
--save_dir ../result --embedding_file --prediction_file`

# Output of scCOFI
scCOFI outputs a latent representation of data which can be used for further downstream analyses and visualized by t-SNE or Umap.
  

