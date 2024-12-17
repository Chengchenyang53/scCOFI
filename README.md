# scCOFI
Single Cell Cross-omics Feature Integration for Single-cell Multi-omics data clustering

# Dependencies
Python 3.8.1

Pytorch 1.6.0

Scanpy 1.6.0

SKlearn 0.22.1

Numpy 1.18.1

h5py 2.9.0

# Run scCOFI
1) Prepare the input data in h5 format. 
2) RNA and ATAC : 

`python -u run_scCOFI.py --n_clusters 'num_cluster' --ae_weight_file 'AE_weight_file' --data_file datafile
--save_dir ../result --embedding_file --prediction_file --filter1 --filter2 --f1 2000 --f2 2000 -el 256 128 64 -dl1 64 128 256 -dl2 64 128 256` 
4) RNA and ADT : 

`python -u run_scCOFI.py --n_clusters 'num_cluster' --ae_weight_file 'AE_weight_file' --data_file datafile
--save_dir ../result --embedding_file --prediction_file`

# Output of scCOFI
scCOFI outputs a latent representation of data which can be used for further downstream analyses and visualized by t-SNE or Umap.
  

