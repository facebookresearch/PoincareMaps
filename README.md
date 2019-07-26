# PoincareMaps

Poincare maps recover continuous hierarchies in single-cell data.

POC: Anna Klimovskaia (klanna@fb.com)

## Dependecies
python3.7
anaconda (sklearn, numpy, pandas, scipy)
seaborn

Pytorch: https://pytorch.org/get-started/locally/


## To replicate our experiments

# Embedding
```bash
python main.py --dset ToggleSwitch       --batchsize -1 --cuda 1 --knn 15 --gamma 2.0 --sigma 1.0 --pca 0  --root root
python main.py --dset MyeloidProgenitors --batchsize -1 --cuda 1 --knn 30 --gamma 2.0 --sigma 2.0 --pca 0  --root root
python main.py --dset krumsiek11_blobs   --batchsize -1 --cuda 1 --knn 30 --gamma 2.0 --sigma 1.0 --pca 20 --root root

python main.py --dset Olsson   			 --batchsize -1 --cuda 1 --knn 15 --gamma 2.0 --sigma 1.0 --pca 20 --root HSPC-1
python main.py --dset Paul               --batchsize -1 --cuda 1 --knn 15 --gamma 2.0 --sigma 1.0 --pca 20 --root root
python main.py --dset Moignard2015       --batchsize -1 --cuda 1 --knn 30 --gamma 1.0 --sigma 2.0 --pca 0  --root PS
python main.py --dset Planaria           --batchsize -1 --cuda 1 --knn 15 --gamma 2.0 --sigma 2.0 --pca 0 --root neoblast\ 1

python main.py --dset MyeloidProgenitors --batchsize -1 --cuda 1 --knn 30 --gamma 2.0 --sigma 2.0 --pca 0  --root root
python main.py --dset Olsson   			 --batchsize -1 --cuda 1 --knn 15 --gamma 2.0 --sigma 1.0 --pca 20 --root HSPC-1
python main.py --dset Planaria           --batchsize -1 --cuda 1 --knn 15 --gamma 2.0 --sigma 2.0 --pca 0 --root neoblast\ 1
```

# Prediction
```bash
python decoder.py --dset Planaria --cuda 1 --method poincare
python decoder.py --dset Planaria --cuda 1 --method UMAP
python decoder.py --dset Planaria --cuda 1 --method ForceAtlas2
```

## Structure of the repository
Folder __datasets__ contains datasets used in the study.

Folder __results__ contains Poincaré map coordinates.

Folder __decoder__ contains weights of the pretrained decoder network.

Folder __predictions__ contains coordinates of sampled (interpolated) points.

Folder __benchmarks__ contains visualization of benchmark embeddings.

## License
PoincareMaps is Attribution-NonCommercial 4.0 International licensed, as found in the LICENSE file.

