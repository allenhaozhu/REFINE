# RBQR

## Prerequisites

- Linux or macOS
- Python 2 or 3
- scipy
- sklearn

## Dataset

These datasets are public datasets.

- PPI contains 3,890 nodes, 76,584 edges and 60 labels.
- Wikipedia contains 4,777 nodes, 184,812 edges and 40 labels.
- Blogcatalog contains 10,312 nodes, 333,983 edges and 39 labels.
- flickr contains 80,513 nodes, 5,899,882 edges and 195 labels. 
- Youtube contains 1,138,499 nodes, 2,990,443 edges and 47 labels.

## Training

### Training on the existing datasets

Create emb directory to save output embedding file
```bash
mkdir emb
```
You can use `python RBQR_NE.py -graph example_graph` to train RBQR model on the example data.

If you want to train on the dataset, you can run 

```bash
python RBQR_NE.py -graph data/PPI.mat -emb1 emb/PPI_sparse.emb -emb2 emb/PPI_spectral.emb -dimension 128
python RBQR_NE.py -graph data/POS.mat -emb1 emb/POS_sparse.emb -emb2 emb/POS_spectral.emb -dimension 128
python RBQR_NE.py -graph data/blogcatalog.mat -emb1 emb/blogcatalog_sparse.emb -emb2 emb/blogcatalog_spectral.emb -dimension 128
python RBQR_NE.py -graph data/flickr.mat -emb1 emb/flickr_sparse.emb -emb2 emb/flickr_spectral.emb -dimension 128
python RBQR_NE.py -graph data/youtube.mat -emb1 emb/youtube_sparse.emb -emb2 emb/youtube_spectral.emb -dimension 128
```
Where PPI_sparse.emb and PPI_spectral.emb are output embedding files and dimension=128 is the default setting for a good result.
If you want to evaluate the embedding without spectral filter via node classification task, you can run

```bash
python predict.py --label data/blogcatalog.mat --embedding emb/blogcatalog_sparse.emb.npy --start-train-ratio 10 --stop-train-ratio 90 --seed 0
python predict.py --label data/POS.mat --embedding emb/POS_sparse.emb.npy --start-train-ratio 10 --stop-train-ratio 90 --seed 0
python predict.py --label data/PPI.mat --embedding emb/PPI_sparse.emb.npy --start-train-ratio 10 --stop-train-ratio 90 --seed 0
python predict.py --label data/flickr.mat --embedding emb/flickr_sparse.emb.npy --start-train-ratio 1 --stop-train-ratio 9 --seed 0
python predict.py --label data/youtube.mat --embedding emb/youtube_sparse.emb.npy --start-train-ratio 1 --stop-train-ratio 9 --seed 0
```

If you want to evaluate the embedding with spectral filter via node classification task, you can run

```bash
python predict.py --label data/blogcatalog.mat --embedding emb/blogcatalog_spectral.emb.npy --start-train-ratio 10 --stop-train-ratio 90 --seed 0
python predict.py --label data/POS.mat --embedding emb/POS_spectral.emb.npy --start-train-ratio 10 --stop-train-ratio 90 --seed 0
python predict.py --label data/PPI.mat --embedding emb/PPI_spectral.emb.npy --start-train-ratio 10 --stop-train-ratio 90 --seed 0
python predict.py --label data/flickr.mat --embedding emb/flickr_spectral.emb.npy --start-train-ratio 1 --stop-train-ratio 9 --seed 0
python predict.py --label data/youtube.mat --embedding emb/youtube_spectral.emb.npy --start-train-ratio 1 --stop-train-ratio 9 --seed 0
```
