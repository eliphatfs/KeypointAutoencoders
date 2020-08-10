# Keypoint Autoencoders: Learning Interest Points of Semantics.
This is the repository that contains the code of the paper `Keypoint Autoencoders: Learning Interest Points of Semantics`.

## Training
To run the training process:
```python
import acae  # Choose one from `acae` and `vnapf`
acae.train()  # Trains the model
acae.visual_test(True)  # Picks a point cloud in the test set and visualize the results
```

## Data Preparation
The data should be stored in the `./point_cloud/train` and `./point_cloud/test`.

All `.h5` files under those folders are loaded. Each file should contain a `data` array of shape `(n, m, 3)` which is `n` point clouds with `m` points, and a `label` array of shape `(n)` which indicates the classes of the point cloud. `m` is required to be same for all point clouds.
The paper uses point clouds generated from the `ModelNet40` dataset, which can be downloaded [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).
