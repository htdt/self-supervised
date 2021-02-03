`w_mse_loss()` from `whitening.py` is W-MSE loss implementation for TensorFlow 2,
it can be used with other popular implementations, e.g. [SimCLRv2](https://github.com/google-research/simclr/tree/master/tf2).


Method uses global flags mechanism as in SimCLRv2:
- `FLAGS.num_samples` - number of samples (d) generated from each image
- `FLAGS.train_batch_size`
- `FLAGS.proj_out_dim`
