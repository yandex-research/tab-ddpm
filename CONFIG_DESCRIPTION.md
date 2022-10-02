# Description of .toml config for TabDDPM
First of all, `train.T` and `eval.T` denote preprocessing for training and for evaluation, respectively.  

Here we list non-obvious parameters.  

Main part:
- `seed = 0` -- evaluation seed (and training, but for training it is fixed to 0)
- `parent_dir = "exp/abalone/check"` -- exp folder
- `real_data_path = "data/abalone/"`
- `model_type = "mlp"` -- model type that approximates the reverse process
- `num_numerical_features ` -- a number of numerical features in dataset
- `device = "cuda:0"`

Model params:
- `is_y_cond` -- false for regression, true for classification
- `d_in` -- input dimension (not necessary, since scripts calculate it automatically)
- `num_calsses` -- zero for regression, a number of classes for classification
- `rtdl_params` -- MLP parameters

```toml
seed = 0
parent_dir = "exp/abalone/check"
real_data_path = "data/abalone/"
model_type = "mlp"
num_numerical_features = 7
device = "cuda:0"

[model_params]
is_y_cond = false
d_in = 11
num_classes = 0

[model_params.rtdl_params]
d_layers = [
    256,
    256,
]
dropout = 0.0

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 1000
lr = 0.001
weight_decay = 1e-05
batch_size = 4096

[train.T]
seed = 0
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = 20800
batch_size = 10000
seed = 0

[eval.type]
eval_model = "catboost"
eval_type = "synthetic"

[eval.T]
seed = 0
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

```