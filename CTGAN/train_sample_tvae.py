import lib
import os
import numpy as np
import argparse
from CTGAN.ctgan import TVAESynthesizer
from pathlib import Path
import torch
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_tvae(
    parent_dir,
    real_data_path,
    train_params = {"batch_size": 512},
    change_val=False,
    device = "cpu"
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    device = torch.device(device)

    if change_val:
        X_num_train, X_cat_train, y_train, _, _, _ = lib.read_changed_val(real_data_path)
    else:
        X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, 'train')
    
    X = lib.concat_to_pd(X_num_train, X_cat_train, y_train)

    X.columns = [str(_) for _ in X.columns]
    
    cat_features = list(map(str, range(X_num_train.shape[1], X_num_train.shape[1]+X_cat_train.shape[1]))) if X_cat_train is not None else []
    if lib.load_json(real_data_path / "info.json")["task_type"] != "regression":
        cat_features += ["y"]

    train_params["batch_size"] = min(y_train.shape[0], train_params["batch_size"])

    print(train_params)
    synthesizer =  TVAESynthesizer( 
                    **train_params,
                    device=device
                ) 
    
    synthesizer.fit(X, cat_features)

    # save_ctabgan(synthesizer, parent_dir)
    with open(parent_dir / "tvae.obj", "wb") as f:
        pickle.dump(synthesizer, f)

    return synthesizer

def sample_tvae(
    synthesizer,
    parent_dir,
    real_data_path,
    num_samples,
    train_params = {"batch_size": 512},
    change_val=False,
    device="cpu",
    seed=0
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    device = torch.device(device)

    if change_val:
        X_num_train, X_cat_train, y_train, _, _, _ = lib.read_changed_val(real_data_path)
    else:
        X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, 'train')
    
    X = lib.concat_to_pd(X_num_train, X_cat_train, y_train)

    X.columns = [str(_) for _ in X.columns]


    cat_features = list(map(str, range(X_num_train.shape[1], X_num_train.shape[1]+X_cat_train.shape[1]))) if X_cat_train is not None else []
    if lib.load_json(real_data_path / "info.json")["task_type"] != "regression":
        cat_features += ["y"]

    with open(parent_dir / "tvae.obj", 'rb')  as f:
        synthesizer = pickle.load(f)
        synthesizer.decoder = synthesizer.decoder.to(device)

    gen_data = synthesizer.sample(num_samples, seed)

    y = gen_data['y'].values
    if len(np.unique(y)) == 1:
        y[0] = 0
        y[1] = 1

    X_cat = gen_data[cat_features].drop('y', axis=1, errors="ignore").values if len(cat_features) else None
    X_num = gen_data.values[:, :X_num_train.shape[1]] if X_num_train is not None else None

    if X_num_train is not None:
        np.save(parent_dir / 'X_num_train', X_num.astype(float))
    if X_cat_train is not None:
        np.save(parent_dir / 'X_cat_train', X_cat.astype(str))
    y = y.astype(float)
    if lib.load_json(real_data_path / "info.json")["task_type"] != "regression":
        y = y.astype(int)
    np.save(parent_dir / 'y_train', y) # only clf !!!

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('real_data_path', type=str)
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('train_size', type=int)
    args = parser.parse_args()

    ctabgan = train_tvae(args.parent_dir, args.real_data_path, change_val=True)
    sample_tvae(ctabgan, args.parent_dir, args.real_data_path, args.train_size, change_val=True)


if __name__ == '__main__':
    main()