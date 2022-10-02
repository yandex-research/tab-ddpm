import lib
import os
import numpy as np
import argparse
from pathlib import Path
from model.ctabgan import CTABGAN
import torch
import pickle

def train_ctabgan(
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

    ctabgan_params = lib.load_json("CTAB-GAN/columns.json")[real_data_path.name]
    train_params["batch_size"] = min(y_train.shape[0], train_params["batch_size"])

    print(train_params)
    synthesizer =  CTABGAN(
                    df = X,
                    test_ratio = 0.0,  
                    **ctabgan_params,
                    **train_params,
                    device=device
                ) 
    
    synthesizer.fit()

    # save_ctabgan(synthesizer, parent_dir)
    with open(parent_dir / "ctabgan.obj", "wb") as f:
        pickle.dump(synthesizer, f)

    return synthesizer

def sample_ctabgan(
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

    ctabgan_params = lib.load_json("CTAB-GAN/columns.json")[real_data_path.name]

    cat_features = ctabgan_params["categorical_columns"]
    # if synthesizer is None:
        # synthesizer = load_ctabgan(X, ctabgan_params, train_params, parent_dir)
    with open(parent_dir / "ctabgan.obj", 'rb')  as f:
        synthesizer = pickle.load(f)
        synthesizer.synthesizer.generator = synthesizer.synthesizer.generator.to(device)
    gen_data = synthesizer.generate_samples(num_samples, seed)

    y = gen_data['y'].values
    if len(np.unique(y)) == 1:
        y[0] = 1

    X_cat = gen_data[cat_features].drop('y', axis=1).values if len(cat_features) else None
    X_num = gen_data.values[:, :X_num_train.shape[1]] if X_num_train is not None else None

    if X_num_train is not None:
        np.save(parent_dir / 'X_num_train', X_num.astype(float))
    if X_cat_train is not None:
        np.save(parent_dir / 'X_cat_train', X_cat.astype(str))
    np.save(parent_dir / 'y_train', y.astype(float).astype(int)) # only clf !!!

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('real_data_path', type=str)
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('train_size', type=int)
    args = parser.parse_args()

    ctabgan = train_ctabgan(args.parent_dir, args.real_data_path, change_val=True)
    sample_ctabgan(ctabgan, args.parent_dir, args.real_data_path, args.train_size, change_val=True)


if __name__ == '__main__':
    main()