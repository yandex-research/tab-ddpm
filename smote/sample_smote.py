import os
import lib
import argparse
import numpy as np
from pathlib import Path
from typing import Union, Any
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

class MySMOTE(SMOTE):
    def __init__(
        self,
        lam1=0.0,
        lam2=1.0,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

        self.lam1=lam1
        self.lam2=lam2

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new
    
class MySMOTENC(SMOTENC):
    def __init__(
        self,
        lam1=0.0,
        lam2=1.0,
        *,
        categorical_features,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None
    ):
        super().__init__(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

        self.lam1=0.0
        self.lam2=1.0

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, lam1=0.0, lam2=1.0
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

def save_data(X, y, path, n_cat_features=0):
    if n_cat_features > 0:
        X_num = X[:, :-n_cat_features]
        X_cat = X[:, -n_cat_features:]
    else:
        X_num = X
        X_cat = None

    
    np.save(path / "X_num_train", X_num.astype(float), allow_pickle=True)
    np.save(path / "y_train", y, allow_pickle=True)
    if X_cat is not None:
        np.save(path / "X_cat_train", X_cat, allow_pickle=True)

def sample_smote(
    parent_dir,
    real_data_path,
    eval_type = "synthetic",
    k_neighbours = 5,
    frac_samples = 1.0,
    frac_lam_del = 0.0,
    change_val = False,
    save = True,
    seed = 0
):
    lam1 = 0.0 + frac_lam_del / 2
    lam2 = 1.0 - frac_lam_del / 2

    real_data_path = Path(real_data_path)
    info = lib.load_json(real_data_path / 'info.json')
    is_regression = info['task_type'] == 'regression'

    X_num = {}
    X_cat = {}
    y = {}

    if change_val:
        X_num['train'], X_cat['train'], y['train'], X_num['val'], X_cat['val'], y['val'] = lib.read_changed_val(real_data_path)
    else:
        X_num['train'], X_cat['train'], y['train'] = lib.read_pure_data(real_data_path, 'train')
        X_num['val'], X_cat['val'], y['val'] = lib.read_pure_data(real_data_path, 'val')
    X_num['test'], X_cat['test'], y['test'] = lib.read_pure_data(real_data_path, 'test')


    X = {k: X_num[k] for k in X_num.keys()}

    if is_regression:
        X['train'] = np.concatenate([X["train"], y["train"].reshape(-1, 1)], axis=1, dtype=object)
        y['train'] = np.where(y["train"] > np.median(y["train"]), 1, 0)
    
    n_num_features = X['train'].shape[1]
    n_cat_features = X_cat['train'].shape[1] if X_cat['train'] is not None else 0
    cat_features = list(range(n_num_features, n_num_features+n_cat_features))
    print(cat_features)

    scaler = MinMaxScaler().fit(X["train"])
    X["train"] = scaler.transform(X["train"]).astype(object)

    if X_cat['train'] is not None:
        for k in X_num.keys():
            X[k] = np.concatenate([X[k], X_cat[k]], axis=1, dtype=object)

    print("Before:", X['train'].shape)

    if eval_type != 'real':
        strat = {k: int((1 + frac_samples) * np.sum(y['train'] == k)) for k in np.unique(y['train'])}
        print(strat)
        if n_cat_features > 0:
            sm = MySMOTENC(
                lam1=lam1,
                lam2=lam2,
                random_state=seed,
                k_neighbors=k_neighbours,
                categorical_features=cat_features,
                sampling_strategy=strat
            )
        else:
            sm = MySMOTE(
                lam1=lam1,
                lam2=lam2,
                random_state=seed,
                k_neighbors=k_neighbours,
                sampling_strategy=strat
            )

        X_res, y_res = sm.fit_resample(X['train'], y['train'])
        if is_regression:
            X_res[:, :X_num["train"].shape[1]+1] = scaler.inverse_transform(X_res[:, :X_num["train"].shape[1]+1])
            y_res = X_res[:, X_num["train"].shape[1]]
            X_res = np.delete(X_res, [X_num["train"].shape[1]], axis=1)
        else:
            X_res[:, :X_num["train"].shape[1]] = scaler.inverse_transform(X_res[:, :X_num["train"].shape[1]])
            y_res = y_res.astype(int)

        if eval_type == "synthetic":
            X_res = X_res[X['train'].shape[0]:]
            y_res = y_res[X['train'].shape[0]:]
        
    disc_cols = []
    for col in range(X_num["train"].shape[1]):
        uniq_vals = np.unique(X_num["train"][:, col])
        if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
            disc_cols.append(col)
    if len(disc_cols):
        X_res[:, :X_num["train"].shape[1]] = lib.round_columns(X_num["train"], X_res[:, :X_num["train"].shape[1]], disc_cols)
    
    if save:
        save_data(X_res, y_res, Path(parent_dir), n_cat_features)

    X['train'] = X_res
    y['train'] = y_res

    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',  type=str)
    parser.add_argument('method', type=str)

    args = parser.parse_args()

    sample_smote(args.data_path, args.method, save=False)

if __name__ == '__main__':
    main()