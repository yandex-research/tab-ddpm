"""
Adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval
"""

import argparse
import lib
import os
import shutil
import zero
from sample import sample
from smote.sample_smote import sample_smote
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import pairwise_distances
from pathlib import Path
import tempfile
from eval_seeds import eval_seeds
import numpy as np
import subprocess
import warnings
import torch

zero.improve_reproducibility(0)

warnings.filterwarnings("ignore", category=FutureWarning)


def privacy_metrics(real_path,fake_path, data_percent=15):

    """
    Returns privacy metrics
    
    Inputs:
    1) real_path -> path to real data
    2) fake_path -> path to corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
    
    """
    task_type = lib.load_json(real_path + "/info.json")["task_type"]
    X_num_real, X_cat_real, y_real = lib.read_pure_data(real_path, 'train')
    X_num_fake, X_cat_fake, y_fake = lib.read_pure_data(fake_path, 'train')

    if task_type == 'regression':
        X_num_real = np.concatenate([X_num_real, y_real[:, np.newaxis]], axis=1)
        X_num_fake = np.concatenate([X_num_fake, y_fake[:, np.newaxis]], axis=1)
    else:
        if X_cat_fake is None:
            X_cat_real = y_real[:, np.newaxis].astype(int).astype(str)
            X_cat_fake = y_fake[:, np.newaxis].astype(int).astype(str)
        else:
            X_cat_real = np.concatenate([X_cat_real, y_real[:, np.newaxis].astype(int).astype(str)], axis=1)
            X_cat_fake = np.concatenate([X_cat_fake, y_fake[:, np.newaxis].astype(int).astype(str)], axis=1)

    if len(y_real) > 50000:
        ixs = np.random.choice(len(y_real), 50000, replace=False)
        X_num_real = X_num_real[ixs]
        X_cat_real = X_cat_real[ixs] if X_cat_real is not None else None
    
    if len(y_fake) > 50000:
        ixs = np.random.choice(len(y_fake), 50000, replace=False)
        X_num_fake = X_num_fake[ixs]
        X_cat_fake = X_cat_fake[ixs] if X_cat_fake is not None else None


    mm = MinMaxScaler().fit(X_num_real)
    X_real = mm.transform(X_num_real)
    X_fake = mm.transform(X_num_fake)
    if X_cat_real is not None:
        ohe = OneHotEncoder().fit(X_cat_real)
        X_cat_real = ohe.transform(X_cat_real) / np.sqrt(2)
        X_cat_fake = ohe.transform(X_cat_fake) / np.sqrt(2)

        X_real = np.concatenate([X_real, X_cat_real.todense()], axis=1)
        X_fake = np.concatenate([X_fake, X_cat_fake.todense()], axis=1)

    # X_real = np.unique(X_real, axis=0)
    # X_fake = np.unique(X_fake, axis=0)

    # Computing pair-wise distances between real and synthetic 
    dist_rf = pairwise_distances(X_fake, Y=X_real, metric='l2', n_jobs=-1)
    # Computing pair-wise distances within real 
    # dist_rr = pairwise_distances(X_real, Y=None, metric='l2', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    # dist_ff = pairwise_distances(X_fake, Y=None, metric='l2', n_jobs=-1)

    
    # Removes distances of data points to themselves to avoid 0s within real and synthetic 
    # rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    # rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    # Computing first and second smallest nearest neighbour distances within real
    # smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    # smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    # smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    # smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    

    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    # min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    # fifth_perc_rr = np.percentile(min_dist_rr,5)
    # min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    # fifth_perc_ff = np.percentile(min_dist_ff,5)
    # nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    # nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
    # nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    # nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    # nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    # nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
        
    # return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6) 
    return min_dist_rf # , min_dist_rr

def sample_wrapper(method, config, num_samples=None, seed=0):
    if method == "ddpm":
        sample(
            num_samples=num_samples,
            batch_size=config['sample']['batch_size'],
            disbalance=config['sample'].get('disbalance', None),
            **config['diffusion_params'],
            parent_dir=config['parent_dir'],
            real_data_path=config['real_data_path'],
            model_path=os.path.join(config['parent_dir'], 'model.pt'),
            model_type=config['model_type'],
            model_params=config['model_params'],
            T_dict=config['train']['T'],
            num_numerical_features=config['num_numerical_features'],
            seed=seed,
            change_val=False,
            device=torch.device(config["device"])
        )
    elif method == "smote":
        sample_smote(
            parent_dir=config['parent_dir'],
            real_data_path=config['real_data_path'],
            **config['smote_params'],
            seed=seed,
            change_val=False
        )

def resample_privacy(config_path, method, q):
    with tempfile.TemporaryDirectory() as dir_:
        config = lib.load_config(config_path)
        if method == "ddpm":
            shutil.copy2(os.path.join(config['parent_dir'], 'model.pt'), os.path.join(dir_, 'model.pt'))
        config["parent_dir"] = str(dir_)
        parent_dir = config["parent_dir"]

        sample_wrapper(method, config, num_samples=config["sample"].get("num_samples", 0))

        dists = privacy_metrics(config["real_data_path"], parent_dir)
        old_privacy = np.median(dists)

        q10 = np.quantile(dists, q=q)
        print(f"Q: {q10}")
        to_drop = np.where(dists < q10)

        X_num, X_cat, y = lib.read_pure_data(parent_dir)
        num_samples = len(y)
        X_num = np.delete(X_num, to_drop, axis=0)
        X_cat = np.delete(X_cat, to_drop, axis=0) if X_cat is not None else None
        y = np.delete(y, to_drop, axis=0)
        i = 1

        while len(y) < num_samples and i <= 10:
            print(f"{len(y)}/{num_samples}")
            
            sample_wrapper(method, config, num_samples=config["sample"].get("batch_size", 0), seed=i)

            i += 1

            X_num_t, X_cat_t, y_t = lib.read_pure_data(parent_dir)
            dists = privacy_metrics(config["real_data_path"], parent_dir)
            to_drop = np.where(dists < q10)
            X_num_t = np.delete(X_num_t, to_drop, axis=0)
            X_cat_t = np.delete(X_cat_t, to_drop, axis=0) if X_cat is not None else None
            y_t = np.delete(y_t, to_drop, axis=0)

            X_num = np.concatenate([X_num, X_num_t], axis=0)[:num_samples]
            X_cat = np.concatenate([X_cat, X_cat_t], axis=0)[:num_samples] if X_cat is not None else None
            y = np.concatenate([y, y_t], axis=0)[:num_samples]

            # np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
            # if X_cat is not None:
            #     np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
            # np.save(os.path.join(parent_dir, 'y_train'), y)

        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
        if X_cat is not None:
            np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
        np.save(os.path.join(parent_dir, 'y_train'), y)

        new_dists = privacy_metrics(config["real_data_path"], parent_dir)

        res = eval_seeds(
            config,
            n_seeds=10,
            eval_type="synthetic",
            model_type="catboost",
            n_datasets=1,
            dump=False
        )
        print(f"Old: {old_privacy:.4f}, New: {np.median(new_dists):.4f}")

    metric = "r2-mean" if "r2-mean" in res["test"] else "f1-mean"
    return res["test"][metric], np.around(np.median(new_dists), 4)

def resample_privacy_qs(config_path, method):
    config = lib.load_config(config_path)
    scores = []
    privacies = []

    eval_res = lib.load_json(Path(config["parent_dir"]) / "eval_catboost.json")["synthetic"]["test"]
    metric = "r2-mean" if "r2-mean" in eval_res else "f1-mean"
    scores.append(eval_res[metric])
    privacies.append(np.median(privacy_metrics(config["real_data_path"], config["parent_dir"])))

    for q in [0.1, 0.2, 0.3, 0.4]:
        score, privacy = resample_privacy(config_path, method, q)
        scores.append(score)
        privacies.append(privacy)
    
    lib.dump_json(
        {"scores": scores, "privacies": privacies},
        Path(config["parent_dir"]) / "privacies.json"
    )

def calc_privacy(config_path, method, seed=0):
    config = lib.load_config(config_path)
    sample_wrapper(method, config, num_samples=config["sample"]["num_samples"], seed=seed)
    timer = zero.Timer()
    timer.run()
    dists = privacy_metrics(config["real_data_path"], config["parent_dir"])
    privacy_val = np.median(dists)
    lib.dump_json({"privacy": privacy_val}, os.path.join(config["parent_dir"], "privacy.json"))
    print(f"Elapsed tine:{str(timer)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('method', type=str)
    args = parser.parse_args()

    calc_privacy(
        args.config,
        args.method
    )

if __name__ == "__main__":
    main()