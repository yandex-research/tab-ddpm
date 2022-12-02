import optuna
import lib
import argparse
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('model', type=str)
parser.add_argument('tune_type', type=str)
parser.add_argument('device', type=str)

args = parser.parse_args()
data_path = Path(f"data/{args.ds_name}")
best_params = None

assert args.tune_type in ("cv", "val")

def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f'suggest_{distribution}')(label, *args)

def _suggest_optional(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    if trial.suggest_categorical(f"optional_{label}", [True, False]):
        return _suggest(trial, distribution, label, *args)
    else:
        return 0.0

def _suggest_mlp_layers(trial: optuna.trial.Trial, mlp_d_layers: list[int]):

    min_n_layers, max_n_layers = mlp_d_layers[0], mlp_d_layers[1]
    d_min, d_max = mlp_d_layers[2], mlp_d_layers[3]

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t


    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last

    return d_layers

def suggest_mlp_params(trial):
    params = {}
    params["lr"] = trial.suggest_loguniform("lr", 5e-5, 0.005)
    params["dropout"] = _suggest_optional(trial, "uniform", "dropout", 0.0, 0.5)
    params["weight_decay"] = _suggest_optional(trial, "loguniform", "weight_decay", 1e-6, 1e-2)
    params["d_layers"] = _suggest_mlp_layers(trial, [1, 8, 6, 10])

    return params

def suggest_catboost_params(trial):
    params = {}
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.001, 1.0)
    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["l2_leaf_reg"] = trial.suggest_uniform("l2_leaf_reg", 0.1, 10.0)
    params["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0.0, 1.0)
    params["leaf_estimation_iterations"] = trial.suggest_int("leaf_estimation_iterations", 1, 10)

    params = params | {
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU", # "GPU", may affect performance
        "thread_count": 4,
        # "devices": "0", # for GPU
    }

    return params

def objective(trial):
    if args.model == "mlp":
        params = suggest_mlp_params(trial)
        train_func = train_mlp
        T_dict = {
            "seed": 0,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": "one-hot",
            "y_policy": "default"
        }
    else:
        params = suggest_catboost_params(trial)
        train_func = train_catboost
        T_dict = {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        }
    trial.set_user_attr("params", params)
    if args.tune_type == "cv":
        score = 0.0
        for fold in range(5):
            metrics_report = train_func(
                parent_dir=None,
                real_data_path=data_path / f"kfolds/{fold}",
                eval_type="real",
                T_dict=T_dict,
                params=params,
                change_val=False,
                device=args.device
            )
            score += metrics_report.get_val_score()
        score /= 5

    elif args.tune_type == "val":
        metrics_report = train_func(
            parent_dir=None,
            real_data_path=data_path,
            eval_type="real",
            T_dict=T_dict,
            params=params,
            change_val=False,
            device=args.device
        )
        score = metrics_report.get_val_score()
    
    return score

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=100, show_progress_bar=True)
    
bets_params = study.best_trial.user_attrs['params']

best_params_path = f"tuned_models/{args.model}/{args.ds_name}_{args.tune_type}.json"

lib.dump_json(bets_params, best_params_path)