import numpy as np
import os
from sklearn.utils import shuffle
import zero
from pathlib import Path
import lib
from lib import concat_features, read_pure_data, read_changed_val
from sklearn.utils import shuffle
import lib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

def train_simple(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    model_name = "tree",
    seed = 0,
    change_val = True,
    params = None, # dummy
    device = None # dummy
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)

    T_dict["normalization"] = "minmax"
    T_dict["cat_encoding"] = None
    T = lib.Transformations(**T_dict)
    info = lib.load_json(os.path.join(real_data_path, 'info.json'))
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)

        ###
        # dists = privacy_metrics(real_data_path, synthetic_data_path)
        # bad_fakes = dists.argsort()[:int(0.25 * len(y_fake))]
        # X_num_fake = np.delete(X_num_fake, bad_fakes, axis=0)
        # X_cat_fake = np.delete(X_cat_fake, bad_fakes, axis=0) if X_cat_fake is not None else None
        # y_fake = np.delete(y_fake, bad_fakes, axis=0)
        ###

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {parent_dir}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path)

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path)
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')
    
    D = lib.Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        lib.TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = lib.transform_dataset(D, T, None)
    X = concat_features(D)
    # ixs = np.random.choice(len(D.y["train"]), min(info["train_size"], len(D.y["train"])), replace=False)
    # X["train"] = X["train"].iloc[ixs]
    # D.y["train"] = D.y["train"][ixs]

    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')
    print(T_dict)
    print('-'*100)
    
    if D.is_regression:
        models = {
            "tree": DecisionTreeRegressor(max_depth=28, random_state=seed),
            "rf": RandomForestRegressor(max_depth=28, random_state=seed),
            "lr": Ridge(max_iter=500, random_state=seed),
            "mlp": MLPRegressor(max_iter=100, random_state=seed)
        }
    else:
        models = {
            "tree": DecisionTreeClassifier(max_depth=28, random_state=seed),
            "rf": RandomForestClassifier(max_depth=28, random_state=seed),
            "lr": LogisticRegression(max_iter=500, n_jobs=2, random_state=seed),
            "mlp": MLPClassifier(max_iter=100, random_state=seed)
        }
    
    model = models[model_name]

    predict = (
        model.predict
        if D.is_regression
        else model.predict_proba
        if D.is_multiclass
        else lambda x: model.predict_proba(x)[:, 1]
    )

    model.fit(X['train'], D.y['train'])

    predictions = {k: predict(v) for k, v in X.items()}

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = lib.MetricsReport(report['metrics'], D.task_type)
    print(model.__class__.__name__)
    metrics_report.print_metrics()
    
    # if parent_dir is not None:
        # lib.dump_json(report, os.path.join(parent_dir, "results_catboost.json"))

    return metrics_report

    