import tomli
import shutil
import os
import argparse
from train_sample_tvae import train_tvae, sample_tvae
from scripts.eval_catboost import train_catboost
import zero
import lib

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true',  default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    ctabgan = None
    if args.train:
        ctabgan = train_tvae(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            train_params=raw_config['train_params'],
            change_val=args.change_val,
            device=raw_config['device']
        )
    if args.sample:
        sample_tvae(
            synthesizer=ctabgan,
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            num_samples=raw_config['sample']['num_samples'],
            train_params=raw_config['train_params'],
            change_val=args.change_val,
            seed=raw_config['sample']['seed'],
            device=raw_config['device']
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        # elif raw_config['eval']['type']['eval_model'] == 'mlp':
        #     train_mlp(
        #         parent_dir=raw_config['parent_dir'],
        #         real_data_path=raw_config['real_data_path'],
        #         eval_type=raw_config['eval']['type']['eval_type'],
        #         T_dict=raw_config['eval']['T'],
        #         seed=raw_config['seed'],
        #         change_val=args.change_val
        #     )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()