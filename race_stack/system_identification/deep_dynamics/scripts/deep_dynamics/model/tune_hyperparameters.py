import yaml
import os
from functools import partial
import torch
import numpy as np
from deep_dynamics.model.models import string_to_dataset, string_to_model
from deep_dynamics.model.train import train
from ray import tune
import pickle
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

import pandas as pd


def main(model_cfg, log_wandb):

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    config = {
        "layers" : tune.choice(range(1,9)),
        "neurons" : tune.randint(4, 256),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "lr" : tune.uniform(1e-4, 1e-2),
        "horizon": tune.choice(range(1,10)),
        "gru_layers": tune.choice(range(9))
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=400,
        grace_period=100,
    )

    # result = tune.run(
    #     partial(tune_hyperparams, model_cfg=model_cfg, log_wandb=log_wandb),
    #     metric='loss',
    #     mode='min',
    #     search_alg=OptunaSearch(),
    #     resources_per_trial={"cpu": 1, "gpu": 1/9},
    #     config=config,
    #     num_samples=200,
    #     scheduler=scheduler,
    #     storage_path="/home/misys/project_ddm/deep-dynamics/deep_dynamics/output/ray_results",
    #     stop={"training_iteration": 400}
    #     # checkpoint_at_end=True
    # )
    analysis = tune.run(
        partial(tune_hyperparams, model_cfg=model_cfg, log_wandb=log_wandb),
        metric='loss',
        mode='min',
        search_alg=OptunaSearch(),
        resources_per_trial={"cpu": 1, "gpu": 1/9},
        config=config,
        num_samples= 100, # 종료 조건 : 200번의 실험을 모두 한 후에 종료 / 200->100
        scheduler=scheduler,
        storage_path="/home/misys/project_ddm/ray_results",  # 예시
        stop={"training_iteration": 400}
    )

    df = analysis.results_df
    print(df.sort_values("loss").head(10))  # 가장 낮은 loss 상위 10개 보기



def tune_hyperparams(hyperparam_config, model_cfg, log_wandb):
    # dataset_file = "/u/jlc9wr/deep-dynamics/deep_dynamics/data/Putnam_park2023_run4_2_{}.npz".format(hyperparam_config["horizon"])
    dataset_file = "/home/misys/shared_dir/npz/1022_test_{}.npz".format(hyperparam_config["horizon"])

    with open(model_cfg, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)

    experiment_name = "%dlayers_%dneurons_%dbatch_%flr_%dhorizon_%dgru" % (
        hyperparam_config["layers"], hyperparam_config["neurons"],
        hyperparam_config["batch_size"], hyperparam_config["lr"],
        hyperparam_config["horizon"], hyperparam_config["gru_layers"]
    )


    output_dir = "/home/misys/project_ddm/deep-dynamics/deep_dynamics/output/tune_hype/%s/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0], experiment_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    data_npy = np.load(dataset_file)
    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"])
    train_dataset, val_dataset = dataset.split(0.8)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparam_config["batch_size"], shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparam_config["batch_size"], shuffle=False)

    param_dict["MODEL"]["LAYERS"] = []
    if hyperparam_config["gru_layers"]:
        layer = dict()
        layer["GRU"] = None
        layer["OUT_FEATURES"] = hyperparam_config["horizon"] ** 2
        layer["LAYERS"] = hyperparam_config["gru_layers"]
        param_dict["MODEL"]["LAYERS"].append(layer)

    for i in range(hyperparam_config["layers"]):
        layer = dict()
        layer["DENSE"] = None
        layer["OUT_FEATURES"] = hyperparam_config["neurons"]
        layer["ACTIVATION"] = "Mish"
        param_dict["MODEL"]["LAYERS"].append(layer)

    param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = hyperparam_config["batch_size"]
    param_dict["MODEL"]["OPTIMIZATION"]["LR"] = hyperparam_config["lr"]
    param_dict["MODEL"]["HORIZON"] = hyperparam_config["horizon"]

    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(dataset.scaler, f)

    train(model, train_data_loader, val_data_loader, experiment_name, log_wandb, output_dir, os.path.basename(os.path.normpath(model_cfg)).split('.')[0] + "_1022", use_ray_tune=True)


if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("--log_wandb", action='store_true', default=False, help="Log experiment in wandb")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    main(argdict["model_cfg"], argdict["log_wandb"])