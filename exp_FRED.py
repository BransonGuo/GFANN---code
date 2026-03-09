import os
import joblib
from colorama import init, Fore
import torch
import random
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import DataLoader
from scipy.sparse.linalg import eigsh as largest_eigsh
from sklearn.decomposition import PCA
import multiprocessing
import functools
import argparse
import time
from datetime import datetime
import pandas as pd
import pathlib
import copy
import matplotlib as mpl
from typing import Any, Callable, Dict
from methods.stat_methods_soft import *
import methods.stat_methods as stat_methods
from models.model_lib_soft_PCA import *
from config import *
from models.fast_nn import FactorAugmentedSparseThroughputNN
from models.far_nn import RegressionNN
from models.far_nn import FactorAugmentedNN
import sys

sys.path.append(WORKPATH)
from logs import log, create_handler
from utils.utils import *
from data.fast_data_standardized import RegressionDataset

# TODO Re-run exp1

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples", type=int, default=500)
parser.add_argument(
    "--m",
    help="number of samples to calculate the diversified " "projection matrix",
    type=int,
    default=256,
)
parser.add_argument("--p", help="data dimension", type=int, default=1000)
parser.add_argument("--r", help="factor dimension", type=int, default=5)
parser.add_argument(
    "--r_bar", help="diversified weight dimension", type=int, default=10
)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=3)
parser.add_argument("--add_width", help="width of add", type=int, default=10)
parser.add_argument("--add_depth", help="depth of add", type=int, default=2)
parser.add_argument("--nn_depth", help="nn_depth", type=int, default=-1)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=150)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-2)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
parser.add_argument("--exp_id", help="exp id", type=int, default=1)
parser.add_argument(
    "--record_dir", help="directory to save record", type=str, default=""
)
parser.add_argument("--log_dir", help="directory to save log", type=str, default="")
parser.add_argument("--suffix", help="suffix of the log file", type=str, default="")
parser.add_argument("--memo", help="memo describing the log file", type=str, default="")
parser.add_argument("--noise", help="noise level", type=float, default=1)
parser.add_argument("--b_f", help="factor bound", type=float, default=1)
parser.add_argument("--b_u", help="factor noise bound", type=float, default=1)
parser.add_argument("--num_epoch", help="num_epoch", type=int, default=200)
parser.add_argument("--factor_id", help="factor_id", type=int, default=200)
parser.add_argument("--hcm_id", help="hcm_id", type=int, default=200)
parser.add_argument("--n_trials", help="n_trials", type=int, default=50)
parser.add_argument(
    "--summary_file", help="summary_file", type=str, default="summary_file"
)
parser.add_argument(
    "--linear",
    help="linear factor model",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--reformat_res",
    help="reformat csv res",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--use_scheduler_step",
    help="whether to call scheduler.step() in NN optimization loops",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--use_proj_mean",
    help="take average of the projection matrix",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--record_proj",
    help="record projection matrix",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--x_func_l", help="x_func_l", type=str, default="[0,1,0,1,0,1,0,1,0,1]"
)
parser.add_argument(
    "--y_func_l", help="y_func_l", type=str, default="[0, 1,2,3,4,5,6,7]"
)
parser.add_argument("--reg_lambda", help="reg_lambda", type=float, default=0)
parser.add_argument("--train_window", help="train_window", type=int, default=252)
parser.add_argument("--valid_window", help="valid_window", type=int, default=40)
parser.add_argument("--test_window", help="test_window", type=int, default=252)
parser.add_argument("--start_test_day", help="start_test_day", type=int, default=450)
parser.add_argument("--data_file", help="data_file", type=str, default="")
parser.add_argument("--fred_idx", help="fred_idx", type=int, default=87)
args = parser.parse_args()
######overwrite session#######
# args.reg_lambda = 40
args.loss_type = "var"  # var; sqr_norm; reconstruction
# args.lambda_orthogonality=2.2
# args.lambda_pca = 1
# # args.batch_size=250
# args.cov_reg = 0
# args.sv_reg = 0
# args.reg_lambda = 10
args.forecast = False
MAX_DEPTH = 3
args.init_schedule = list(range(100))
args.init_schedule_ori = list(range(1, 50))
args.save_study = False
args.opt = True
# fit_by_epochs means if want to refit the model after optuna, and without validation data
args.retrain = False
args.fit_by_epochs = False
args.analyze = False
args.use_loss = True
# args.n_trials = 150
args.batch_size = args.test_window
args.reformat_res = True
args.m = args.n
# hyper-parameters
args.rolling_train = False
args.delay = False
args.rank = False
args.normalize = True
args.fred_data = True

if args.use_proj_mean:
    args.record_proj = True
start_time = time.time()
if len(args.suffix) == 0:
    args.suffix = f"n{args.n}b{args.batch_size}noi{args.noise}"
suffix = (
    datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f")
    + str(args.seed)
    + str(args.lr)
    + args.suffix
)
suffix = "del/" + suffix
# set a logger file

seed_everything(seed=args.seed)


model_l = [
    # # # benchmark================================
    # 'oracleNNOpt',
    # 'factorAugmentedNNOpt',
    # 'vanillaNNOpt',
    # 'autoencoderOpt',
    # 'lasso',
    # 'pcr',
    # 'pls',
    # 'di',
    # 'arp',
    # # hard================================
    # 'oripCA_NNOpt',
    "oripCA_NN_PCA_ADDOpt",
    # 'oripCA_NN_PCAOpt',
    # 'oripCA_NN_ADD_PCAOpt'
    # #  SOFT================================
    # 'pCA_NNOpt_var',
    "pCA_NN_PCA_ADDOpt",
    # 'nN_PCA_NNOpt',
    # 'pCAA_NNOpt_var',
    # 'pCA_NN_PCAOpt',
    # 'pCA_NN_ADD_PCAOpt',
]

train_window, valid_window, test_window = (
    args.train_window,
    args.valid_window,
    args.test_window,
)
start_test_day = args.start_test_day

assert start_test_day >= train_window + valid_window

from data.fredmd_data import fred_md_data

corpus = fred_md_data(
    "/nfs/home/bransong/projects/FAST/FAST-NN/data/FRED-MD/transformed_modern.csv"
)
idx = args.fred_idx


def get_index_array(l, r):
    index = []
    for i in range(r - l + 1):
        index.append(i + l)
    return index


def split_x_y(data_corpus, idx):
    x = np.concatenate([data_corpus[:, :idx], data_corpus[:, idx + 1 :]], 1)
    y = data_corpus[:, idx]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)
    return x, y


# normalized; and random splits between train and valid
i = 0
window_size = 200
train, valid, test = corpus.get_split_data(
    get_index_array(i, i + window_size - 1),
    get_index_array(i + window_size, corpus.valid_n - 1),
    normalize=False,
)
train_x, train_y = split_x_y(train, idx)
valid_x, valid_y = split_x_y(valid, idx)
test_x, test_y = split_x_y(test, idx)
train_window, valid_window, test_window = (
    train_x.shape[0],
    valid_x.shape[0],
    test_x.shape[0],
)
x_all = np.vstack([train_x, valid_x, test_x])
y_all = np.concatenate([train_y, valid_y, test_y])
dff = pd.DataFrame(x_all)
args.p = x_all.shape[1]

# ===================================================================
if args.rank:
    x_all = pd.DataFrame(x_all).rank(axis=1).values


def merge_dic_to_df(
    best_valid_loss_dic, best_valid_score_dic, test_loss_dic, test_score_dic
):
    best_valid_loss_dic_temp = {f"{k}_valid": v for k, v in best_valid_loss_dic.items()}
    best_valid_score_dic_temp = {
        f"{k}_valid_score": v for k, v in best_valid_score_dic.items()
    }
    test_loss_dic_temp = {f"{k}_test": v for k, v in test_loss_dic.items()}
    test_score_dic_temp = {f"{k}_test_score": v for k, v in test_score_dic.items()}
    summary_dict = dict(
        args.__dict__,
        **best_valid_loss_dic_temp,
        **best_valid_score_dic_temp,
        **test_loss_dic_temp,
        **test_score_dic_temp,
    )
    df = pd.DataFrame([summary_dict])
    return df


def write_summary(
    res_df,
    res_by_score_df,
    test_pred_dic,
    test_pred_by_score_dic,
    pred_all_y,
    model_l,
    verbose=False,
    include_benchmark=True,
):
    output_path = (
        str(pathlib.Path(logger.handlers[1].baseFilename).parent)
        + f"/{args.summary_file}.csv"
    )
    output_path_by_score = output_path[:-4] + "_by_score.csv"
    if not args.fred_data:
        for model in model_l:
            res_temp = results_analytics(pd.DataFrame(test_pred_dic[model]), pred_all_y)
            res_temp.columns = [f"{model}_" + x for x in res_temp.columns]
            res_df = pd.concat([res_df, res_temp], axis=1)

            res_by_score_temp = results_analytics(
                test_pred_by_score_dic[model], pred_all_y
            )
            res_by_score_temp.columns = [
                f"{model}_" + x for x in res_by_score_temp.columns
            ]
            res_by_score_df = pd.concat([res_by_score_df, res_by_score_temp], axis=1)
        if include_benchmark:
            res_temp = results_analytics(pred_all_y - pred_all_y + 1, pred_all_y)
            res_temp.columns = ["buy_hold_" + x for x in res_temp.columns]
            res_df = pd.concat([res_df, res_temp], axis=1)
    if verbose:
        print(res_df.T)
    if not args.reformat_res:
        res_df.to_csv(output_path, index=False, mode="a", header=True)
        res_by_score_df.to_csv(output_path_by_score, index=False, mode="a", header=True)
    else:
        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            res_df = pd.concat([df_existing, res_df])
        if os.path.exists(output_path_by_score):
            df_by_score_existing = pd.read_csv(output_path_by_score)
            res_by_score_df = pd.concat([df_by_score_existing, res_by_score_df])
        res_df.to_csv(output_path, index=False)
        res_by_score_df.to_csv(output_path_by_score, index=False)


def merge_res(res_l):
    metric_l = ["valid_loss", "valid_score", "test_loss", "test_score"]
    res_df, res_by_score_df, test_pred_dic, test_pred_by_score_dic = copy.deepcopy(
        res_l[0]
    )
    for (
        res_df_temp,
        res_by_score_df_temp,
        test_pred_dic_temp,
        test_pred_by_score_dic_temp,
    ) in res_l[1:]:
        res_df.iloc[:, -len(metric_l) * len(model_l) :] = (
            res_df.iloc[:, -len(metric_l) * len(model_l) :]
            + res_df_temp.iloc[:, -len(metric_l) * len(model_l) :]
        )
        res_by_score_df.iloc[:, -len(metric_l) * len(model_l) :] = (
            res_by_score_df.iloc[:, -len(metric_l) * len(model_l) :]
            + res_by_score_df_temp.iloc[:, -len(metric_l) * len(model_l) :]
        )
        for model in model_l:
            test_pred_dic[model] = np.vstack(
                [test_pred_dic[model], test_pred_dic_temp[model]]
            )
            test_pred_by_score_dic[model] = np.vstack(
                [test_pred_by_score_dic[model], test_pred_by_score_dic_temp[model]]
            )
    return res_df, res_by_score_df, test_pred_dic, test_pred_by_score_dic


def record_model_performance(model_name, model_path=None):
    pred = models_dic[model_name].fit_and_predict(
        x_train_obs,
        y_train.squeeze(),
        x_valid_obs,
        y_valid.squeeze(),
        x_test_obs,
        y_test=y_test.squeeze(),
        n_jobs=1,
        study_name=f"{models_dic[model_name].__class__.__name__}_{suffix}",
        retrain=args.retrain,
        fit_by_epochs=args.fit_by_epochs,
    )
    pred = pred.reshape(-1, 1)
    try:
        pred_valid = models_dic[model_name].predict(
            torch.tensor(x_valid_obs, dtype=torch.float32).to(device),
            y_true=y_valid.squeeze(),
            y_history=y_train.squeeze(),
            X_history=torch.tensor(x_train_obs, dtype=torch.float32).to(device),
        )
        valid_loss = mse_loss(
            pred_valid, torch.from_numpy(y_valid.squeeze()).reshape(-1, 1).to(device)
        ).item()
    except:
        try:
            pred_valid = models_dic[model_name].predict(
                x_valid_obs,
                y_true=y_valid.squeeze(),
                y_history=y_train.squeeze(),
                X_history=x_train_obs,
            )
            valid_loss = mse_loss(
                torch.from_numpy(pred_valid).reshape(-1, 1).to(device),
                torch.from_numpy(y_valid.squeeze()).reshape(-1, 1).to(device),
            ).item()
        except:
            valid_loss = models_dic[model_name].global_best_valid_loss
    test_loss = mse_loss(
        torch.from_numpy(pred).reshape(-1, 1),
        torch.from_numpy(y_test.squeeze()).reshape(-1, 1),
    ).item()
    if args.fred_data:
        if torch.is_tensor(pred_valid):
            pred_valid = pred_valid.detach().cpu().numpy()

        tss_this_run = np.mean(np.square(y_valid))
        pred_recon = pred_valid.reshape(-1, 1)
        y_valid_ = y_valid.reshape(-1, 1)
        assert y_valid_.shape == pred_recon.shape
        rss_t = np.mean(np.square(y_valid_ - pred_recon))
        r_sqr = 1 - rss_t / tss_this_run
        valid_score = -r_sqr

        tss_this_run = np.mean(np.square(y_test))
        pred_recon = pred.reshape(-1, 1)
        y_test_ = y_test.reshape(-1, 1)
        assert y_test_.shape == pred_recon.shape
        rss_t = np.mean(np.square(y_test_ - pred_recon))
        r_sqr = 1 - rss_t / tss_this_run
        test_score = -r_sqr

    else:
        valid_score = models_dic[model_name].best_valid_score
        ret_series = calc_ret_series(pred, y_test)
        test_score = -calc_sharpe_ratio(ret_series)
    if args.save_study:
        joblib.dump(
            models_dic[model_name].study,
            f"{model_path}/{models_dic[model_name].__class__.__name__}_study_{suffix}.pkl",
        )
        if (
            torch.nn.modules.module.Module
            in models_dic[model_name].model.__class__.__bases__
        ):
            models_dic[model_name].model.load_state_dict(
                models_dic[model_name].best_state_dict
            )
            # Export the model to ONNX
            onnx_path = f"{model_path}/{models_dic[model_name].__class__.__name__}_{suffix}.onnx"
            torch.onnx.export(
                models_dic[model_name].model,
                torch.from_numpy(x_test_obs[:1]).to(torch.float32).to("cuda"),
                onnx_path,
                verbose=False,
            )
            from onnx2torch import convert
            import onnx

            onnx_model = onnx.load(onnx_path)
            torch_model = convert(onnx_model)
            torch_model.eval()
            pred_load = torch_model(torch.from_numpy(x_test_obs).to(torch.float32))
            test_loss_load = mse_loss(
                pred_load, torch.from_numpy(y_test.squeeze()).reshape(-1, 1)
            )
            # assert round(test_loss.item(), 4) == round(test_loss_load.item(), 4)
    try:
        model_kwargs = copy.deepcopy(models_dic[model_name].best_model_kwargs)
        model_kwargs["best_epoch"] = models_dic[model_name].best_epoch
        model_kwargs["global_best_valid_loss"] = models_dic[
            model_name
        ].global_best_valid_loss
        model_kwargs.update(models_dic[model_name].study.best_params)
    except:
        model_kwargs = {}
    return pred, valid_loss, valid_score, test_loss, test_score, model_kwargs


def joint_train(model_names, logger, parallel=False):
    colors = [
        Fore.RED,
        Fore.YELLOW,
        Fore.BLUE,
        Fore.GREEN,
        Fore.CYAN,
        Fore.LIGHTRED_EX,
        Fore.LIGHTYELLOW_EX,
        Fore.LIGHTBLUE_EX,
        Fore.LIGHTGREEN_EX,
        Fore.LIGHTCYAN_EX,
        Fore.LIGHTMAGENTA_EX,
        Fore.LIGHTRED_EX,
        Fore.LIGHTWHITE_EX,
        Fore.LIGHTBLACK_EX,
        Fore.GREEN,
        Fore.CYAN,
    ] * 2
    model_color = {}
    # best_valid_test_loss_l = {}
    for i, name in enumerate(model_names):
        # best_valid[name] = 1e9
        model_color[name] = colors[i]
        # best_valid_test_loss_l[name] = []
    valid_loss_dic, valid_score_dic = {}, {}
    test_loss_dic, test_score_dic = {}, {}
    best_param_dic = {}
    test_pred_dic = {}
    model_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent)
    for model_name in model_names:
        (
            test_pred_dic[model_name],
            valid_loss_dic[model_name],
            valid_score_dic[model_name],
            test_loss_dic[model_name],
            test_score_dic[model_name],
            best_param_dic[model_name],
        ) = record_model_performance(model_name, model_path=model_path)
    logger.info(
        f"valid_loss_dic {valid_loss_dic}, valid_score_dic {valid_score_dic}, test_loss_dic {test_loss_dic}, test_score_dic {test_score_dic}"
    )
    res_df = merge_dic_to_df(
        valid_loss_dic, valid_score_dic, test_loss_dic, test_score_dic
    )
    return (
        res_df,
        copy.deepcopy(res_df),
        test_pred_dic,
        copy.deepcopy(test_pred_dic),
        best_param_dic,
    )


def write_summary_opt(best_valid, test_perf, best_param_dic):
    output_path = (
        str(pathlib.Path(logger.handlers[1].baseFilename).parent)
        + f"\\{args.summary_file}.csv"
    )
    output_path = output_path.replace("\\", "/")
    summary_dict_l = []
    for k in best_valid.keys():
        valid_test_dic = {
            "model_name": k,
            f"valid_score": best_valid[k],
            f"test_score": test_perf[k],
        }
        args_dic = {f"_{k}": v for k, v in args.__dict__.items()}
        summary_dict_l.append(dict(args_dic, **valid_test_dic, **best_param_dic[k]))
    df = pd.DataFrame(summary_dict_l)
    if not args.reformat_res:
        df.to_csv(output_path, index=False, mode="a", header=True)
    else:
        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            df = pd.concat([df_existing, df])
        df.to_csv(output_path, index=False)


def train_space(trial, test_window):
    return {
        "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
        "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [test_window]),
    }


def train_space_reg(trial, test_window):
    return {
        "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
        "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [test_window]),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 1),
    }


def train_space_reg_var(trial, pcaa=False):
    trial_dict = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
        "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [64]),
        "lambda_orthogonality": trial.suggest_float(
            "lambda_orthogonality", 1e-6, 1, log=True
        ),
        "lambda_pca": trial.suggest_float("lambda_pca", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 1),
    }
    if pcaa:
        trial_dict["lambda_weight"] = trial.suggest_categorical(
            "lambda_weight", [0, 0.001, 0.003]
        )
    return trial_dict


def train_space_sparsity(trial):
    return {
        "lambda_sparsity": trial.suggest_categorical(
            "lambda_sparsity",
            [
                0.5,
                0.2,
                0.1,
                0.05,
                0.02,
                0.01,
                0.005,
                0.002,
                0.001,
                0.0005,
                0.0002,
                0.0001,
            ],
        ),
    }


def model_space(trial, depth_range=(3, 6)):
    r_bar = trial.suggest_int("r_bar", 4, 5)
    return {
        "r_bar": r_bar,
        "depth": trial.suggest_int("depth", depth_range[0], depth_range[1]),
        "width": trial.suggest_int("width", 32, 32),
        "check_depth": True,
    }


def model_space_add(trial, depth_range=(3, 6), min_depth=0):
    r_bar = trial.suggest_int("r_bar", 4, 5)
    add_depth = trial.suggest_int("add_depth", 1, 1)
    depth = trial.suggest_int(
        "depth", add_depth + min_depth, max(add_depth + min_depth, depth_range[1])
    )
    return {
        "r_bar": r_bar,
        "depth": depth,
        "width": trial.suggest_int("width", 32, 32),
        "add_width": trial.suggest_int("add_width", 2, 5),
        "add_depth": add_depth,
        "check_depth": True,
    }


def model_space_bottleneck(trial, depth_range=(3, 6), min_depth=0):
    r_bar = trial.suggest_int("r_bar", 4, 5)
    return {
        "r_bar": r_bar,
        "depth": trial.suggest_int("depth", depth_range[0], depth_range[1]),
        "width": trial.suggest_int("width", 32, 32),
        "bottleneck_width": trial.suggest_int("bottleneck_width", 5, 20),
        "check_depth": True,
        "input_dropout": False,
    }


def build_models_dic(
    args: argparse.Namespace,
    train_space_fn: Callable[..., Dict[str, Any]],
    model_params: Dict[str, Any],
    model_params_dp: Dict[str, Any],
) -> Dict[str, Any]:
    oripCA_NNOpt = stat_methods.PCA_NNOpt(
        trial_train=train_space_fn,
        trial_model=functools.partial(model_space, depth_range=(2, 3)),
        loss_type=args.loss_type,
        **model_params,
    )
    oripCA_NN_PCA_ADDOpt = stat_methods.PCA_NN_PCA_ADDOpt(
        trial_train=train_space_fn,
        trial_model=functools.partial(model_space_add, depth_range=(2, 3), min_depth=2),
        loss_type=args.loss_type,
        **model_params,
    )
    autoencoderOpt = AutoencoderOpt(
        trial_train=train_space_fn,
        trial_model=functools.partial(model_space_bottleneck, depth_range=(3, 3)),
        **model_params,
    )
    factorAugmentedNNOpt = FactorAugmentedNNOpt(
        trial_train=train_space_fn,
        trial_model=functools.partial(model_space, depth_range=(2, 3)),
        **model_params_dp,
    )
    pCA_NNOpt_var = PCA_NNOpt(
        trial_train=train_space_reg_var,
        trial_model=functools.partial(model_space, depth_range=(2, 3)),
        loss_type="var",
        **model_params,
    )
    pCAA_NNOpt_var = PCAA_NNOpt(
        trial_sparsity=train_space_sparsity,
        trial_train=functools.partial(train_space_reg_var, pcaa=True),
        trial_model=functools.partial(model_space, depth_range=(2, 3)),
        loss_type="var",
        use_scheduler_step=args.use_scheduler_step,
        **model_params,
    )

    return {
        "fan_fast": NNEstimator(4),
        "fan_farm": FARM(),
        "lasso": Lasso(),
        "pcr": PCR(),
        "pls": PLS(),
        "arp": ARPAdapter(p_grid=[0, 1, 2, 3, 4, 6, 8, 12]),
        "di": DiffusionIndexARAdapter(
            p_grid=[0, 1, 2, 3, 6, 12], k_grid=[1, 2, 3, 5, 8], factor_lags=0
        ),
        "autoencoderOpt": autoencoderOpt,
        "pCAA_NNOpt_var": pCAA_NNOpt_var,
        "pCA_NNOpt_var": pCA_NNOpt_var,
        "factorAugmentedNNOpt": factorAugmentedNNOpt,
        "oripCA_NNOpt": oripCA_NNOpt,
        "oripCA_NN_PCA_ADDOpt": oripCA_NN_PCA_ADDOpt,
    }


if __name__ == "__main__":
    log_separately = True
    if args.log_dir == "" and log_separately:
        args.log_dir = suffix + f"FRED_seed{args.seed}_trial{args.n_trials}"
        # set a logger file
    os.makedirs(f"{WORKPATH}/logs/{args.log_dir}", exist_ok=True)
    script_path = os.path.dirname(__file__)
    script_folder = os.path.basename(script_path)
    try:
        shutil.copytree(script_path, f"{WORKPATH}/logs/{args.log_dir}/{script_folder}")
    except FileExistsError:
        pass
    logger = log(
        path=f"{WORKPATH}/logs/{args.log_dir}/", file="logs" + suffix.split("/")[-1]
    )
    logger.info("Train {}".format(args))

    res_dic, res_score_dic = {}, {}
    res_l = []
    if args.forecast:
        y_all = pd.DataFrame(y_all).shift(-1).values.squeeze()
    if not args.rolling_train:
        total_window = len(y_all)
        # train_window, valid_window = round(total_window * p_train), round(total_window * p_valid)
        test_window = total_window - train_window - valid_window
    else:
        total_window = train_window + valid_window + test_window
    data = [
        (x_all[i : i + total_window], y_all[i : i + total_window])
        for i in range(0, len(x_all) - total_window + test_window, test_window)
    ]
    for x_df, y_df in data:
        x_train_obs, y_train = x_df[:train_window], y_df[:train_window]
        if args.delay:
            # x_valid_obs, y_valid = x_df[train_window:train_window+valid_window-1], y_df[train_window:train_window+valid_window-1]
            x_valid_obs, y_valid = (
                x_df[train_window + 0 : train_window + valid_window - 1],
                y_df[train_window + 0 : train_window + valid_window - 1],
            )
        else:
            x_valid_obs, y_valid = (
                x_df[train_window : train_window + valid_window],
                y_df[train_window : train_window + valid_window],
            )
        x_test_obs, y_test = (
            x_df[train_window + valid_window : total_window],
            y_df[train_window + valid_window : total_window],
        )
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", x_test_obs.shape)
        if args.normalize:
            x_train_mean, x_train_std = x_train_obs.mean(axis=0), x_train_obs.std(
                axis=0
            )
            x_train_obs = (x_train_obs - x_train_mean) / x_train_std
            x_valid_obs = (x_valid_obs - x_train_mean) / x_train_std
            x_test_obs = (x_test_obs - x_train_mean) / x_train_std
            y_train_mean, y_train_std = y_train.mean(), y_train.std()
            y_train = (y_train - y_train_mean) / y_train_std
            y_valid = (y_valid - y_train_mean) / y_train_std
            y_test = (y_test - y_train_mean) / y_train_std
        train_obs_data = RegressionDataset(x_train_obs, y_train)
        train_obs_dataloader = DataLoader(
            train_obs_data, batch_size=args.batch_size, shuffle=False
        )

        valid_obs_data = RegressionDataset(x_valid_obs, y_valid)
        valid_obs_dataloader = DataLoader(
            valid_obs_data, batch_size=args.batch_size, shuffle=False
        )

        test_obs_data = RegressionDataset(x_test_obs, y_test)
        test_obs_dataloader = DataLoader(
            test_obs_data, batch_size=args.batch_size, shuffle=False
        )
        mse_loss = nn.MSELoss()
        benchmark_squared_loss = mse_loss(
            torch.tensor(y_test[:-1]), torch.tensor(y_test[1:])
        ) * len(y_test[1:])
        print("benchmark_squared_loss------", benchmark_squared_loss)

        # model
        # other data, just to learn the structure...why not from train data?
        # unlabelled_x, _, y_knowledge = far_data(args.m)
        unlabelled_x, y_knowledge = x_train_obs[: args.m], y_train[: args.m]
        # cov_mat = np.matmul(np.transpose(unlabelled_x), unlabelled_x)
        cov_mat = np.cov(unlabelled_x.T)
        # first r_bar components
        eigen_values, eigen_vectors = largest_eigsh(
            cov_mat, x_train_obs.shape[1], which="LM"
        )
        # ? why divided by np.sqrt(p), where p is num_features
        dp_matrix_large = eigen_vectors / np.sqrt(x_train_obs.shape[1])
        dp_matrix = eigen_vectors[:, -args.r_bar :] / np.sqrt(x_train_obs.shape[1])
        print(f"Diversified projection matrix size {np.shape(dp_matrix)}")
        # ========================================================================================

        # n by r_bar
        estimate_f = np.matmul(unlabelled_x, dp_matrix)
        cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
        # r_bar by p
        cov_fx_mat = np.matmul(np.transpose(estimate_f), unlabelled_x)
        # r_bar by p, estimate the throughput ui is the residuals of fitting {xi} on {fi} via linear regression
        # this is just the dp_matrix.T = eigen_vectors.T/np.sqrt(p)
        # let v be the eigenvector, (V'X'XV)'V'X'XV = I, and V'V = I
        rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)

        # ========================================================================================

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

        model_params = {
            "input_width": args.p,
            "n_trials": args.n_trials,
            "epoch": args.num_epoch,
            "random_seed": args.seed,
            "loss_fn": mse_loss,
            "device": device,
            "init_schedule": args.init_schedule,
            "patience": 100,
            "analyze": args.analyze,
            "use_loss": args.use_loss,
        }

        model_params_dp = {
            "input_width": args.p,
            "n_trials": args.n_trials,
            "epoch": args.num_epoch,
            "random_seed": args.seed,
            "loss_fn": mse_loss,
            "device": device,
            "init_schedule": args.init_schedule,
            "patience": 100,
            "dp_matrix": dp_matrix_large,
            "analyze": args.analyze,
            "use_loss": args.use_loss,
            "unlabelled_x": unlabelled_x,
        }

        oracle_model_params = copy.deepcopy(model_params)
        oracle_model_params["input_width"] = args.r

        train_space_fn = functools.partial(train_space, test_window=args.test_window)
        models_dic = build_models_dic(
            args=args,
            train_space_fn=train_space_fn,
            model_params=model_params,
            model_params_dp=model_params_dp,
        )

        projection_sum, counter = None, 0

        if args.opt:
            res = joint_train(model_l, logger=logger)
            param_df_l = []
            param_dic_l = res[-1]
            res = res[:-1]
            for model_name, param_dic in param_dic_l.items():
                if "dp_matrix" in param_dic:
                    del param_dic["dp_matrix"]
                if "rs_matrix" in param_dic:
                    del param_dic["rs_matrix"]
                if "unlabelled_x" in param_dic:
                    del param_dic["unlabelled_x"]
                param_dic["model_name"] = model_name
                param_df_l.append(pd.DataFrame([param_dic]))
            param_df = pd.concat(param_df_l)
            param_df["factor_id"], param_df["hcm_id"] = args.factor_id, args.hcm_id
            output_path = (
                str(pathlib.Path(logger.handlers[1].baseFilename).parent)
                + f"/df_param.csv"
            )
            if not args.reformat_res:
                param_df.to_csv(output_path, mode="a", header=True)
            else:
                if os.path.exists(output_path):
                    df_existing = pd.read_csv(output_path)
                    param_df = pd.concat([df_existing, param_df])
                param_df.to_csv(output_path, index=False)
            res_l.append(res)
    res_df, res_by_score_df, test_pred_dic, test_pred_by_score_dic = merge_res(res_l)
    pred_all_y = y_all[-(len(y_all) - train_window - valid_window) :]
    write_summary(
        res_df,
        res_by_score_df,
        test_pred_dic,
        test_pred_by_score_dic,
        pred_all_y,
        model_l=model_l,
        verbose=False,
    )
    df_pred = pd.DataFrame(
        np.hstack(list(test_pred_dic.values())), columns=test_pred_dic.keys()
    )
    df_pred.columns = [
        "_".join(
            [
                x,
                str(train_window),
                str(valid_window),
                str(test_window),
                str(args.lr),
                str(args.seed),
            ]
        )
        for x in df_pred.columns
    ]
    df_pred_by_score = pd.DataFrame(
        np.hstack(list(test_pred_by_score_dic.values())),
        columns=test_pred_by_score_dic.keys(),
    )
    df_pred_by_score.columns = [
        "_".join(
            [
                x,
                str(train_window),
                str(valid_window),
                str(valid_window),
                str(args.lr),
                str(args.seed),
            ]
        )
        for x in df_pred_by_score.columns
    ]
    output_path = (
        str(pathlib.Path(logger.handlers[1].baseFilename).parent) + f"/df_pred.csv"
    )
    output_path_by_score = output_path[:-4] + "_by_score.csv"
    if not os.path.exists(output_path):
        df_pred["y"] = pred_all_y
        df_pred["date"] = dff.index[-(len(y_all) - train_window - valid_window) :]
        df_pred.T.to_csv(output_path)
    else:
        df_pred.T.to_csv(output_path, mode="a", header=False)
    if not os.path.exists(output_path_by_score):
        df_pred_by_score["y"] = pred_all_y
        df_pred_by_score["date"] = dff.index[
            -(len(y_all) - train_window - valid_window) :
        ]
        df_pred_by_score.T.to_csv(output_path_by_score)
    else:
        df_pred_by_score.T.to_csv(output_path_by_score, mode="a", header=False)
