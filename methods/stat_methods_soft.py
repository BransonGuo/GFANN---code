from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigsh as largest_eigsh
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from collections import OrderedDict
import copy
from abc import ABC, abstractmethod
import time
from datetime import datetime
from colorama import Fore

from models.model_lib_soft_PCA import *

from config import *
import sys
from data.fast_data_standardized import RegressionDataset
from models.far_nn import *
from models.fast_nn import FactorAugmentedSparseThroughputNN

sys.path.append(WORKPATH)
from utils.utils import *

sys.path = [p for p in sys.path if p != WORKPATH]


def results_analytics(signal: np.ndarray, y: np.ndarray):
    res_dic = {}
    pos = signal.copy()
    ret_series = calc_ret_series(pos, y)
    res_dic["sharpe_ratio0"] = calc_sharpe_ratio(ret_series)
    res_dic["pct_max_dd0"] = calc_max_dd(ret_series)
    res_dic["turnover0"] = calc_turnover(signal)

    pos = signal / pd.DataFrame(signal).rolling(10, min_periods=5).std().bfill()
    ret_series = calc_ret_series(pos, y)
    res_dic["sharpe_ratio"] = calc_sharpe_ratio(ret_series)
    res_dic["pct_max_dd"] = calc_max_dd(ret_series)
    res_dic["turnover"] = calc_turnover(pos)

    signal = calibrate_signal(signal, y, window=60)
    res_dic["dir_accuracy"] = calc_directional_accuracy(signal, y)
    res_dic["IC"] = calc_IC(signal, y)
    signal = winsorize_(signal, threshold=0.05)
    pos = calc_pos_from_signal(signal, window_size=60, threshold=0.1, mode="continuous")
    res_dic["turnover1"] = calc_turnover(pos)
    # ret_series = calc_ret_series(signal, y)
    ret_series = calc_ret_series_from_pos(pos, y)
    res_dic["sharpe_ratio1"] = calc_sharpe_ratio(ret_series)
    res_dic["pct_max_dd1"] = calc_max_dd(ret_series)

    pos = calc_pos_from_signal(signal, window_size=60, threshold=0.1, mode="discrete")
    res_dic["turnover2"] = calc_turnover(pos)
    # ret_series = calc_ret_series(signal, y)
    ret_series = calc_ret_series_from_pos(pos, y)
    res_dic["sharpe_ratio2"] = calc_sharpe_ratio(ret_series)
    res_dic["pct_max_dd2"] = calc_max_dd(ret_series)

    pos = calc_pos_from_signal(signal, window_size=60, threshold=0.1, mode="absolute")
    res_dic["turnover3"] = calc_turnover(pos)
    # ret_series = calc_ret_series(signal, y)
    ret_series = calc_ret_series_from_pos(pos, y)
    res_dic["sharpe_ratio3"] = calc_sharpe_ratio(ret_series)
    res_dic["pct_max_dd3"] = calc_max_dd(ret_series)
    return pd.DataFrame(res_dic)


def assign_attributes(obj, localdict, names):
    for name in names:
        setattr(obj, name, localdict[name])


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_loop(
    data_loader,
    model,
    loss_fn,
    optimizer,
    freeze_proj=False,
    initializing=False,
    **kwargs,
):
    model.train()
    tau = kwargs.get("tau", None)
    reg_lambda = kwargs.get("reg_lambda", None)
    penalize_weights = kwargs.get("penalize_weights", True)
    reg_lambda_corr_loss = kwargs.get("reg_lambda_corr_loss", 0)
    analyze = kwargs.get("analyze", False)
    compute_score = kwargs.get("compute_score", False)
    loss_sum = 0
    pred_l, y_l = [], []
    for batch, (x, y) in enumerate(data_loader):
        global projection_sum
        global counter
        if initializing:
            try:
                for param in model.pre_nn_stack.parameters():
                    param.requires_grad = True
            except:
                pass
            try:
                for param in model.pca_layer.parameters():
                    param.requires_grad = True
            except:
                pass
            # schedule to perform PCA
            if batch == 0:
                pred = model(
                    x,
                    y=y,
                    is_training=True,
                    initializing=initializing,
                    record_proj=True,
                )
            else:
                pred = model(x, is_training=True, initializing=False)
        else:
            try:
                for param in model.pre_nn_stack.parameters():
                    param.requires_grad = False
            except:
                pass
            try:
                for param in model.pca_layer.parameters():
                    param.requires_grad = False
            except:
                pass
            pred = model(x, is_training=True, initializing=False, print_change=False)
        loss = loss_fn(pred, y)
        if (
            (reg_lambda is not None)
            and (reg_lambda > 0)
            and callable(getattr(model, "regularization_loss", None))
        ):
            reg_loss = model.regularization_loss(
                tau=tau, penalize_weights=penalize_weights
            )
            loss += reg_lambda * reg_loss
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_l.append(pred)
        y_l.append(y)
    with torch.no_grad():
        pred_all, y_all = (
            torch.concat(pred_l).cpu().numpy(),
            torch.concat(y_l).cpu().numpy(),
        )
        if analyze:
            res = results_analytics(pred_all, y_all)
            print(
                "***********************TRAIN*******************************************"
            )
            res["loss"] = loss_sum
            print(res)
        if compute_score:
            pos = calc_pos_from_signal(pred_all, window_size=60, threshold=0.1)
            ret_series = calc_ret_series(pos, y_all)
            score = -calc_sharpe_ratio(ret_series)
        else:
            score = 0
    return loss_sum / len(data_loader), score


def test_loop(data_loader, model, loss_fn, analyze=False, **kwargs):
    compute_score = kwargs.get("compute_score", False)
    loss_sum = 0
    pred_l, y_l = [], []
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            pred = model(x, is_training=False)
            loss_sum += loss_fn(pred, y).item()
            pred_l.append(pred)
            y_l.append(y)
        if analyze:
            pred_all, y_all = (
                torch.concat(pred_l).cpu().numpy(),
                torch.concat(y_l).cpu().numpy(),
            )
            res = results_analytics(pred_all, y_all)
            res["loss"] = loss_sum
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(res)
    pred_all, y_all = (
        torch.concat(pred_l).cpu().numpy(),
        torch.concat(y_l).cpu().numpy(),
    )
    if compute_score:
        ret_series = calc_ret_series(pred_all, y_all)
        score = -calc_sharpe_ratio(ret_series)
    else:
        score = 0
    return loss_sum / len(data_loader), score, pred_all


def estimate_factor_structure_from_observation(x, loadings):
    cov_b = np.matmul(np.transpose(loadings), loadings)
    inv_cov_b = np.linalg.inv(cov_b)
    factor = np.matmul(np.matmul(x, loadings), inv_cov_b)
    idiosyncratic = x - np.matmul(factor, np.transpose(loadings))
    return factor, idiosyncratic


class NN_Opt(ABC):  # Inherit from ABC(Abstract base class)
    @abstractmethod  # Decorator to define an abstract method
    def define_model(self, trial, **kwargs):
        pass

    def _maybe_step_scheduler(self, scheduler):
        if getattr(self, "use_scheduler_step", False):
            scheduler.step()

    def _maybe_step_scheduler(self, scheduler):
        if getattr(self, "use_scheduler_step", False):
            scheduler.step()

    def get_best_model_kwargs(self, best_trial_kwargs):
        best_model_kwargs = copy.deepcopy(self.model_kwargs)
        for k in best_model_kwargs.keys():
            if k in best_trial_kwargs.keys():
                best_model_kwargs[k] = best_trial_kwargs[k]
        return best_model_kwargs

    def objective(self, trial, x, y, valid_x, valid_y, cv_mode=None, k_fold=1):
        # Generate the model.
        hpspace = self.trial_train(trial)
        lambda_orthogonality = hpspace.get('lambda_orthogonality', None)
        lambda_orthogonality2 = hpspace.get('lambda_orthogonality2', None)
        lambda_pca = hpspace.get('lambda_pca', None)
        lambda_pca2 = hpspace.get('lambda_pca2', None)
        lambda_weight = hpspace.get('lambda_weight', None)
        try:
            model = self.define_model(trial, p=self.input_width, 
                                      lambda_orthogonality=lambda_orthogonality, lambda_pca=lambda_pca, lambda_weight=lambda_weight,
                                      lambda_pca2=lambda_pca2).to(self.device)
        except AssertionError:
            print('An error occurred due to illegal hyperparameters, please check the hyperparameters')
            raise optuna.exceptions.TrialPruned()
        lr, optimizer_name, batch_size = hpspace['lr'], hpspace['optimizer_name'], hpspace['batch_size']
        self.reg_lambda = hpspace.get('reg_lambda', None)
        # Generate the optimizers.
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # create dataloader.
        train_data = RegressionDataset(x, y.reshape(-1, 1))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_data = RegressionDataset(valid_x, valid_y.reshape(-1, 1))
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        print('*******************', trial.params)
        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)
        if cv_mode == None:
            best_valid_loss = float('inf')
            train_loss_best_valid = float('inf')
            # Training of the model.
            self.hp_tau = 1e-1
            anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
            anneal_tau = self.hp_tau * 10
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
            cur_valid = 1e9
            last_update = 1e9
            for epoch in range(self.epoch):
                anneal_tau -= anneal_rate
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}\n--------------------")
                if epoch in self.init_schedule:
                    train_losses, train_score = train_loop(train_dataloader, model, self.loss_fn,
                                              optimizer, initializing=True, reg_lambda=self.reg_lambda, 
                                              tau=anneal_tau, analyze=self.analyze)
                # do nothing schedule
                else:
                    train_losses, train_score = train_loop(train_dataloader, model, self.loss_fn,
                                              optimizer, reg_lambda=self.reg_lambda, 
                                              tau=anneal_tau, analyze=self.analyze)
                self._maybe_step_scheduler(scheduler)
                valid_losses, valid_score, _ = test_loop(valid_dataloader, model, self.loss_fn)
                if self.use_loss:
                    if valid_losses < best_valid_loss and epoch > 10:
                        train_loss_best_valid = train_losses
                        train_score_best_valid = train_score
                        best_valid_loss = valid_losses
                        best_valid_score = valid_score
                        print(
                            f'best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score} '
                            f'best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}')
                        if valid_losses < self.global_best_valid_loss:
                            # Create a new instance of the model's class
                            self.best_optuna_model = self.define_model(trial, p=self.input_width, 
                                        lambda_orthogonality=lambda_orthogonality, lambda_pca=lambda_pca).to(self.device)
                            self.best_optuna_model.load_state_dict(model.state_dict())  # Copy the parameters
                            self.global_best_valid_loss = valid_losses
                            self.best_epoch = epoch
                            print(f'*****Global best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score} '
                                  f'best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}')
                    trial.report(valid_losses, epoch)
                else:
                    if valid_score < best_valid_score and epoch > 10:
                        train_loss_best_valid = train_losses
                        train_score_best_valid = train_score
                        best_valid_loss = valid_losses
                        best_valid_score = valid_score
                        if best_valid_score < self.global_best_valid_loss:
                            self.global_best_valid_loss = best_valid_score
                            self.best_epoch = epoch
                            # Create a new instance of the model's class
                            self.best_optuna_model = self.define_model(trial, p=self.input_width, 
                                        lambda_orthogonality=lambda_orthogonality, lambda_pca=lambda_pca).to(self.device)
                            self.best_optuna_model.load_state_dict(model.state_dict())  # Copy the parameters
                            print(f'*****Global best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score_best_valid} '
                                  f'best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}')
                        print(
                            f'best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score_best_valid} '
                            f'best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}')
                    trial.report(valid_score, epoch)
                # Handle pruning based on the intermediate value.
                # if trial.should_prune():
                #     raise optuna.exceptions.TrialPruned()
                if epoch % 10 == 0:
                    print(f"train loss = {train_losses}, valid loss = {valid_losses}")
                if early_stopper.early_stop(valid_losses):
                    print(f'Early stopping triggered after epoch {epoch + 1}')
                    break
            print(f'best_epoch = {self.best_epoch}, train_loss_best_valid = {train_loss_best_valid}, '
                  f'train_score_best_valid = {train_score_best_valid}, best_valid_loss = {best_valid_loss}, '
                  f'best_valid_score = {best_valid_score}')
            self.best_valid_score = best_valid_score
            if self.use_loss:
                return best_valid_loss
            else:
                return best_valid_score

        elif cv_mode == 'standard':
            raise NotImplementedError

    def predict(self, test_x, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return self.model(test_x)

    def single_fit_and_predict(self, model, train_dataloader, valid_dataloader, optimizer, test_x=None, epochs=1):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        best_valid_loss = float('inf')
        best_valid_score = float('inf')
        last_update = float('inf')
        self.hp_tau = 1e-1
        anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
        anneal_tau = self.hp_tau * 10
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        for epoch in range(epochs):
            anneal_tau -= anneal_rate
            if epoch % 10 == 0:
                print(f"Epoch {epoch}\n--------------------")
            if epoch in self.init_schedule:
                train_losses, train_score = train_loop(train_dataloader, model, self.loss_fn,
                                                       optimizer, initializing=True, reg_lambda=self.reg_lambda,
                                                       tau=anneal_tau, analyze=self.analyze)
            # do nothing schedule
            else:
                train_losses, train_score = train_loop(train_dataloader, model, self.loss_fn,
                                                       optimizer, reg_lambda=self.reg_lambda, 
                                                       tau=anneal_tau, analyze=self.analyze)
            self._maybe_step_scheduler(scheduler)
            valid_losses, valid_score, _ = test_loop(valid_dataloader, model, self.loss_fn)
            if self.use_loss:
                if valid_losses < best_valid_loss and epoch > 10:
                    last_update = epoch
                    train_loss_best_valid = train_losses
                    train_score_best_valid = train_score
                    best_valid_loss = valid_losses
                    best_valid_score = valid_score
                    print(
                        f'best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score} '
                        f'best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}')
                    if test_x is not None:
                        model.eval()
                        with torch.no_grad():
                            pred_y = model(torch.tensor(test_x, dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                            self.best_state_dict = copy.deepcopy(model.state_dict())
            else:
                if valid_score < best_valid_score and epoch > 10:
                    last_update = epoch
                    train_loss_best_valid = train_losses
                    train_score_best_valid = train_score
                    best_valid_loss = valid_losses
                    best_valid_score = valid_score
                    print(
                        f'best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score_best_valid} '
                        f'best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}')
                    if test_x is not None:
                        model.eval()
                        with torch.no_grad():
                            pred_y = model(torch.tensor(test_x, dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                            self.best_state_dict = copy.deepcopy(model.state_dict())
            if epoch % 50 == 0:
                print(f"epoch {epoch}, train loss = {train_losses}, valid loss = {valid_losses}, train score = {train_score}, valid_score = {valid_score}")
        print(f'model {self.__class__.__name__}, last_update = {last_update}, train_loss_best_valid = {train_loss_best_valid}, best_valid_loss = {best_valid_loss} , '
              f'train_score_best_valid = {train_score_best_valid}, best_valid_score = {best_valid_score}')
        if test_x is not None:
            return train_loss_best_valid, train_score_best_valid, best_valid_loss, best_valid_score, pred_y
        else:
            return train_loss_best_valid, best_valid_loss

    def single_fit_by_epochs(self, model, train_dataloader, optimizer, test_x, epochs=1):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hp_tau = 1e-1
        anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
        anneal_tau = self.hp_tau * 10
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        for epoch in range(epochs):
            anneal_tau -= anneal_rate
            if epoch % 10 == 0:
                print(f"Epoch {epoch}\n--------------------")
            if epoch in self.init_schedule:
                train_losses, train_score = train_loop(train_dataloader, model, self.loss_fn,
                                                       optimizer, initializing=True, reg_lambda=self.reg_lambda,
                                                       tau=anneal_tau, analyze=self.analyze)
            # do nothing schedule
            else:
                train_losses, train_score = train_loop(train_dataloader, model, self.loss_fn,
                                                       optimizer, reg_lambda=self.reg_lambda, 
                                                       tau=anneal_tau, analyze=self.analyze)
            self._maybe_step_scheduler(scheduler)
            if epoch % 50 == 0:
                print(f"epoch {epoch}, train loss = {train_losses}, train score = {train_score}")
        print(f'model {self.__class__.__name__}')
        with torch.no_grad():
            model.eval()
            pred_y = model(torch.tensor(test_x, dtype=torch.float32).to(self.device)).cpu().detach().numpy()
            self.best_state_dict = copy.deepcopy(model.state_dict())
        return pred_y

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, direction='minimize', study_name="trial", timeout=200000,
                        n_jobs=1, retrain=False, fit_by_epochs=False, **kwargs):
        """
        timeout:
        Stop study after the given number of second(s). :obj:`None` represents no limit in
        terms of elapsed time. The study continues to create trials until the number of
        trials reaches ``n_trials``, ``timeout`` period elapses
        n_jobs:
        The number of parallel jobs. If this argument is set to ``-1``, the number is
        set to CPU count.
        """
        sampler = TPESampler(seed=self.random_seed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        func = lambda trial: self.objective(trial, x, y, valid_x, valid_y)
        study.optimize(func, n_trials=self.n_trials, timeout=timeout, n_jobs=n_jobs)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_params)
        print('Best value:', study.best_trial.value)
        # Generate the optimizers.
        optimizer_name, lr, batch_size = study.best_params.pop("optimizer"), study.best_params.pop(
            "lr"), study.best_params.pop("batch_size")
        self.best_model_kwargs = self.get_best_model_kwargs(study.best_params)
        self.best_model_kwargs['dp_matrix'], self.best_model_kwargs['rs_matrix'] = calculate_predefined_matrix(x, self.best_model_kwargs['r_bar'])
        if not retrain:
            self.model = self.best_optuna_model
            self.model.eval()
            pred_y = self.model(torch.tensor(test_x, dtype=torch.float32).to(self.device)).cpu().detach().numpy()
        else:
            self.model = self.model_class(**self.best_model_kwargs).to(self.device)
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)
            # create dataloader.
            train_data = RegressionDataset(x, y.reshape(-1, 1))
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
            valid_data = RegressionDataset(valid_x, valid_y.reshape(-1, 1))
            valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
            if fit_by_epochs:
                pred_y = self.single_fit_by_epochs(self.model, train_dataloader,
                                                    optimizer,
                                                    test_x=test_x, epochs=self.best_epoch)
                self.train_loss_best_valid = None
                self.train_score_best_valid = None
                self.best_valid_loss = None
                self.best_valid_score = None
            else:
                train_loss_best_valid, train_score_best_valid, best_valid_loss, best_valid_score, pred_y = self.single_fit_and_predict(self.model, train_dataloader,
                                                                                            valid_dataloader, optimizer,
                                                                                            test_x=test_x, epochs=self.epoch)
                self.train_loss_best_valid = train_loss_best_valid
                self.train_score_best_valid = train_score_best_valid
                self.best_valid_loss = best_valid_loss
                self.best_valid_score = best_valid_score
        self.study = study
        # if self.save_study:
        #     # start_time = time.time()
        #     # prefix = datetime.fromtimestamp(start_time).strftime("%m%d-%H%M%S")
        #     joblib.dump(study, f'{self.__class__.__name__}_study_{self.suffix}.pkl')
        return pred_y

    def fit_and_predict_cv(self, x, y, valid_x, valid_y, test_x, cv_mode='standard', k_fold=3):
        self.fit_cv(x, y, valid_x, valid_y, cv_mode, k_fold)
        return self.predict(test_x)

    def fit_cv(self, x, y, test_x, test_y, cv_mode, k_fold):
        pass


class NN_Opt_(ABC):  # Inherit from ABC(Abstract base class)
    @abstractmethod  # Decorator to define an abstract method
    def define_model(self, trial, **kwargs):
        pass

    def get_best_model_kwargs(self, best_trial_kwargs):
        best_model_kwargs = copy.deepcopy(self.model_kwargs)
        for k in best_model_kwargs.keys():
            if k in best_trial_kwargs.keys():
                best_model_kwargs[k] = best_trial_kwargs[k]
        return best_model_kwargs

    def objective(self, trial, x, y, valid_x, valid_y, cv_mode=None, k_fold=1):
        # Generate the model.
        hpspace = self.trial_train(trial)
        lambda_orthogonality = hpspace.get("lambda_orthogonality", None)
        lambda_orthogonality2 = hpspace.get("lambda_orthogonality2", None)
        lambda_pca = hpspace.get("lambda_pca", None)
        lambda_pca2 = hpspace.get("lambda_pca2", None)
        lambda_weight = hpspace.get("lambda_weight", None)
        try:
            model = self.define_model(
                trial,
                p=self.input_width,
                lambda_orthogonality=lambda_orthogonality,
                lambda_pca=lambda_pca,
                lambda_weight=lambda_weight,
                lambda_pca2=lambda_pca2,
            ).to(self.device)
        except AssertionError:
            print(
                "An error occurred due to illegal hyperparameters, please check the hyperparameters"
            )
            raise optuna.exceptions.TrialPruned()
        lr, optimizer_name, batch_size = (
            hpspace["lr"],
            hpspace["optimizer_name"],
            hpspace["batch_size"],
        )
        self.reg_lambda = hpspace.get("reg_lambda", None)
        # Generate the optimizers.
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # create dataloader.
        train_data = RegressionDataset(x, y.reshape(-1, 1))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_data = RegressionDataset(valid_x, valid_y.reshape(-1, 1))
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        print("*******************", trial.params)
        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)
        if cv_mode == None:
            best_valid_loss = float("inf")
            train_loss_best_valid = float("inf")
            # Training of the model.
            self.hp_tau = 1e-1
            anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
            anneal_tau = self.hp_tau * 10
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
            cur_valid = 1e9
            last_update = 1e9
            for epoch in range(self.epoch):
                anneal_tau -= anneal_rate
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}\n--------------------")
                if epoch in self.init_schedule:
                    train_losses, train_score = train_loop(
                        train_dataloader,
                        model,
                        self.loss_fn,
                        optimizer,
                        initializing=True,
                        reg_lambda=self.reg_lambda,
                        tau=anneal_tau,
                        analyze=self.analyze,
                    )
                # do nothing schedule
                else:
                    train_losses, train_score = train_loop(
                        train_dataloader,
                        model,
                        self.loss_fn,
                        optimizer,
                        reg_lambda=self.reg_lambda,
                        tau=anneal_tau,
                        analyze=self.analyze,
                    )
                self._maybe_step_scheduler(scheduler)
                valid_losses, valid_score, _ = test_loop(
                    valid_dataloader, model, self.loss_fn
                )
                if self.use_loss:
                    if valid_losses < best_valid_loss and epoch > 10:
                        train_loss_best_valid = train_losses
                        train_score_best_valid = train_score
                        best_valid_loss = valid_losses
                        best_valid_score = valid_score
                        print(
                            f"best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score} "
                            f"best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}"
                        )
                        if valid_losses < self.global_best_valid_loss:
                            # Create a new instance of the model's class
                            self.best_optuna_model = self.define_model(
                                trial,
                                p=self.input_width,
                                lambda_orthogonality=lambda_orthogonality,
                                lambda_pca=lambda_pca,
                            ).to(self.device)
                            self.best_optuna_model.load_state_dict(
                                model.state_dict()
                            )  # Copy the parameters
                            self.global_best_valid_loss = valid_losses
                            self.best_epoch = epoch
                            print(
                                f"*****Global best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score} "
                                f"best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}"
                            )
                    trial.report(valid_losses, epoch)
                else:
                    if valid_score < best_valid_score and epoch > 10:
                        train_loss_best_valid = train_losses
                        train_score_best_valid = train_score
                        best_valid_loss = valid_losses
                        best_valid_score = valid_score
                        if best_valid_score < self.global_best_valid_loss:
                            self.global_best_valid_loss = best_valid_score
                            self.best_epoch = epoch
                            # Create a new instance of the model's class
                            self.best_optuna_model = self.define_model(
                                trial,
                                p=self.input_width,
                                lambda_orthogonality=lambda_orthogonality,
                                lambda_pca=lambda_pca,
                            ).to(self.device)
                            self.best_optuna_model.load_state_dict(
                                model.state_dict()
                            )  # Copy the parameters
                            print(
                                f"*****Global best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score_best_valid} "
                                f"best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}"
                            )
                        print(
                            f"best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score_best_valid} "
                            f"best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}"
                        )
                    trial.report(valid_score, epoch)
                # Handle pruning based on the intermediate value.
                # if trial.should_prune():
                #     raise optuna.exceptions.TrialPruned()
                if epoch % 10 == 0:
                    print(f"train loss = {train_losses}, valid loss = {valid_losses}")
                if early_stopper.early_stop(valid_losses):
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break
            print(
                f"best_epoch = {self.best_epoch}, train_loss_best_valid = {train_loss_best_valid}, "
                f"train_score_best_valid = {train_score_best_valid}, best_valid_loss = {best_valid_loss}, "
                f"best_valid_score = {best_valid_score}"
            )
            self.best_valid_score = best_valid_score
            if self.use_loss:
                return best_valid_loss
            else:
                return best_valid_score

        elif cv_mode == "standard":
            raise NotImplementedError

    def predict(self, test_x, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return self.model(test_x)

    def single_fit_and_predict(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        test_x=None,
        epochs=1,
    ):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        best_valid_loss = float("inf")
        best_valid_score = float("inf")
        last_update = float("inf")
        self.hp_tau = 1e-1
        anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
        anneal_tau = self.hp_tau * 10
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        for epoch in range(epochs):
            anneal_tau -= anneal_rate
            if epoch % 10 == 0:
                print(f"Epoch {epoch}\n--------------------")
            if epoch in self.init_schedule:
                train_losses, train_score = train_loop(
                    train_dataloader,
                    model,
                    self.loss_fn,
                    optimizer,
                    initializing=True,
                    reg_lambda=self.reg_lambda,
                    tau=anneal_tau,
                    analyze=self.analyze,
                )
            # do nothing schedule
            else:
                train_losses, train_score = train_loop(
                    train_dataloader,
                    model,
                    self.loss_fn,
                    optimizer,
                    reg_lambda=self.reg_lambda,
                    tau=anneal_tau,
                    analyze=self.analyze,
                )
            self._maybe_step_scheduler(scheduler)
            valid_losses, valid_score, _ = test_loop(
                valid_dataloader, model, self.loss_fn
            )
            if self.use_loss:
                if valid_losses < best_valid_loss and epoch > 10:
                    last_update = epoch
                    train_loss_best_valid = train_losses
                    train_score_best_valid = train_score
                    best_valid_loss = valid_losses
                    best_valid_score = valid_score
                    print(
                        f"best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score} "
                        f"best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}"
                    )
                    if test_x is not None:
                        model.eval()
                        with torch.no_grad():
                            pred_y = (
                                model(
                                    torch.tensor(test_x, dtype=torch.float32).to(
                                        self.device
                                    )
                                )
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            self.best_state_dict = copy.deepcopy(model.state_dict())
            else:
                if valid_score < best_valid_score and epoch > 10:
                    last_update = epoch
                    train_loss_best_valid = train_losses
                    train_score_best_valid = train_score
                    best_valid_loss = valid_losses
                    best_valid_score = valid_score
                    print(
                        f"best updated, epoch = {epoch}, train_loss_best_valid = {train_losses}, train_score_best_valid = {train_score_best_valid} "
                        f"best_valid_loss = {best_valid_loss}, best_valid_score = {best_valid_score}"
                    )
                    if test_x is not None:
                        model.eval()
                        with torch.no_grad():
                            pred_y = (
                                model(
                                    torch.tensor(test_x, dtype=torch.float32).to(
                                        self.device
                                    )
                                )
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            self.best_state_dict = copy.deepcopy(model.state_dict())
            if epoch % 50 == 0:
                print(
                    f"epoch {epoch}, train loss = {train_losses}, valid loss = {valid_losses}, train score = {train_score}, valid_score = {valid_score}"
                )
        print(
            f"model {self.__class__.__name__}, last_update = {last_update}, train_loss_best_valid = {train_loss_best_valid}, best_valid_loss = {best_valid_loss} , "
            f"train_score_best_valid = {train_score_best_valid}, best_valid_score = {best_valid_score}"
        )
        if test_x is not None:
            return (
                train_loss_best_valid,
                train_score_best_valid,
                best_valid_loss,
                best_valid_score,
                pred_y,
            )
        else:
            return train_loss_best_valid, best_valid_loss

    def single_fit_by_epochs(
        self, model, train_dataloader, optimizer, test_x, epochs=1
    ):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hp_tau = 1e-1
        anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
        anneal_tau = self.hp_tau * 10
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        for epoch in range(epochs):
            anneal_tau -= anneal_rate
            if epoch % 10 == 0:
                print(f"Epoch {epoch}\n--------------------")
            if epoch in self.init_schedule:
                train_losses, train_score = train_loop(
                    train_dataloader,
                    model,
                    self.loss_fn,
                    optimizer,
                    initializing=True,
                    reg_lambda=self.reg_lambda,
                    tau=anneal_tau,
                    analyze=self.analyze,
                )
            # do nothing schedule
            else:
                train_losses, train_score = train_loop(
                    train_dataloader,
                    model,
                    self.loss_fn,
                    optimizer,
                    reg_lambda=self.reg_lambda,
                    tau=anneal_tau,
                    analyze=self.analyze,
                )
            self._maybe_step_scheduler(scheduler)
            if epoch % 50 == 0:
                print(
                    f"epoch {epoch}, train loss = {train_losses}, train score = {train_score}"
                )
        print(f"model {self.__class__.__name__}")
        with torch.no_grad():
            model.eval()
            pred_y = (
                model(torch.tensor(test_x, dtype=torch.float32).to(self.device))
                .cpu()
                .detach()
                .numpy()
            )
            self.best_state_dict = copy.deepcopy(model.state_dict())
        return pred_y

    def fit_and_predict(
        self,
        x,
        y,
        valid_x,
        valid_y,
        test_x,
        direction="minimize",
        study_name="trial",
        timeout=200000,
        n_jobs=1,
        retrain=False,
        fit_by_epochs=False,
        **kwargs,
    ):
        """
        timeout:
        Stop study after the given number of second(s). :obj:`None` represents no limit in
        terms of elapsed time. The study continues to create trials until the number of
        trials reaches ``n_trials``, ``timeout`` period elapses
        n_jobs:
        The number of parallel jobs. If this argument is set to ``-1``, the number is
        set to CPU count.
        """
        sampler = TPESampler(
            seed=self.random_seed
        )  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(
            direction=direction, study_name=study_name, sampler=sampler
        )
        func = lambda trial: self.objective(trial, x, y, valid_x, valid_y)
        study.optimize(func, n_trials=self.n_trials, timeout=timeout, n_jobs=n_jobs)
        print("Number of finished trials:", len(study.trials))
        print("Best trial:", study.best_params)
        print("Best value:", study.best_trial.value)
        # Generate the optimizers.
        optimizer_name, lr, batch_size = (
            study.best_params.pop("optimizer"),
            study.best_params.pop("lr"),
            study.best_params.pop("batch_size"),
        )
        self.best_model_kwargs = self.get_best_model_kwargs(study.best_params)
        self.best_model_kwargs["dp_matrix"], self.best_model_kwargs["rs_matrix"] = (
            calculate_predefined_matrix(x, self.best_model_kwargs["r_bar"])
        )
        if not retrain:
            self.model = self.best_optuna_model
            self.model.eval()
            pred_y = (
                self.model(torch.tensor(test_x, dtype=torch.float32).to(self.device))
                .cpu()
                .detach()
                .numpy()
            )
        else:
            self.model = self.model_class(**self.best_model_kwargs).to(self.device)
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)
            # create dataloader.
            train_data = RegressionDataset(x, y.reshape(-1, 1))
            train_dataloader = DataLoader(
                train_data, batch_size=batch_size, shuffle=False
            )
            valid_data = RegressionDataset(valid_x, valid_y.reshape(-1, 1))
            valid_dataloader = DataLoader(
                valid_data, batch_size=batch_size, shuffle=False
            )
            if fit_by_epochs:
                pred_y = self.single_fit_by_epochs(
                    self.model,
                    train_dataloader,
                    optimizer,
                    test_x=test_x,
                    epochs=self.best_epoch,
                )
                self.train_loss_best_valid = None
                self.train_score_best_valid = None
                self.best_valid_loss = None
                self.best_valid_score = None
            else:
                (
                    train_loss_best_valid,
                    train_score_best_valid,
                    best_valid_loss,
                    best_valid_score,
                    pred_y,
                ) = self.single_fit_and_predict(
                    self.model,
                    train_dataloader,
                    valid_dataloader,
                    optimizer,
                    test_x=test_x,
                    epochs=self.epoch,
                )
                self.train_loss_best_valid = train_loss_best_valid
                self.train_score_best_valid = train_score_best_valid
                self.best_valid_loss = best_valid_loss
                self.best_valid_score = best_valid_score
        self.study = study
        # if self.save_study:
        #     # start_time = time.time()
        #     # prefix = datetime.fromtimestamp(start_time).strftime("%m%d-%H%M%S")
        #     joblib.dump(study, f'{self.__class__.__name__}_study_{self.suffix}.pkl')
        return pred_y

    def fit_and_predict_cv(
        self, x, y, valid_x, valid_y, test_x, cv_mode="standard", k_fold=3
    ):
        self.fit_cv(x, y, valid_x, valid_y, cv_mode, k_fold)
        return self.predict(test_x)

    def fit_cv(self, x, y, test_x, test_y, cv_mode, k_fold):
        pass


class VanillaNNOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = RegressionNN
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class Vanilla_Bottleneck_NNOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = Regression_bottleneck_NN
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class Vanilla_ADD_NNOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = RegressionNN_additive
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class PCA_NNOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        """

        Parameters
        ----------
        trial_train: function to define hyperparameter space for training
        trial_model: function to define hyperparameter space for model
        input_width
        fold
        n_trials
        epoch
        random_seed
        loss_fn
        device
        N_TRAIN_EXAMPLES
        N_VALID_EXAMPLES
        init_schedule
        reg_lambda
        kwargs
        """
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = PCA_NN
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class PCAA_NNOpt(NN_Opt):
    def __init__(
        self,
        trial_sparsity,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=100,
        min_delta=0,
        **kwargs,
    ):
        """

        Parameters
        ----------
        trial_train: function to define hyperparameter space for training
        trial_model: function to define hyperparameter space for model
        input_width
        fold
        n_trials
        epoch
        random_seed
        loss_fn
        device
        N_TRAIN_EXAMPLES
        N_VALID_EXAMPLES
        init_schedule
        reg_lambda
        kwargs
        """
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model
        self.trial_sparsity = trial_sparsity

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        trial_model_dic.update(self.trial_sparsity(trial))
        kwargs.update(trial_model_dic)
        if "unlabelled_x" in self.model_kwargs:
            dp_matrix, rs_matrix = calculate_predefined_matrix(
                self.model_kwargs["unlabelled_x"], kwargs["r_bar"]
            )
            self.model_kwargs["dp_matrix"] = dp_matrix
            self.model_kwargs["rs_matrix"] = rs_matrix
        self.model_class = PCAA_NN
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class FactorAugmentedNNOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        """

        Parameters
        ----------
        trial_train: function to define hyperparameter space for training
        trial_model: function to define hyperparameter space for model
        input_width
        fold
        n_trials
        epoch
        random_seed
        loss_fn
        device
        N_TRAIN_EXAMPLES
        N_VALID_EXAMPLES
        init_schedule
        reg_lambda
        kwargs
        """
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = FactorAugmentedNN
        self.model_kwargs = kwargs
        self.model_kwargs["dp_matrix"] = self.dp_matrix[:, -kwargs["r_bar"] :]
        model = self.model_class(**self.model_kwargs)
        return model


class NN_PCA_NNOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = NN_PCA_NN
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class PCA_NN_ADD_PCAOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = PCA_NN_ADD_PCA
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class PCA_NN_PCAOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = PCA_NN_PCA
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


class PCA_NN_PCA_ADDOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model
        self.kwargs = kwargs

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(self.kwargs)
        kwargs.update(trial_model_dic)
        self.model_class = PCA_NN_PCA_ADD
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


from sklearn.svm import SVR


class SVREstimator:
    def __init__(self, fold_validation=5):
        self.model = None
        self.choice_C = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        self.choice_large_C = [0.2, 0.5, 1]
        self.choice_small_C = [0.001, 0.0005, 0.0002, 0.0001]
        self.fold = fold_validation

    def kfold_fit(self, x, y):
        self.model = None
        min_error = 1e9
        best_C = 0
        n = np.shape(x)[0]
        block_size = n // self.fold
        for C in self.choice_C:
            mse = 0.0
            for i in range(self.fold):
                train_x = np.concatenate(
                    [x[: block_size * i, :], x[block_size * (i + 1) :, :]], 0
                )
                train_y = np.concatenate(
                    [y[: block_size * i,], y[block_size * (i + 1) :,]], 0
                )
                test_x = x[block_size * i : block_size * (i + 1), :]
                test_y = y[block_size * i : block_size * (i + 1),]
                svr = SVR(C=C, epsilon=0.1, kernel="rbf")
                svr.fit(train_x, train_y)
                pred = svr.predict(test_x)
                assert pred.shape == test_y.shape
                mse += mean_squared_error(pred, test_y)
            if mse < min_error:
                min_error = mse
                best_C = C
        self.model = SVR(C=best_C, epsilon=0.1, kernel="rbf")
        self.model.fit(x, y)

    def model_fit(self, x, y, test_x, test_y, candidate_C):
        min_error = 1e9
        best_C = 0
        for C in candidate_C:
            svr = SVR(C=C, epsilon=0.1, kernel="rbf")
            svr.fit(x, y)
            pred = svr.predict(test_x)
            assert pred.shape == test_y.shape
            mse = mean_squared_error(pred, test_y)
            if mse < min_error:
                min_error = mse
                best_C = C
        return best_C, min_error

    def fit(self, x, y, test_x, test_y):
        self.model = None
        valid_error = 0.0
        C_1, valid_error = self.model_fit(x, y, test_x, test_y, self.choice_C)
        if C_1 == 0.2:
            C_1, valid_error = self.model_fit(x, y, test_x, test_y, self.choice_large_C)
        elif C_1 == 0.001:
            C_1, valid_error = self.model_fit(x, y, test_x, test_y, self.choice_small_C)
        self.model = SVR(C=C_1, epsilon=0.1, kernel="linear")
        self.model.fit(x, y)
        print(f"(SVR Estimator) best C = {C_1}, valid mse = {valid_error}")

    def predict(self, x, **kwargs):
        if self.model is not None:
            return self.model.predict(x)
        else:
            raise ValueError("Please first fit the model")

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, **kwargs):
        self.fit(x, y, valid_x, valid_y)
        return self.predict(test_x)


class PCR:
    def __init__(self):
        self.model = None
        self.pc_map = None
        self.loading = None

    def fit(self, x, y, test_x, test_y, fit_intercept=False):
        # x: [n, p]
        # y: [n,]
        n = np.shape(x)[0]
        self.model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        x_xt = np.matmul(x, np.transpose(x)) / n
        eigen_values, eigen_vectors = largest_eigsh(x_xt, 10, which="LM")
        ev_diff = np.log(eigen_values[0:9]) - np.log(eigen_values[1:])
        k = int(np.minimum(np.argmin(ev_diff) + 1, 6))
        print(f"number of estimated factors: {10 - k}")
        est_factor = eigen_vectors[:, k:] * np.sqrt(n)
        self.loading = np.transpose(np.matmul(np.transpose(est_factor), x)) / n
        est_idiosyncratic = x - np.matmul(est_factor, np.transpose(self.loading))
        self.model.fit(est_factor, y)

    def predict(self, x, **kwargs):
        if self.model is not None:
            est_factor, _ = estimate_factor_structure_from_observation(x, self.loading)
            return self.model.predict(est_factor)
        else:
            raise ValueError("Please first fit the model")

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, **kwargs):
        self.fit(x, y, valid_x, valid_y, fit_intercept=False)
        return self.predict(test_x)


class FARM:
    def __init__(self, use_sp=True):
        self.model_pc = None
        self.model_sp = None
        self.use_sp = use_sp
        self.loading = None

    def fit(self, x, y, test_x, test_y, fit_intercept=False):
        # x: [n, p]
        # y: [n,]
        n = np.shape(x)[0]
        self.model_pc = linear_model.LinearRegression(fit_intercept=fit_intercept)
        x_xt = np.matmul(x, np.transpose(x)) / n
        eigen_values, eigen_vectors = largest_eigsh(x_xt, 10, which="LM")
        ev_diff = np.log(eigen_values[0:9]) - np.log(eigen_values[1:])
        k = int(np.minimum(np.argmin(ev_diff) + 1, 6))
        print(f"number of estimated factors: {10 - k}")
        est_factor = eigen_vectors[:, k:] * np.sqrt(n)
        self.loading = np.transpose(np.matmul(np.transpose(est_factor), x)) / n
        est_idiosyncratic = x - np.matmul(est_factor, np.transpose(self.loading))
        self.model_pc.fit(est_factor, y)
        res_y = y - self.model_pc.predict(est_factor)
        # print(f'[train] std y = {np.std(y)}, std res_y = {np.std(res_y)}')

        if self.use_sp:
            test_factor, test_idiosyncratic = (
                estimate_factor_structure_from_observation(test_x, self.loading)
            )
            test_res_y = test_y - self.model_pc.predict(test_factor)
            # print(f'[valid] std y = {np.std(test_y)}, std res_y = {np.std(test_res_y)}')
            self.model_sp = Lasso()
            self.model_sp.fit(
                est_idiosyncratic,
                res_y,
                test_idiosyncratic,
                test_res_y,
                fit_intercept=fit_intercept,
            )

    def predict(self, x, **kwargs):
        if self.model_pc is not None:
            est_factor, est_idiosyncratic = estimate_factor_structure_from_observation(
                x, self.loading
            )
            y = self.model_pc.predict(est_factor)
            if self.model_sp is not None:
                y = y + self.model_sp.predict(est_idiosyncratic)
            return y
        else:
            raise ValueError("Please first fit the model")

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, **kwargs):
        self.fit(x, y, valid_x, valid_y, fit_intercept=False)
        return self.predict(test_x)


class Lasso:
    def __init__(self, fold_validation=5, fit_intercept=True):
        self.model = None
        self.choice_lambda = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        self.choice_large_lambda = [0.2, 0.5, 1]
        self.choice_small_lambda = [0.001, 0.0005, 0.0002, 0.0001]
        self.fold = fold_validation
        self.fit_intercept = fit_intercept

    def kfold_fit(self, x, y, fit_intercept=False):
        self.model = None
        min_error = 1e9
        best_alpha = 0
        n = np.shape(x)[0]
        block_size = n // self.fold
        for alpha in self.choice_lambda:
            mse = 0.0
            for i in range(self.fold):
                train_x = np.concatenate(
                    [x[: block_size * i, :], x[block_size * (i + 1) :, :]], 0
                )
                train_y = np.concatenate(
                    [y[: block_size * i,], y[block_size * (i + 1) :,]], 0
                )
                test_x = x[block_size * i : block_size * (i + 1), :]
                test_y = y[block_size * i : block_size * (i + 1),]
                a = linear_model.Lasso(
                    alpha=alpha, fit_intercept=fit_intercept, max_iter=100000
                )
                a.fit(train_x, train_y)
                pred = a.predict(test_x)
                assert pred.shape == test_y.shape
                mse += mean_squared_error(pred, test_y)
            if mse < min_error:
                min_error = mse
                best_alpha = alpha
        self.model = linear_model.Lasso(
            alpha=best_alpha, fit_intercept=fit_intercept, max_iter=100000
        )
        self.model.fit(x, y)

    def model_fit(self, x, y, test_x, test_y, candidate_alpha, fit_intercept=False):
        min_error = 1e9
        best_alpha = 0
        for alpha in candidate_alpha:
            tmp = linear_model.Lasso(
                alpha=alpha, fit_intercept=fit_intercept, max_iter=100000, tol=0.1
            )
            tmp.fit(x, y)
            pred = tmp.predict(test_x)
            assert pred.shape == test_y.shape
            mse = np.mean(np.square(pred - test_y))
            # print(f"mse for alpha {alpha}: {mse / self.fold}")
            if mse < min_error:
                min_error = mse
                best_alpha = alpha
        return best_alpha, min_error

    def fit(self, x, y, test_x, test_y, fit_intercept=False):
        self.model = None
        valid_error = 0.0
        alpha_1, valid_error = self.model_fit(
            x, y, test_x, test_y, self.choice_lambda, fit_intercept
        )
        if alpha_1 == 0.2:
            alpha_1, valid_error = self.model_fit(
                x, y, test_x, test_y, self.choice_large_lambda, fit_intercept
            )
        elif alpha_1 == 0.001:
            alpha_1, valid_error = self.model_fit(
                x, y, test_x, test_y, self.choice_small_lambda, fit_intercept
            )
        self.model = linear_model.Lasso(
            alpha=alpha_1, fit_intercept=fit_intercept, max_iter=100000
        )
        self.model.fit(x, y)
        print(f"(Lasso Estimator) best alpha = {alpha_1}, valid mse = {valid_error}")

    def predict(self, x, **kwargs):
        if self.model is not None:
            return self.model.predict(x)
        else:
            raise ValueError("Please first fit the model")

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, **kwargs):
        self.fit(x, y, valid_x, valid_y, fit_intercept=self.fit_intercept)
        return self.predict(test_x)


class PLS:
    def __init__(self, fold_validation=5, n_components_grid=None):
        """
        Parameters
        ----------
        fold_validation : int
            Number of folds for k-fold validation (used in kfold_fit).
        n_components_grid : list[int] or None
            Candidate numbers of PLS components; if None, defaults are used.
        """
        self.model = None
        self.fold = fold_validation
        self.scaler_X = None
        self.scaler_y = None

        # Default candidate components (later clipped to valid range).
        self.n_components_grid = (
            n_components_grid if n_components_grid is not None else [1, 2, 3, 5, 8, 12]
        )

    # ---- Helper functions ----
    def _ensure_2d_y(self, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def _clip_grid(self, n_samples, n_features):
        # Max PLS components cannot exceed min(n_samples - 1, n_features).
        max_k = max(1, min(n_samples - 1, n_features))
        return sorted(set([k for k in self.n_components_grid if 1 <= k <= max_k]))

    def _fit_scalers(self, X, y):
        self.scaler_X = StandardScaler(with_mean=True, with_std=True)
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(self._ensure_2d_y(y))
        return Xs, ys

    def _transform_X(self, X):
        return self.scaler_X.transform(X)

    def _inverse_y(self, y_scaled):
        return self.scaler_y.inverse_transform(self._ensure_2d_y(y_scaled))

    # ---- Cross-validation (blocked K-fold, aligned with the Lasso split style) ----
    def kfold_fit(self, x, y):
        """
        Select best n_components on self.n_components_grid with blocked K-fold,
        then refit on the full training set.
        """
        self.model = None
        X = np.asarray(x)
        y = self._ensure_2d_y(y)

        n, p = X.shape
        grid = self._clip_grid(n, p)
        if len(grid) == 0:
            raise ValueError(
                "No valid n_components in grid; check sample size and feature count."
            )

        # Fit scalers first (on the full training set here).
        Xs, ys = self._fit_scalers(X, y)

        min_error = 1e18
        best_k = grid[0]

        block_size = n // self.fold
        for k in grid:
            mse = 0.0
            for i in range(self.fold):
                idx_te = (
                    slice(block_size * i, block_size * (i + 1))
                    if i < self.fold - 1
                    else slice(block_size * i, n)
                )
                idx_tr = (
                    np.r_[0 : block_size * i, block_size * (i + 1) : n]
                    if i < self.fold - 1
                    else np.r_[0 : block_size * i]
                )

                Xtr, ytr = Xs[idx_tr], ys[idx_tr]
                Xte, yte = Xs[idx_te], ys[idx_te]

                # Train in standardized space.
                pls = PLSRegression(n_components=k, scale=False)
                pls.fit(Xtr, ytr)

                # Predict in standardized y, then inverse-transform for MSE.
                yhat_te_scaled = pls.predict(Xte)
                # scaler_y is fit on full training data; keep the same inverse path.
                yhat_te = self._inverse_y(yhat_te_scaled)
                yte_orig = self._inverse_y(yte)

                # Compute MSE in the original target unit.
                mse += mean_squared_error(yte_orig, yhat_te)

            if mse < min_error:
                min_error = mse
                best_k = k

        # Refit on full training data with the best k.
        self.model = PLSRegression(n_components=best_k, scale=False)
        self.model.fit(Xs, ys)
        print(
            f"(PLS Estimator) best n_components = {best_k}, cv mse = {min_error / self.fold:.6f}"
        )

    # ---- Single validation-set grid search ----
    def model_fit(self, x, y, valid_x, valid_y, candidate_components=None):
        """
        Select the best number of components from candidate_components
        on a given validation set.
        Returns (best_k, best_mse).
        """
        X_tr, y_tr = np.asarray(x), self._ensure_2d_y(y)
        X_va, y_va = np.asarray(valid_x), self._ensure_2d_y(valid_y)

        n, p = X_tr.shape
        grid = (
            candidate_components
            if candidate_components is not None
            else self.n_components_grid
        )
        grid = self._clip_grid(n, p)
        if len(grid) == 0:
            raise ValueError(
                "No valid n_components in grid; check sample size and feature count."
            )

        # Fit scalers on training only; transform validation with training scalers.
        self.scaler_X = StandardScaler(with_mean=True, with_std=True)
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = self.scaler_X.fit_transform(X_tr)
        ytr_s = self.scaler_y.fit_transform(y_tr)
        Xva_s = self._transform_X(X_va)
        yva_orig = y_va  # Original validation y, used for comparison.

        min_error = 1e18
        best_k = grid[0]

        for k in grid:
            pls = PLSRegression(n_components=k, scale=False)
            pls.fit(Xtr_s, ytr_s)
            yhat_va_scaled = pls.predict(Xva_s)
            yhat_va = self._inverse_y(
                yhat_va_scaled
            )  # Inverse-transform to original unit.
            mse = mean_squared_error(yva_orig, yhat_va)
            if mse < min_error:
                min_error = mse
                best_k = k

        return best_k, min_error

    def fit(self, x, y, valid_x, valid_y):
        """
        Select the best n_components on validation, then refit on training.
        """
        best_k, valid_mse = self.model_fit(
            x, y, valid_x, valid_y, candidate_components=self.n_components_grid
        )

        # Refit final model on training data (scalers should match final training setup).
        X_tr, y_tr = np.asarray(x), self._ensure_2d_y(y)
        Xtr_s = self.scaler_X.fit_transform(
            X_tr
        )  # Refit once to align with final training.
        ytr_s = self.scaler_y.fit_transform(y_tr)

        self.model = PLSRegression(n_components=best_k, scale=False)
        self.model.fit(Xtr_s, ytr_s)
        print(
            f"(PLS Estimator) best n_components = {best_k}, valid mse = {valid_mse:.6f}"
        )

    def predict(self, x, **kwargs):
        """
        Return predictions in original target units
        (y is inverse-transformed automatically).
        """
        if self.model is None or self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Please first fit the model")
        Xs = self._transform_X(np.asarray(x))
        yhat_scaled = self.model.predict(Xs)
        yhat = self._inverse_y(yhat_scaled)
        # For univariate y, return 1D.
        return yhat.ravel() if yhat.shape[1] == 1 else yhat

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, **kwargs):
        self.fit(x, y, valid_x, valid_y)
        return self.predict(test_x)


class ARP:
    """
    Univariate AR(p) with blocked CV and one-step-ahead out-of-sample prediction.

    API:
      - kfold_fit(y)
      - model_fit(y_tr, y_va, candidate_p)
      - fit(y_tr, y_va)
      - predict_batch(y_true, y_history)             # non-recursive, batched
      - predict_recursive(n_steps=..., y_history=...)# recursive (iterated)
      - fit_and_predict(y_tr, y_va, y_te, recursive=False)
    """

    def __init__(self, fold_validation=5, p_grid=None, fit_intercept=True, horizon=1):
        self.fold = fold_validation
        self.fit_intercept = fit_intercept
        self.h = int(horizon)
        self.p_grid = p_grid if p_grid is not None else [0, 1, 2, 3, 4, 6, 8, 12]
        self.model = None
        self.best_p = None

    # ------- helpers -------
    def _build_design(self, y, p, h):
        y = np.asarray(y).reshape(-1)
        n = len(y)
        m = n - h - p + 1
        if m <= 0:
            raise ValueError("Not enough observations for given p and horizon.")
        if p == 0:
            # intercept-only: one regressor = a column of ones
            X = np.ones((m, 1), dtype=float)
        else:
            # AR lags: columns are y_{t-1},...,y_{t-p}
            X = np.column_stack([y[p - 1 - j : p - 1 - j + m] for j in range(p)])
        y_target = y[p + h - 1 : p + h - 1 + m]
        return X, y_target

    # ------- blocked k-fold CV over p -------
    def kfold_fit(self, y):
        y = np.asarray(y).reshape(-1)
        n = len(y)
        block = n // self.fold if self.fold > 0 else n
        min_mse, best_p = 1e18, None
        for p in self.p_grid:
            ok = True
            mse = 0.0
            for i in range(self.fold):
                lo = block * i
                hi = block * (i + 1) if i < self.fold - 1 else n
                y_te = y[lo:hi]
                y_tr = np.concatenate([y[:lo], y[hi:]], axis=0)
                try:
                    Xtr, ytr = self._build_design(y_tr, p, self.h)
                    Xte, yte = self._build_design(y_te, p, self.h)
                except ValueError:
                    ok = False
                    break
                mdl = LinearRegression(
                    fit_intercept=(self.fit_intercept if p > 0 else False)
                )
                mdl.fit(Xtr, ytr)
                pred = mdl.predict(Xte)
                mse += mean_squared_error(yte, pred)
            if ok and mse < min_mse:
                min_mse, best_p = mse, p
        if best_p is None:
            raise ValueError("No valid p found; reduce p or increase data.")
        # final fit on all data with best_p
        Xall, yall = self._build_design(y, best_p, self.h)
        self.model = LinearRegression(
            fit_intercept=(self.fit_intercept if best_p > 0 else False)
        ).fit(Xall, yall)
        self.best_p = best_p
        print(f"(AR(p)) best p = {best_p}, cv mse = {min_mse / self.fold:.6f}")

    # ------- validation split selection -------
    def model_fit(self, y_tr, y_va, candidate_p=None):
        y_tr = np.asarray(y_tr).reshape(-1)
        y_va = np.asarray(y_va).reshape(-1)
        grid = candidate_p if candidate_p is not None else self.p_grid
        min_mse, best_p = 1e18, None
        for p in grid:
            try:
                Xtr, ytr = self._build_design(y_tr, p, self.h)
                Xva, yva = self._build_design(y_va, p, self.h)
            except ValueError:
                continue
            mdl = LinearRegression(
                fit_intercept=(self.fit_intercept if p > 0 else False)
            ).fit(Xtr, ytr)
            pred = mdl.predict(Xva)
            mse = mean_squared_error(yva, pred)
            if mse < min_mse:
                min_mse, best_p = mse, p
        if best_p is None:
            raise ValueError("No valid p in candidate set for given data.")
        return best_p, min_mse

    def fit(self, y_tr, y_va, candidate_p=None):
        best_p, valid_mse = self.model_fit(y_tr, y_va, candidate_p)
        Xtr, ytr = self._build_design(np.asarray(y_tr).reshape(-1), best_p, self.h)
        self.model = LinearRegression(
            fit_intercept=(self.fit_intercept if best_p > 0 else False)
        ).fit(Xtr, ytr)
        self.best_p = best_p
        print(f"(AR(p)) best p = {best_p}, valid mse = {valid_mse:.6f}")

    # ------- non-recursive, batched 1-step OOS -------
    def predict_batch(self, y_true, y_history):
        """
        Batched non-recursive forecasts for a segment (uses true y to build lags).
        y_true:    (T,) realized y over the segment you want forecasts for
        y_history: (>=p,) realized y preceding the segment start
        Returns: (T,)
        """
        if self.model is None or self.best_p is None:
            raise ValueError("Please fit the AR model first (model/best_p missing).")

        p = int(self.best_p)
        h = int(self.h)

        y_true = np.asarray(y_true).reshape(-1)
        y_hist = np.asarray(y_history).reshape(-1)
        T = y_true.shape[0]

        if p > 0 and y_hist.size < p:
            raise ValueError(
                f"y_history needs at least p={p} points; got {y_hist.size}."
            )

        # Build design on concatenated series, then take the last T rows (the segment)
        y_full = np.concatenate([y_hist, y_true])
        X_full, _ = self._build_design(y_full, p, h)

        if X_full.shape[0] < T:
            raise ValueError(
                f"Not enough design rows: got {X_full.shape[0]}, need {T}. "
                f"Check y_history length vs p={p}, h={h}."
            )

        X_seg = X_full[-T:, :]  # align with the segment
        y_pred = self.model.predict(X_seg)
        return np.asarray(y_pred).ravel()

    # ------- recursive (iterated) multi-step -------
    def predict_recursive(self, *, n_steps, y_history):
        """
        Iterated multi-step path using model's own predictions as lags.
        Returns: (n_steps,)
        """
        if self.model is None or self.best_p is None:
            raise ValueError("Please fit the AR model first (model/best_p missing).")

        p = int(self.best_p)
        y_hist = list(np.asarray(y_history).reshape(-1))
        preds = []

        for _ in range(int(n_steps)):
            if p == 0:
                x_t = np.ones((1, 1), dtype=float)  # intercept-only
            else:
                if len(y_hist) < p:
                    raise ValueError(
                        f"y_history needs at least p={p} points; got {len(y_hist)}."
                    )
                x_t = np.array(y_hist[-p:][::-1]).reshape(1, -1)
            y_hat = float(self.model.predict(x_t))
            preds.append(y_hat)
            y_hist.append(y_hat)  # feedback
        return np.array(preds)

    # ------- fit then predict a segment -------
    def fit_and_predict(self, y_tr, y_va, y_te, recursive=False):
        """
        Fit on (y_tr, y_va), then produce OOS predictions for y_te.
        - Default: non-recursive batched 1-step OOS using true lags.
        - If recursive=True: iterated multi-step path.
        Returns: (len(y_te),)
        """
        y_tr = np.asarray(y_tr).reshape(-1)
        y_va = np.asarray(y_va).reshape(-1)
        y_te = np.asarray(y_te).reshape(-1)
        self.fit(y_tr, y_va)

        history = np.concatenate([y_tr, y_va])

        if recursive:
            return self.predict_recursive(n_steps=len(y_te), y_history=history)

        # non-recursive, batched
        return self.predict_batch(y_true=y_te, y_history=history)


import numpy as np
import torch


class ARPAdapter(ARP):
    """
    Adapter that keeps the Lasso-like API:
      - fit_and_predict(x_train, y_train, x_valid, y_valid, x_test, **kwargs)
          * default: non-recursive 1-step OOS (requires y_test=...)
          * set recursive=True to get iterated path (no y_test needed)
      - predict(x, *, y_true, y_history)  -> batched, non-recursive segment forecasts

    Notes
    -----
    - All x_* inputs are ignored (AR is univariate).
    - y_* must be 1D (shape (T,)).
    - For non-recursive evaluation, we need y_history (>= p) to build lag features.
    """

    def fit_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, **kwargs):
        """
        Non-recursive (default): requires y_test=...
        Recursive: set recursive=True (uses only history, no y_test).

        Extra kwargs:
            y_test: array-like, required for non-recursive
            recursive: bool, default False
        """
        # sanitize y arrays
        y_tr = np.asarray(y_train).reshape(-1)
        y_va = np.asarray(y_valid).reshape(-1)

        # fit base ARP on (train, valid)
        self.fit(y_tr, y_va)

        recursive = bool(kwargs.get("recursive", False))
        history = np.concatenate([y_tr, y_va])

        if recursive:
            # produce an iterated path of length len(x_test)
            n_steps = len(x_test) if hasattr(x_test, "shape") else int(len(x_test))
            return self.predict_recursive(n_steps=n_steps, y_history=history)

        # --- non-recursive, 1-step OOS (default) ---
        if "y_test" not in kwargs:
            raise ValueError(
                "Non-recursive 1-step OOS requires y_test=... in fit_and_predict()."
            )
        y_te = np.asarray(kwargs["y_test"]).reshape(-1)

        # batch predict with true lags
        return self.predict_batch(y_true=y_te, y_history=history)

    def predict(self, x, *, y_true, y_history, **kwargs):
        """
        Batched, non-recursive predictions for an arbitrary segment.
        Parameters (keyword-only):
            y_true:    1D realized y for the segment to predict (length T)
            y_history: 1D realized y BEFORE the segment (must have >= p points if p>0)
        Returns: (T,)
        """
        # ignore x (can be torch or numpy), but accept it to keep API symmetry
        if isinstance(x, torch.Tensor):
            _ = x.detach().cpu().numpy()  # no-op; keep for dtype safety if needed
        return self.predict_batch(
            y_true=np.asarray(y_true).reshape(-1),
            y_history=np.asarray(y_history).reshape(-1),
        )


class DiffusionIndexAR:
    """
    Stock–Watson diffusion index forecasting:
      y_{t+h} = alpha + sum_{j=1..p} phi_j y_{t+1-j} + sum_{ell=0..L} beta_ell' F_{t-ell} + eps

    where F_t are K static factors from PCA on standardized X_t.

    API (mirrors ARP style):
      - kfold_fit(X, y)
      - model_fit(X_tr, y_tr, X_va, y_va, p_grid, k_grid)
      - fit(X_tr, y_tr, X_va, y_va)
      - predict_batch(X_seg, y_true, y_history, X_history)           # non-recursive, batched
      - predict_recursive(n_steps, y_history, X_seg, X_history=None)  # recursive (iterated)
      - fit_and_predict(X_tr, y_tr, X_va, y_va, X_te, y_te, recursive=False)
    """

    def __init__(
        self,
        fold_validation=5,
        p_grid=None,
        k_grid=None,
        factor_lags=0,
        fit_intercept=True,
        horizon=1,
        pca_random_state=0,
    ):
        self.fold = fold_validation
        self.h = int(horizon)
        self.p_grid = p_grid if p_grid is not None else [0, 1, 2, 3, 4, 6, 8, 12]
        self.k_grid = k_grid if k_grid is not None else [1, 2, 3, 5, 8]
        self.factor_lags = int(factor_lags)  # L
        self.fit_intercept = fit_intercept
        self.pca_random_state = pca_random_state

        # learned objects
        self.scaler_X = None
        self.pca = None
        self.model = None  # LinearRegression on [y-lags | factor(-lags)]
        self.best_p = None
        self.best_k = None

    # ---------- PCA helpers ----------
    def _fit_pca(self, X, k):
        """Fit scaler and PCA on TRAIN features only; return train factors F (n x k)."""
        self.scaler_X = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler_X.fit_transform(np.asarray(X))
        self.pca = PCA(
            n_components=int(k), svd_solver="full", random_state=self.pca_random_state
        )
        F = self.pca.fit_transform(Xs)
        return F  # shape (n, k)

    def _transform_factors(self, X):
        """Transform features X to factors using fitted scaler_X and PCA."""
        if self.scaler_X is None or self.pca is None:
            raise ValueError("PCA/scaler not fitted. Call fit/model_fit first.")
        Xs = self.scaler_X.transform(np.asarray(X))
        return self.pca.transform(Xs)

    # ---------- Design builder ----------
    def _build_design(self, y, F, p, h, L):
        """
        Build the linear regression design matrix using y-lags and factor lags.
        y: (n,), F: (n, K) aligned in time
        p: AR order; h: forecast horizon; L: # of factor lags
        Returns X_design (m, p + K*(L+1)), y_target (m,)
        """
        y = np.asarray(y).reshape(-1)
        F = np.asarray(F)
        if y.shape[0] != F.shape[0]:
            raise ValueError("y and F must have the same length.")

        n = y.shape[0]
        K = F.shape[1]
        lag_req = max(int(p), int(L))
        m = n - h - lag_req + 1
        if m <= 0:
            raise ValueError("Not enough observations for given p, L, and horizon.")

        rows = []
        for t in range(lag_req, lag_req + m):  # t is forecast origin
            # y-lags: y_{t-1..t-p}
            ar = [y[t - j] for j in range(1, p + 1)] if p > 0 else []
            # factor lags: F_t, F_{t-1}, ..., F_{t-L}
            fac = []
            for ell in range(L + 1):
                fac.extend(F[t - ell, :])
            rows.append(ar + fac)

        X_design = np.asarray(rows)
        y_target = y[lag_req + h - 1 : lag_req + h - 1 + m]
        return X_design, y_target

    # ---------- Blocked k-fold selection over (p, K) ----------
    def kfold_fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n, pfeat = X.shape
        block = n // self.fold if self.fold > 0 else n

        min_mse = np.inf
        best = None

        for K in self.k_grid:
            if K < 1 or K > min(n, pfeat):
                continue
            for p in self.p_grid:
                mse, ok = 0.0, True
                for i in range(self.fold):
                    lo = block * i
                    hi = block * (i + 1) if i < self.fold - 1 else n
                    X_te, y_te = X[lo:hi], y[lo:hi]
                    X_tr = np.concatenate([X[:lo], X[hi:]], axis=0)
                    y_tr = np.concatenate([y[:lo], y[hi:]], axis=0)
                    try:
                        F_tr = self._fit_pca(X_tr, K)
                        F_te = self._transform_factors(X_te)
                        Xtr_d, ytr_d = self._build_design(
                            y_tr, F_tr, p, self.h, self.factor_lags
                        )
                        Xte_d, yte_d = self._build_design(
                            y_te, F_te, p, self.h, self.factor_lags
                        )
                    except ValueError:
                        ok = False
                        break
                    mdl = LinearRegression(fit_intercept=self.fit_intercept).fit(
                        Xtr_d, ytr_d
                    )
                    pred = mdl.predict(Xte_d)
                    mse += mean_squared_error(yte_d, pred)
                if ok and mse < min_mse:
                    min_mse = mse
                    best = (p, K)

        if best is None:
            raise ValueError("No valid (p, K) found; adjust grids or data length.")

        # final fit on all data using best (p, K)
        p_best, K_best = best
        F_all = self._fit_pca(X, K_best)
        Xall, yall = self._build_design(y, F_all, p_best, self.h, self.factor_lags)
        self.model = LinearRegression(fit_intercept=self.fit_intercept).fit(Xall, yall)
        self.best_p, self.best_k = p_best, K_best
        print(
            f"(DI-AR) best p = {p_best}, best K = {K_best}, cv mse = {min_mse / self.fold:.6f}"
        )

    # ---------- Validation split selection ----------
    def model_fit(self, X_tr, y_tr, X_va, y_va, p_grid=None, k_grid=None):
        X_tr = np.asarray(X_tr)
        y_tr = np.asarray(y_tr).reshape(-1)
        X_va = np.asarray(X_va)
        y_va = np.asarray(y_va).reshape(-1)

        p_grid = p_grid if p_grid is not None else self.p_grid
        k_grid = k_grid if k_grid is not None else self.k_grid

        min_mse = np.inf
        best = None

        for K in k_grid:
            if K < 1 or K > min(len(y_tr), X_tr.shape[1]):
                continue
            # get pca factors
            F_tr = self._fit_pca(X_tr, K)
            F_va = self._transform_factors(X_va)
            for p in p_grid:
                try:
                    Xtr_d, ytr_d = self._build_design(
                        y_tr, F_tr, p, self.h, self.factor_lags
                    )
                    Xva_d, yva_d = self._build_design(
                        y_va, F_va, p, self.h, self.factor_lags
                    )
                except ValueError:
                    continue
                mdl = LinearRegression(fit_intercept=self.fit_intercept).fit(
                    Xtr_d, ytr_d
                )
                pred = mdl.predict(Xva_d)
                mse = mean_squared_error(yva_d, pred)
                if mse < min_mse:
                    min_mse, best = mse, (p, K)

        if best is None:
            raise ValueError("No valid (p, K) on validation set.")
        return best[0], best[1], min_mse

    def fit(self, X_tr, y_tr, X_va, y_va, p_grid=None, k_grid=None):
        p_best, k_best, valid_mse = self.model_fit(
            X_tr, y_tr, X_va, y_va, p_grid, k_grid
        )
        # final training on TRAIN only (to mimic your ARP style)
        F_tr = self._fit_pca(np.asarray(X_tr), k_best)
        Xtr_d, ytr_d = self._build_design(
            np.asarray(y_tr).reshape(-1), F_tr, p_best, self.h, self.factor_lags
        )
        self.model = LinearRegression(fit_intercept=self.fit_intercept).fit(
            Xtr_d, ytr_d
        )
        self.best_p, self.best_k = p_best, k_best
        print(
            f"(DI-AR) best p = {p_best}, best K = {k_best}, valid mse = {valid_mse:.6f}"
        )

    # ---------- Non-recursive, batched 1-step OOS ----------
    def predict_batch(self, X_seg, y_true, y_history, X_history):
        """
        Batched non-recursive forecasts for a segment (uses TRUE y to build lags).
        Requires:
          - X_history: features aligned with y_history (needed if factor_lags > 0)
          - X_seg    : features for the segment
        Returns: (T,)
        """
        if self.model is None or self.best_p is None or self.best_k is None:
            raise ValueError("Model not fitted.")

        X_seg = np.asarray(X_seg)
        y_true = np.asarray(y_true).reshape(-1)
        y_hist = np.asarray(y_history).reshape(-1)
        X_hist = np.asarray(X_history)

        if X_seg.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"len(X_seg)={X_seg.shape[0]} != len(y_true)={y_true.shape[0]}"
            )

        # Build factors for history+segment using fitted PCA
        F_hist = self._transform_factors(X_hist)
        F_seg = self._transform_factors(X_seg)
        F_full = np.vstack([F_hist, F_seg])

        # Build design on concatenated sequences, then take the last T rows for the segment
        y_full = np.concatenate([y_hist, y_true])
        p, h, L = int(self.best_p), int(self.h), int(self.factor_lags)

        X_full, _ = self._build_design(y_full, F_full, p, h, L)
        T = y_true.shape[0]
        # Rows belonging to the segment are the last T rows
        if X_full.shape[0] < T:
            raise ValueError(
                "History too short to build segment design; check y_history/X_history vs p, L, h."
            )
        X_seg_design = X_full[-T:, :]

        y_pred = self.model.predict(X_seg_design)
        return np.asarray(y_pred).ravel()

    # ---------- Recursive (iterated) multi-step ----------
    def predict_recursive(self, *, n_steps, y_history, X_seg, X_history=None):
        """
        Iterated multi-step path: feeds back its own predictions as lags.
        Needs X_seg (length n_steps) to compute contemporaneous factors each step.
        If factor_lags > 0, you must also supply X_history for factor lag construction.
        Returns: (n_steps,)
        """
        if self.model is None or self.best_p is None or self.best_k is None:
            raise ValueError("Model not fitted.")

        y_hist = list(np.asarray(y_history).reshape(-1))
        X_seg = np.asarray(X_seg)
        if X_seg.shape[0] != int(n_steps):
            raise ValueError("X_seg length must equal n_steps.")

        p, h, L = int(self.best_p), int(self.h), int(self.factor_lags)

        # Prepare factor history if needed
        if L > 0:
            if X_history is None:
                raise ValueError("X_history is required when factor_lags > 0.")
            F_hist = self._transform_factors(np.asarray(X_history))
        else:
            F_hist = np.zeros(
                (len(y_hist), self.best_k)
            )  # unused when L=0 but keeps shapes tidy

        preds = []
        # rolling over steps
        for t in range(int(n_steps)):
            # assemble factors for current origin
            F_t = self._transform_factors(X_seg[t : t + 1])  # shape (1, K)
            # Gather factor lags: F_t, F_{t-1}, ..., F_{t-L}
            fac_blocks = [F_t]
            for ell in range(1, L + 1):
                # pull from (history + prior segment rows)
                if len(fac_blocks) - 1 >= t:  # we only have t previous seg rows so far
                    # Use most recent available: either from F_hist or prior F_seg transforms
                    idx = (
                        len(F_hist) + t - (ell - 1)
                    )  # index in F_full up to current time
                else:
                    idx = len(F_hist) - (ell - t)  # still in history
                if idx <= 0:
                    raise ValueError("Not enough factor history to construct lags.")
                # We can simply re-transform 1-row slices:
                # (more efficient would be caching, but this is safe.)
                fac_blocks.append(
                    self._transform_factors(
                        (X_history if idx <= len(F_hist) else X_seg)[(idx - 1) : (idx)]
                    )
                )
            # flatten factor blocks newest->oldest
            fac_vec = np.concatenate([blk.reshape(-1) for blk in fac_blocks], axis=0)

            # Build AR lag vector
            if p == 0:
                x_ar = np.ones((1, 1), dtype=float)
            else:
                if len(y_hist) < p:
                    raise ValueError(
                        f"Need at least p={p} y_history points; got {len(y_hist)}."
                    )
                x_ar = np.array(y_hist[-p:][::-1]).reshape(1, -1)

            # Concatenate features: [AR lags | factor lags]
            X_row = (
                np.hstack(
                    [
                        x_ar.reshape(1, -1) if p > 0 else np.ones((1, 1)),
                        fac_vec.reshape(1, -1),
                    ]
                )
                if L >= 0
                else x_ar
            )

            y_hat = float(self.model.predict(X_row))
            preds.append(y_hat)
            y_hist.append(y_hat)  # recursive feedback for next AR lags

        return np.array(preds)

    # ---------- Fit then predict a segment ----------
    def fit_and_predict(self, X_tr, y_tr, X_va, y_va, X_te, y_te, recursive=False):
        """
        Fit on (X_tr, y_tr) with tuning by (X_va, y_va), then produce OOS predictions for (X_te, y_te).
        - Default: non-recursive batched 1-step OOS using TRUE lags and PCA trained on X_tr.
        - If recursive=True: iterated multi-step path using X_te (and X_history if L>0).
        Returns: (len(y_te),)
        """
        X_tr = np.asarray(X_tr)
        y_tr = np.asarray(y_tr).reshape(-1)
        X_va = np.asarray(X_va)
        y_va = np.asarray(y_va).reshape(-1)
        X_te = np.asarray(X_te)
        y_te = np.asarray(y_te).reshape(-1)

        self.fit(X_tr, y_tr, X_va, y_va)

        # histories (for design building): concat train + valid
        y_hist = np.concatenate([y_tr, y_va])
        X_hist = np.vstack([X_tr, X_va])

        if recursive:
            return self.predict_recursive(
                n_steps=len(y_te),
                y_history=y_hist,
                X_seg=X_te,
                X_history=(X_hist if self.factor_lags > 0 else None),
            )

        # Non-recursive, batched 1-step OOS
        return self.predict_batch(
            X_seg=X_te, y_true=y_te, y_history=y_hist, X_history=X_hist
        )


class DiffusionIndexARAdapter(DiffusionIndexAR):
    """
    Adapter to keep a Lasso-like API around DiffusionIndexAR.

    - fit_and_predict(x_train, y_train, x_valid, y_valid, x_test, **kwargs)
        * Default: non-recursive 1-step OOS (requires y_test=...)
        * Set recursive=True for iterated path (no y_test needed)
    - predict(x, *, y_true, y_history=None, X_history=None)
        * Batched non-recursive segment prediction
        * If y_history/X_history are omitted, uses the history cached at fit time
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._X_hist = None  # features history (train+valid)
        self._y_hist = None  # target history (train+valid)

    def fit_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, **kwargs):
        """
        Non-recursive (default): requires y_test=...
        Recursive: set recursive=True (uses only history, no y_test).

        Extra kwargs:
            y_test:   array-like, required for non-recursive
            recursive: bool, default False
        """
        # sanitize arrays
        X_tr = np.asarray(x_train)
        y_tr = np.asarray(y_train).reshape(-1)
        X_va = np.asarray(x_valid)
        y_va = np.asarray(y_valid).reshape(-1)
        X_te = np.asarray(x_test)

        # fit and cache histories
        self.fit(X_tr, y_tr, X_va, y_va)
        self._X_hist = np.vstack([X_tr, X_va])
        self._y_hist = np.concatenate([y_tr, y_va])

        recursive = bool(kwargs.get("recursive", False))

        if recursive:
            # iterated multi-step path (uses X_te; needs X_history if factor_lags>0)
            return self.predict_recursive(
                n_steps=len(X_te),
                y_history=self._y_hist,
                X_seg=X_te,
                X_history=(self._X_hist if self.factor_lags > 0 else None),
            )

        # --- non-recursive, 1-step OOS (default) ---
        if "y_test" not in kwargs:
            raise ValueError(
                "Non-recursive 1-step OOS requires y_test=... in fit_and_predict()."
            )
        y_te = np.asarray(kwargs["y_test"]).reshape(-1)
        if y_te.shape[0] != X_te.shape[0]:
            raise ValueError(
                f"Length mismatch: len(X_test)={X_te.shape[0]} vs len(y_test)={y_te.shape[0]}"
            )

        return self.predict_batch(
            X_seg=X_te, y_true=y_te, y_history=self._y_hist, X_history=self._X_hist
        )

    def predict(self, x, *, y_true, y_history=None, X_history=None):
        """
        Batched, non-recursive predictions for a segment.

        Parameters (keyword-only):
            y_true:     1D realized y for the segment (length T)
            y_history:  1D realized y BEFORE the segment (>= p), defaults to cached train+valid
            X_history:  Features aligned with y_history (needed if factor_lags>0),
                        defaults to cached train+valid features

        Returns:
            np.ndarray of shape (T,)
        """
        # accept torch or numpy for x
        X_seg = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        y_true = np.asarray(y_true).reshape(-1)

        # histories (fallback to cached if not provided)
        y_hist = (
            np.asarray(y_history).reshape(-1)
            if y_history is not None
            else (self._y_hist if self._y_hist is not None else None)
        )
        X_hist = (
            np.asarray(X_history)
            if X_history is not None
            else (self._X_hist if self._X_hist is not None else None)
        )

        if y_hist is None or X_hist is None:
            raise ValueError(
                "y_history and X_history are required (or call fit_and_predict first)."
            )
        if y_true.shape[0] != X_seg.shape[0]:
            raise ValueError(
                f"Length mismatch: len(X_seg)={X_seg.shape[0]} vs len(y_true)={y_true.shape[0]}"
            )

        return self.predict_batch(
            X_seg=X_seg, y_true=y_true, y_history=y_hist, X_history=X_hist
        )


def train_loop_reg(data_loader, model, loss_fn, optimizer, reg_lambda, reg_tau):
    loss_rec = {"l2_loss": 0.0}
    if reg_tau is not None:
        loss_rec["reg_loss"] = 0.0
    for batch, (x, y) in enumerate(data_loader):
        pred = model(x, is_training=True)
        loss = loss_fn(pred, y)
        loss_rec["l2_loss"] += loss.item()
        if reg_tau is not None:
            reg_loss = model.regularization_loss(reg_tau, True)
            loss_rec["reg_loss"] += reg_lambda * reg_loss
            loss += reg_lambda * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_rec["l2_loss"] /= len(data_loader)
    if reg_tau is not None:
        loss_rec["reg_loss"] /= len(data_loader)
    return loss_rec


def test_loop_reg(data_loader, model, loss_fn, reg_lambda, reg_tau):
    loss_sum = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred = model(x, is_training=False)
            loss_sum += loss_fn(pred, y).item()
    loss_rec = {"l2_loss": loss_sum / len(data_loader)}
    if reg_tau is not None:
        loss_rec["reg_loss"] = reg_lambda * model.regularization_loss(reg_tau, True)
    return loss_rec


class NNEstimator:
    def __init__(self, r_bar=4):
        self.model = True
        self.r_bar = r_bar
        self.learning_rate = 1e-3
        self.epoch = 300
        self.model_color = Fore.YELLOW
        self.depth = 3
        self.width = 32
        self.hp_tau = 1e-1
        self.n_ensemble = 1
        self.p = None
        self.choice_lambda = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
                              0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        self.dp_matrix = None
        self.rs_matrix = None
        self.global_best_valid_loss = float('inf')

    def single_fit_and_predict(self, train_data_loader, valid_data_loader, test_x, reg_lambda):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = \
            FactorAugmentedSparseThroughputNN(p=self.p, r_bar=self.r_bar, depth=self.depth,
                                              width=self.width, sparsity=self.r_bar,
                                              dp_matrix=self.dp_matrix, rs_mat=self.rs_matrix).to(device)
        anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.epoch
        anneal_tau = self.hp_tau * 10

        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        cur_valid = 1e9
        last_update = 1e9
        for epoch in range(self.epoch):
            anneal_tau -= anneal_rate
            train_losses = train_loop_reg(train_data_loader, self.model, mse_loss, optimizer, reg_lambda, anneal_tau)
            scheduler.step()
            valid_losses = test_loop_reg(valid_data_loader, self.model, mse_loss, reg_lambda, anneal_tau)
            if valid_losses['l2_loss'] < cur_valid:
                cur_valid = valid_losses['l2_loss']
                last_update = epoch
                with torch.no_grad():
                    test_y = self.model(torch.tensor(test_x, dtype=torch.float32).to(device)).cpu().detach().numpy()
        print(f'[FAST-NN] lambda = {reg_lambda}, last_update = {last_update}, train_losses = {train_losses}, valid loss = {cur_valid}')
        return cur_valid, test_y

    def model_fit_and_predict(self, x, y, valid_x, valid_y, test_x, candidate_lambda):
        dp_matrix, rs_matrix = calculate_predefined_matrix(x, self.r_bar)
        self.p = np.shape(x)[1]
        self.dp_matrix, self.rs_matrix = dp_matrix, rs_matrix
        y_ex = np.reshape(y, (np.shape(y)[0], 1))
        valid_y_ex = np.reshape(valid_y, (np.shape(valid_y)[0], 1))

        # build dataset
        torch_train = RegressionDataset(x, y_ex)
        train_data_loader = DataLoader(torch_train, batch_size=np.shape(x)[0] // 4)
        torch_valid = RegressionDataset(valid_x, valid_y_ex)
        valid_data_loader = DataLoader(torch_valid, batch_size=np.shape(valid_x)[0])

        self.best_valid_loss = 1e9
        best_lambda = None
        test_y = None
        for reg_lambda in candidate_lambda:
            # create model
            valid_error = 0.0
            y_pred = 0.0
            for t in range(self.n_ensemble):
                valid_error_, y_pred_ = \
                    self.single_fit_and_predict(train_data_loader, valid_data_loader, test_x, reg_lambda)
                valid_error += valid_error_
                y_pred += y_pred_
            valid_error /= self.n_ensemble
            y_pred /= self.n_ensemble
            if self.best_valid_loss > valid_error:
                self.best_valid_loss = valid_error
                best_lambda = reg_lambda
                test_y = y_pred
        return self.best_valid_loss, best_lambda, test_y

    def fit_and_predict(self, x, y, valid_x, valid_y, test_x, **kwargs):
        self.best_valid_loss, best_lambda, test_y = self.model_fit_and_predict(x, y, valid_x, valid_y, test_x, self.choice_lambda)
        test_y = np.reshape(test_y, (np.shape(test_y)[0],))
        print(f"(FAST-NN Estimator) best alpha = {best_lambda}, valid mse = {self.best_valid_loss}")
        return test_y
    
    def predict(self, test_x, **kwargs):
        with torch.no_grad():
            return self.model(test_x)
    
    @property
    def best_valid_score(self):
        return self.best_valid_loss

def calculate_predefined_matrix(unlabelled_x, r_bar):
    p = np.shape(unlabelled_x)[1]
    cov_mat = np.matmul(np.transpose(unlabelled_x), unlabelled_x)
    eigen_values, eigen_vectors = largest_eigsh(cov_mat, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)
    estimate_f = np.matmul(unlabelled_x, dp_matrix)
    cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
    cov_fx_mat = np.matmul(np.transpose(estimate_f), unlabelled_x)
    rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)
    return dp_matrix, rs_matrix



def calculate_predefined_matrix(unlabelled_x, r_bar):
    p = np.shape(unlabelled_x)[1]
    cov_mat = np.matmul(np.transpose(unlabelled_x), unlabelled_x)
    eigen_values, eigen_vectors = largest_eigsh(cov_mat, r_bar, which="LM")
    dp_matrix = eigen_vectors / np.sqrt(p)
    estimate_f = np.matmul(unlabelled_x, dp_matrix)
    cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
    cov_fx_mat = np.matmul(np.transpose(estimate_f), unlabelled_x)
    rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)
    return dp_matrix, rs_matrix


class AutoencoderOpt(NN_Opt):
    def __init__(
        self,
        trial_train,
        trial_model,
        input_width,
        fold=5,
        n_trials=50,
        epoch=300,
        random_seed=0,
        loss_fn=None,
        device="cuda",
        N_TRAIN_EXAMPLES=float("inf"),
        N_VALID_EXAMPLES=float("inf"),
        init_schedule=[0],
        reg_lambda=None,
        save_study=False,
        suffix="",
        patience=50,
        min_delta=0,
        **kwargs,
    ):
        # default values change with models
        # assign arguments to self
        assign_attributes(
            self,
            vars(),
            [
                "input_width",
                "fold",
                "n_trials",
                "random_seed",
                "device",
                "epoch",
                "loss_fn",
                "N_TRAIN_EXAMPLES",
                "N_VALID_EXAMPLES",
                "init_schedule",
                "reg_lambda",
                "save_study",
                "suffix",
                "patience",
                "min_delta",
            ],
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_kwargs = kwargs
        self.best_iteration = 1
        self.global_best_valid_loss = float("inf")
        self.trial_train = trial_train
        self.trial_model = trial_model

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(trial_model_dic)
        self.model_class = Autoencoder
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model
