from sklearn.svm import SVR
from sklearn import linear_model
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigsh as largest_eigsh
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from data.fast_data_standardized import RegressionDataset
from models.far_nn import *
import copy
from abc import ABC, abstractmethod
from models.model_lib_PCA import *
from config import *
import sys

sys.path.append(WORKPATH)
from utils.utils import *

sys.path = [p for p in sys.path if p != WORKPATH]


def results_analytics(signal: np.ndarray, y: np.ndarray):
    res_dic = {}
    signal = calibrate_signal(signal, y, window=60)
    res_dic["dir_accuracy"] = calc_directional_accuracy(signal, y)
    res_dic["IC"] = calc_IC(signal, y)
    pos = calc_pos_from_signal(signal)
    res_dic["turnover"] = calc_turnover(pos)
    # ret_series = calc_ret_series(signal, y)
    ret_series = calc_ret_series_from_pos(pos, y)
    res_dic["sharpe_ratio"] = calc_sharpe_ratio(ret_series)
    res_dic["pct_max_dd"] = calc_max_dd(ret_series)
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
    penalize_weights = kwargs.get("penalize_weights", None)
    reg_lambda_corr_loss = kwargs.get("reg_lambda_corr_loss", 0)
    analyze = kwargs.get("analyze", False)
    compute_score = kwargs.get("compute_score", False)
    loss_sum = 0
    pred_l, y_l = [], []
    for batch, (x, y) in enumerate(data_loader):
        global projection_sum
        global counter
        if initializing:
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

    def get_best_model_kwargs(self, best_trial_kwargs):
        best_model_kwargs = copy.deepcopy(self.model_kwargs)
        for k in best_model_kwargs.keys():
            if k in best_trial_kwargs.keys():
                best_model_kwargs[k] = best_trial_kwargs[k]
        return best_model_kwargs

    def objective(self, trial, x, y, valid_x, valid_y, cv_mode=None, k_fold=1):
        # Generate the model.
        try:
            model = self.define_model(trial, p=self.input_width).to(self.device)
        except AssertionError:
            print(
                "An error occurred due to illegal hyperparameters, please check the hyperparameters"
            )
            raise optuna.exceptions.TrialPruned()
        hpspace = self.trial_train(trial)
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
                # scheduler.step()
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
                                trial, p=self.input_width
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
                                trial, p=self.input_width
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
            # scheduler.step()
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
            # scheduler.step()
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

        def func(trial):
            return self.objective(trial, x, y, valid_x, valid_y)

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
        # if 'dp_matrix' in self.best_model_kwargs:
        #     self.best_model_kwargs['dp_matrix'] = self.dp_matrix[:, -self.best_model_kwargs['r_bar']:]
        # study.best_params['r_bar'] is the best, but self.model_kwargs['r_bar'] is the most recent
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
        self.model_kwargs = kwargs
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
        self.kwargs = kwargs

    def define_model(self, trial, **kwargs):
        trial_model_dic = self.trial_model(trial)
        kwargs.update(self.kwargs)
        kwargs.update(trial_model_dic)
        self.model_class = PCA_NN_ADD_PCA
        self.model_kwargs.update(kwargs)
        model = self.model_class(**self.model_kwargs)
        return model


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
