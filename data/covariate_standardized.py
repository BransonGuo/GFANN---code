import numpy as np
from sklearn.preprocessing import StandardScaler
import data.univariate_funcs as univariate_funcs

func_zoo = [
    univariate_funcs.func1,
    univariate_funcs.func2,
    univariate_funcs.func3,
    univariate_funcs.func4,
    univariate_funcs.func5,
    univariate_funcs.func6,
    univariate_funcs.func7,
    univariate_funcs.func8,
    univariate_funcs.func9
]


class FactorModel:
    """
        The data generating process of linear factor model

        ...

        Attributes
        ----------
        loadings : numpy.array
            [p, r] factor loading matrix

    """

    def __init__(self, p, r=5, b_f=1, b_u=1, loadings=None, func_idx=0, func_l=[]):
        """
            Parameters
            ----------
            p : int
                number of covariates
            r : int
                number of factors
            b_f : float
                noise level of factors
            b_u : float
                noise level of idiosyncratic components
            loadings : numpy.array
                pre-specified factor loading matrix

            Returns
            -------
            loadings : numpy.array
                [p, r] matrix, factor loadings
        """

        self.p = p
        self.r = r
        self.b_f = b_f
        self.b_u = b_u
        self.func_idx = func_idx
        self.func_l = func_l
        if len(self.func_l) == 0:
            self.func_l = np.random.randint(0, 9, 10)
        if r > 0:
            if loadings is None:
                self.loadings = np.reshape(
                    np.random.uniform(-np.sqrt(3), np.sqrt(3), p * r), (p, r))
            else:
                self.loadings = loadings
        else:
            self.loadings = None

    def sample(self, n, latent=False):
        """
            Parameters
            ----------
            n : int
                number of samples
            latent : bool
                whether return the latent factor structure

            Returns
            -------
            obs : np.array
                [n, p] matrix, observations
            factor : np.array
                [n, r] matrix, factor
            idiosyncratic_error : np.array
                [n, p] matrix, idiosyncratic error
        """
        # create factor
        if self.r > 0:
            factor = np.reshape(
                np.random.uniform(-self.b_f, self.b_f, n * self.r), (n, self.r))
        idiosyncratic_error = np.reshape(
            np.random.uniform(-self.b_u, self.b_u, self.p * n), (n, self.p))
        # create obs
        if self.r > 0:
            if self.func_idx == 0:
                # factor is n by r, loading is p by r, idio is n by p
                obs = np.matmul(factor, np.transpose(
                    self.loadings)) + idiosyncratic_error
            if self.func_idx == 1:
                # factor is n by r, loading is p by r, idio is n by p
                obs = np.matmul(np.exp(factor), np.transpose(
                    self.loadings)) + idiosyncratic_error
            if self.func_idx == 2:
                # factor is n by r, loading is p by r, idio is n by p
                obs = np.matmul(factor, np.transpose(self.loadings))
                obs = obs/(obs.max() - obs.min()) * 2 * self.b_f
                obs = np.exp(obs) + idiosyncratic_error
        else:
            obs = idiosyncratic_error
        if latent and self.r > 0:
            return obs, factor, idiosyncratic_error
        else:
            return obs
