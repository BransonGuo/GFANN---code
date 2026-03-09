import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from scipy.sparse.linalg import eigsh as largest_eigsh
import torch.nn.utils.parametrize as parametrize


class RowFixedNorm(nn.Module):
    """
    Effective weight is W = s * normalize_rows(W_raw).
    - Each row direction is unit-normalized.
    - Learnable s provides scale flexibility.
    """

    def __init__(self, out_features, init_scale=1.0, eps=1e-12, per_row=False):
        super().__init__()
        self.eps = eps
        if per_row:
            self.scale = nn.Parameter(torch.ones(out_features) * init_scale)  # (out,)
        else:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))  # scalar

    def forward(self, w):
        # w: (out, in)
        w_hat = w / w.norm(dim=1, keepdim=True).clamp_min(self.eps)
        # broadcast: (out,1) * (out,in)
        return w_hat * self.scale.view(-1, 1)


class PcaOperation(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, output_size, track_gradient=True, cov_reg=0, sv_reg=0):
        super().__init__()
        self.output_size = output_size
        self.track_gradient = track_gradient
        self.cov_reg = cov_reg
        self.sv_reg = sv_reg

    def forward(self, input):
        covariance_matrix = torch.cov(input.t())
        covariance_matrix = (
            covariance_matrix
            + torch.eye(covariance_matrix.shape[0]).to("cuda") * self.cov_reg
        )
        p = covariance_matrix.shape[0]
        if self.track_gradient:
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
            except:
                eigenvalues, eigenvectors = largest_eigsh(
                    covariance_matrix.cpu().detach().numpy(),
                    self.output_size,
                    which="LM",
                )
                eigenvalues, eigenvectors = torch.tensor(eigenvalues).to(
                    "cuda"
                ), torch.tensor(eigenvectors).to("cuda")
            count = 0
            while sorted(eigenvalues.diff())[0] <= 1e-10 and count < 20:
                print("*****************************************")
                covariance_matrix = (
                    covariance_matrix
                    + 0.0001
                    * torch.randn_like(covariance_matrix)
                    * torch.diag(covariance_matrix).mean()
                )
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                count += 1
            if not eigenvectors.requires_grad:
                eigenvectors.requires_grad = True
            eigenvectors = torch.index_select(
                eigenvectors, 1, torch.arange(p - self.output_size, p).to("cuda")
            )
        else:
            eigenvalues, eigenvectors = largest_eigsh(
                covariance_matrix.cpu().detach().numpy(), self.output_size, which="LM"
            )
            eigenvalues, eigenvectors = torch.tensor(eigenvalues).to(
                "cuda"
            ), torch.tensor(eigenvectors).to("cuda")
        eigenvectors = eigenvectors.divide(np.sqrt(p))
        projection = eigenvectors
        output = torch.matmul(input, projection)
        return output, projection


class PcaLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        p_norm=1,
        weight_norm=True,
        lambda_orthogonality=1,
        loss_type="var",
        **kwargs
    ):
        """
        : loss_type: 'var' for variance, 'reconstruction' for reconstruction loss, 'sqr_norm' for squared norm loss
        lambda_orthogonality: weight for the orthogonality loss, empirical range is 0-4
        """
        super().__init__()
        self.p_norm = p_norm
        self.output_size = output_size
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.loss_type = loss_type
        if weight_norm:
            self.linear.register_full_backward_hook(self.weight_norm_hook)
        self.lambda_orthogonality = lambda_orthogonality
        self.eigenvectors = None

    def forward(
        self, x, initializing, record_proj=False, use_proj_mean=False, **kwargs
    ):
        t = self.linear(x)
        cov_matrix_out = torch.cov(t.T)
        if self.loss_type == "var":
            cov_matrix_in = torch.cov(x.T)
            var_in = torch.trace(cov_matrix_in) / (torch.tensor(x.shape[1]))
            var_out = torch.trace(cov_matrix_out) / t.shape[1]
            self.info_loss = abs(var_in - var_out)
        k = cov_matrix_out.shape[0]
        off_diag = torch.tril(cov_matrix_out, diagonal=-1)
        self.orthogonality_loss = torch.norm(off_diag, p=self.p_norm) / (
            k * (k - 1) / 2
        )
        return t, None

    def weight_norm_hook(self, module, grad_input, grad_output):
        """Hook to normalize the weight vector."""
        w = module.weight.data
        module.weight.data = (
            w / w.norm(2, dim=1, keepdim=True) / torch.sqrt(torch.tensor(w.shape[1]))
        )

    def pca_loss(self, x):
        total_loss = (
            1 - self.lambda_orthogonality
        ) * self.orthogonality_loss + self.lambda_orthogonality * self.info_loss
        return total_loss


class AdditiveLayer(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, input_size_l, output_size_l):
        super().__init__()
        assert len(input_size_l) == len(output_size_l)
        self.n_subnetwork = len(input_size_l)
        self.subnetwork_idx = input_size_l.copy()
        self.subnetwork_idx.insert(0, 0)
        self.subnetwork_idx = np.cumsum(self.subnetwork_idx)
        self.layer_list = torch.nn.ModuleList(
            [
                nn.Linear(input_size_l[i], output_size_l[i])
                for i in range(self.n_subnetwork)
            ]
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input):
        x_l = [
            self.layer_list[i](
                input[
                    :,
                    torch.arange(self.subnetwork_idx[i], self.subnetwork_idx[i + 1])
                    % input.shape[1],
                ]
            )
            for i in range(self.n_subnetwork)
        ]
        x = torch.concat(x_l, -1)
        x = x + self.bias
        return x


class DecorrLayer(nn.Module):
    def __init__(self, input_size, output_size, p_norm=1, init_with_eye=False):
        super().__init__()
        self.p_norm = p_norm
        self.linear = nn.Linear(input_size, output_size)
        if init_with_eye:
            self.linear.weight = nn.Parameter(
                torch.eye(input_size, output_size), requires_grad=True
            )

    def forward(self, x):
        x = self.linear(x)
        return x

    def decorr_loss(self, x):
        cov_matrix_out = torch.cov(x.t())
        loss = torch.norm(torch.tril(cov_matrix_out, diagonal=-1), p=self.p_norm) / (
            (cov_matrix_out.shape[0] - 1) ** 2
        )
        return loss


class CustomFunctionModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class AdditiveLayer_(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, input_size_l, output_size_l, activation=False, activation_l=[]):
        super().__init__()
        assert len(input_size_l) == len(output_size_l)
        self.n_subnetwork = len(input_size_l)
        self.subnetwork_idx = input_size_l.copy()
        self.subnetwork_idx.insert(0, 0)
        self.subnetwork_idx = np.cumsum(self.subnetwork_idx)
        self.activation = activation
        if not activation:
            self.layer_list = torch.nn.ModuleList(
                [
                    nn.Linear(input_size_l[i], output_size_l[i])
                    for i in range(self.n_subnetwork)
                ]
            )
            self.bias = torch.nn.Parameter(torch.zeros(1))
        else:
            self.activation_l = torch.nn.ModuleList(
                [
                    x if issubclass(type(x), nn.Module) else CustomFunctionModule(x)
                    for x in activation_l
                ]
            )

    def forward(self, input):
        if self.activation:
            x_l = [
                self.activation_l[i](
                    input[
                        :,
                        torch.arange(self.subnetwork_idx[i], self.subnetwork_idx[i + 1])
                        % input.shape[1],
                    ]
                )
                for i in range(self.n_subnetwork)
            ]
            x = torch.concat(x_l, -1)
        else:
            x_l = [
                self.layer_list[i](
                    input[
                        :,
                        torch.arange(self.subnetwork_idx[i], self.subnetwork_idx[i + 1])
                        % input.shape[1],
                    ]
                )
                for i in range(self.n_subnetwork)
            ]
            x = torch.concat(x_l, -1)
            x = x + self.bias
        return x


class PCA_NN(nn.Module):
    def __init__(
        self,
        p,
        r_bar,
        depth,
        width,
        input_dropout=False,
        dropout_rate=0.0,
        device="cpu",
        check_depth=False,
        **kwargs
    ):
        super().__init__()
        if check_depth:
            assert depth >= 2
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        lambda_orthogonality = kwargs.get("lambda_orthogonality", None)
        lambda_pca = kwargs.get("lambda_pca", None)
        self.lambda_pca = lambda_pca
        loss_type = kwargs.get("loss_type", None)
        self.linear = nn.Linear(p, r_bar)
        self.pca_layer = PcaLayer(
            p,
            r_bar,
            initialize_once=True,
            lambda_orthogonality=lambda_orthogonality,
            loss_type=loss_type,
        )
        if depth == 2:
            relu_nn = [("linear1", nn.Linear(r_bar, 1))]
        else:
            relu_nn = [
                ("linear1", nn.Linear(r_bar, width)),
                ("relu1", nn.LeakyReLU(0.1)),
            ]
            for i in range(3, depth):
                relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
                relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
            relu_nn.append(("linear{}".format(depth), nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))

    def forward(self, x, is_training=False, initializing=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        use_proj_mean = kwargs.get("use_proj_mean", False)
        record_proj = kwargs.get("record_proj", False)
        x, projection = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.pca_loss = self.pca_layer.pca_loss(x)
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.relu_stack(x)
        return x

    def regularization_loss(self, **kwargs):
        return self.lambda_pca * self.pca_loss


class PCAA_NN(nn.Module):
    def __init__(
        self,
        p,
        r_bar,
        depth,
        width,
        input_dropout=False,
        dropout_rate=0.0,
        device="cpu",
        check_depth=False,
        **kwargs
    ):
        super().__init__()
        if check_depth:
            assert depth >= 2
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        lambda_orthogonality = kwargs.get("lambda_orthogonality", None)
        self.lambda_pca = kwargs.get("lambda_pca", None)
        self.lambda_sparsity = kwargs.get("lambda_sparsity", 1)
        self.lambda_weight = kwargs.get("lambda_weight", 1)
        loss_type = kwargs.get("loss_type", None)
        self.dp_matrix = kwargs.get("dp_matrix", None)
        self.rs_matrix = kwargs.get("rs_matrix", None)
        if self.rs_matrix is not None:
            self.reconstruct = nn.Linear(r_bar, p, bias=False)
            rs_matrix_tensor = torch.tensor(
                np.transpose(self.rs_matrix), dtype=torch.float32
            )
            self.reconstruct.weight = nn.Parameter(
                rs_matrix_tensor, requires_grad=False
            )
        self.pca_layer = PcaLayer(
            p,
            r_bar,
            initialize_once=True,
            lambda_orthogonality=lambda_orthogonality,
            loss_type=loss_type,
        )
        sparsity = r_bar
        self.variable_selection = nn.Linear(p, sparsity, bias=False)
        if depth == 2:
            relu_nn = [("linear1", nn.Linear(r_bar + sparsity, 1))]
        else:
            relu_nn = [
                ("linear1", nn.Linear(r_bar + sparsity, width)),
                ("relu1", nn.LeakyReLU(0.1)),
            ]
            for i in range(3, depth):
                relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
                relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
            relu_nn.append(("linear{}".format(depth), nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))

    def forward(self, x, is_training=False, initializing=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        use_proj_mean = kwargs.get("use_proj_mean", False)
        record_proj = kwargs.get("record_proj", False)
        x1, projection = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.pca_loss = self.pca_layer.pca_loss(x)
        if self.rs_matrix is not None:
            x2 = self.variable_selection(x - self.reconstruct(x1))
        else:
            x_recon = x @ self.pca_layer.linear.weight.T @ self.pca_layer.linear.weight
            x2 = self.variable_selection(x - x_recon)
        x = torch.concat((x1, x2), -1)
        x = self.relu_stack(x)
        return x

    def regularization_loss(self, tau, penalize_weights=False):
        """
        Parameters
        ----------
        tau : float
                the hyper-parameter tau in the paper
        penalize_weights : bool
                whether to apply the L1 regularization to the neural network weights

        Returns
        ----------
        value : torch.tensor
                a scalar of the regularization loss
        """
        l1_penalty = torch.abs(self.variable_selection.weight) / tau
        clipped_l1 = torch.clamp(l1_penalty, max=1.0)
        if penalize_weights:
            for param in self.relu_stack.parameters():
                if len(param.shape) > 1:
                    clipped_l1 += self.lambda_weight * torch.sum(torch.abs(param))
        return (
            self.lambda_sparsity * torch.sum(clipped_l1)
            + self.lambda_pca * self.pca_loss
        )


class PCA_NN_PCA_ADD(nn.Module):
    def __init__(
        self,
        p,
        r_bar,
        depth,
        width,
        add_width,
        kernel_width=1,
        add_depth=2,
        nn_depth=-1,
        input_dropout=False,
        dropout_rate=0.0,
        device="cpu",
        check_depth=False,
        **kwargs
    ):
        super().__init__()
        init_with_eye = kwargs.get("init_with_eye", False)
        if check_depth:
            assert depth >= add_depth + 2
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(r_bar)
        lambda_orthogonality = kwargs.get("lambda_orthogonality", None)
        lambda_orthogonality2 = kwargs.get("lambda_orthogonality2", None)
        lambda_pca = kwargs.get("lambda_pca", None)
        self.lambda_pca = lambda_pca
        lambda_pca2 = kwargs.get("lambda_pca2", None)
        self.lambda_pca2 = lambda_pca
        loss_type = kwargs.get("loss_type", None)
        self.pca_layer = PcaLayer(
            p,
            r_bar,
            initialize_once=True,
            lambda_orthogonality=lambda_orthogonality,
            loss_type=loss_type,
        )
        relu_nn = []
        if nn_depth == -1:
            # infer nn_depth
            nn_depth = depth - add_depth - 2
        if nn_depth >= 1:
            relu_nn.extend(
                [("linear1", nn.Linear(r_bar, width)), ("relu1", nn.LeakyReLU(0.1))]
            )
        if nn_depth > 1:
            for i in range(2, 2 + (nn_depth - 1)):
                relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
                relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))
        if len(relu_nn) == 0:
            self.pca_layer2 = PcaLayer(
                r_bar,
                r_bar,
                weight_norm=False,
                lambda_orthogonality=lambda_orthogonality,
                loss_type=loss_type,
            )
        else:
            self.pca_layer2 = PcaLayer(
                width,
                r_bar,
                weight_norm=False,
                lambda_orthogonality=lambda_orthogonality,
                loss_type=loss_type,
            )
        add_nn = [
            ("add1", AdditiveLayer([kernel_width] * r_bar, [add_width] * r_bar)),
            ("add_relu1", nn.LeakyReLU(0.1)),
        ]
        for i in range(2, 2 + (add_depth - 1)):
            add_nn.append(
                (
                    "add{}".format(i),
                    AdditiveLayer([add_width] * r_bar, [add_width] * r_bar),
                )
            )
            add_nn.append(("add_relu{}".format(i), nn.LeakyReLU(0.1)))
        self.add_nn_stack = nn.Sequential(OrderedDict(add_nn))
        self.last_layer = nn.Linear(r_bar * add_width, 1)

    def forward(self, x, is_training=False, initializing=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        use_proj_mean = kwargs.get("use_proj_mean", False)
        record_proj = kwargs.get("record_proj", False)
        x, projection = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.pca_loss = self.pca_layer.pca_loss(x)
        x = self.relu_stack(x)
        x_decorr, projection = self.pca_layer2(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.corr_loss = self.pca_layer2.pca_loss(x)
        x = self.add_nn_stack(x_decorr)
        x = self.last_layer(x)
        x = x.reshape(-1, 1)
        return x

    def regularization_loss(self, **kwargs):
        return (
            self.lambda_pca2 * self.corr_loss + self.lambda_pca * self.pca_loss
        )  # self.lambda_pca2 *


class PCA_NN_PCA(nn.Module):
    def __init__(
        self,
        p,
        r_bar,
        depth,
        width,
        input_dropout=False,
        dropout_rate=0.0,
        device="cpu",
        check_depth=False,
        **kwargs
    ):
        super().__init__()
        init_with_eye = kwargs.get("init_with_eye", False)
        if check_depth:
            assert depth >= 2
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(r_bar)
        lambda_orthogonality = kwargs.get("lambda_orthogonality", None)
        lambda_pca = kwargs.get("lambda_pca", None)
        self.lambda_pca = lambda_pca
        lambda_pca2 = kwargs.get("lambda_pca2", None)
        self.lambda_pca2 = lambda_pca2
        loss_type = kwargs.get("loss_type", None)
        self.pca_layer = PcaLayer(
            p,
            r_bar,
            initialize_once=True,
            lambda_orthogonality=lambda_orthogonality,
            loss_type=loss_type,
        )
        relu_nn = []
        nn_depth = depth - 2
        if nn_depth >= 1:
            relu_nn.extend(
                [("linear1", nn.Linear(r_bar, width)), ("relu1", nn.LeakyReLU(0.1))]
            )
        if nn_depth > 1:
            for i in range(2, 2 + (nn_depth - 1)):
                relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
                relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))
        if len(relu_nn) == 0:
            self.pca_layer2 = PcaLayer(
                r_bar,
                r_bar,
                lambda_orthogonality=lambda_orthogonality,
                loss_type=loss_type,
            )
        else:
            self.pca_layer2 = PcaLayer(
                width,
                r_bar,
                lambda_orthogonality=lambda_orthogonality,
                loss_type=loss_type,
            )
        self.last_layer = nn.Linear(r_bar, 1)

    def forward(self, x, is_training=False, initializing=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        use_proj_mean = kwargs.get("use_proj_mean", False)
        record_proj = kwargs.get("record_proj", False)
        x, projection = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.pca_loss = self.pca_layer.pca_loss(x)
        x = self.relu_stack(x)
        x, projection = self.pca_layer2(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.corr_loss = self.pca_layer2.pca_loss(x)
        x = self.last_layer(x)
        x = x.reshape(-1, 1)
        return x

    def regularization_loss(self, **kwargs):
        return self.lambda_pca2 * self.corr_loss + self.lambda_pca * self.pca_loss


class PCA_NN_ADD_PCA(nn.Module):
    def __init__(
        self,
        p,
        r_bar,
        depth,
        width,
        add_width,
        kernel_width=1,
        add_depth=2,
        nn_depth=-1,
        input_dropout=False,
        dropout_rate=0.0,
        device="cpu",
        check_depth=False,
        **kwargs
    ):
        super().__init__()
        init_with_eye = kwargs.get("init_with_eye", False)
        if check_depth:
            assert depth >= add_depth + 2
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(r_bar)
        lambda_orthogonality = kwargs.get("lambda_orthogonality", None)
        lambda_pca = kwargs.get("lambda_pca", None)
        self.lambda_pca = lambda_pca
        lambda_pca2 = kwargs.get("lambda_pca2", None)
        self.lambda_pca2 = lambda_pca2
        loss_type = kwargs.get("loss_type", None)
        self.pca_layer = PcaLayer(
            p,
            r_bar,
            initialize_once=True,
            lambda_orthogonality=lambda_orthogonality,
            loss_type=loss_type,
        )
        relu_nn = []
        if nn_depth == -1:
            # infer nn_depth
            nn_depth = depth - add_depth - 1
        if nn_depth >= 1:
            relu_nn.extend(
                [("linear1", nn.Linear(r_bar, width)), ("relu1", nn.LeakyReLU(0.1))]
            )
        if nn_depth > 1:
            for i in range(2, 2 + (nn_depth - 1)):
                relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
                relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))
        if len(relu_nn) == 0:
            self.replace_layer = nn.Linear(r_bar, r_bar)
        else:
            self.replace_layer = nn.Linear(width, r_bar)
        add_nn = [
            ("add1", AdditiveLayer([kernel_width] * r_bar, [add_width] * r_bar)),
            ("add_relu1", nn.LeakyReLU(0.1)),
        ]
        for i in range(2, 2 + (add_depth - 1)):
            add_nn.append(
                (
                    "add{}".format(i),
                    AdditiveLayer([add_width] * r_bar, [add_width] * r_bar),
                )
            )
            add_nn.append(("add_relu{}".format(i), nn.LeakyReLU(0.1)))
        self.add_nn_stack = nn.Sequential(OrderedDict(add_nn))
        self.pca_layer2 = PcaLayer(
            r_bar * add_width,
            r_bar,
            lambda_orthogonality=lambda_orthogonality,
            loss_type=loss_type,
        )
        self.last_layer = nn.Linear(r_bar, 1)

    def forward(self, x, is_training=False, initializing=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        use_proj_mean = kwargs.get("use_proj_mean", False)
        record_proj = kwargs.get("record_proj", False)
        x, projection = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.pca_loss = self.pca_layer.pca_loss(x)
        x = self.relu_stack(x)
        x = self.add_nn_stack(x)
        x, projection = self.pca_layer2(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.corr_loss = self.pca_layer2.pca_loss(x)
        x = self.last_layer(x)
        x = x.reshape(-1, 1)
        return x

    def regularization_loss(self, **kwargs):
        return self.lambda_pca2 * self.corr_loss + self.lambda_pca * self.pca_loss


class NN_PCA_NN(nn.Module):
    def __init__(
        self,
        p,
        r_bar,
        depth,
        width,
        dp_mat=None,
        fix_dp_mat=True,
        input_dropout=False,
        dropout_rate=0.0,
        device="cpu",
        check_depth=False,
        **kwargs
    ):
        super().__init__()
        if check_depth:
            assert depth >= 3
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        lambda_orthogonality = kwargs.get("lambda_orthogonality", None)
        lambda_pca = kwargs.get("lambda_pca", None)
        self.lambda_pca = lambda_pca
        loss_type = kwargs.get("loss_type", None)
        linear = nn.Linear(p, p)
        parametrize.register_parametrization(
            linear,
            "weight",
            RowFixedNorm(out_features=linear.out_features, init_scale=1.0),
        )
        pre_nn = [
            ("pre_linear1", linear),
            ("pre_batchnorm1", nn.BatchNorm1d(p)),
            ("pre_relu1", nn.LeakyReLU(0.1)),
        ]
        self.pre_nn_stack = nn.Sequential(OrderedDict(pre_nn))
        self.pca_layer = PcaLayer(
            p, r_bar, lambda_orthogonality=lambda_orthogonality, loss_type=loss_type
        )
        relu_nn = [("linear2", nn.Linear(r_bar, width)), ("relu2", nn.LeakyReLU(0.1))]
        self.activation = nn.LeakyReLU(0.1)
        for i in range(3, depth):
            relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
            relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
        relu_nn.append(("linear{}".format(depth), nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))

    def forward(self, x, is_training=False, initializing=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        use_proj_mean = kwargs.get("use_proj_mean", False)
        record_proj = kwargs.get("record_proj", False)
        x = self.pre_nn_stack(x)
        x = self.batch_norm(x)
        x, projection = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        if is_training:
            self.pca_loss = self.pca_layer.pca_loss(x)
        x = self.relu_stack(x)
        return x

    def regularization_loss(self, **kwargs):
        return self.lambda_pca * self.pca_loss


class Autoencoder(nn.Module):
    def __init__(
        self,
        p,
        depth,
        width,
        bottleneck_width,
        input_dropout=False,
        dropout_rate=0.0,
        check_depth=False,
        **kwargs
    ):
        super(Autoencoder, self).__init__()
        if check_depth:
            assert depth >= 3
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        self.batch_norm = nn.BatchNorm1d(p)
        if depth == 3:
            relu_nn = [
                ("linear1", nn.Linear(p, bottleneck_width)),
                ("relu1", nn.LeakyReLU(0.1)),
            ]
        else:
            relu_nn = [("linear1", nn.Linear(p, width)), ("relu1", nn.LeakyReLU(0.1))]
            for i in range(4, depth):
                relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
                relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
            relu_nn.append(
                ("linear{}".format(depth), nn.Linear(width, bottleneck_width))
            )
        self.encoder = nn.Sequential(OrderedDict(relu_nn))
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_width, width),
            nn.ReLU(),
            nn.Linear(width, 1),  # Output layer for regression task (n x 1)
        )

    def forward(self, x, is_training=False, **kwargs):
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
