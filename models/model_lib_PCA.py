import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from scipy.sparse.linalg import eigsh as largest_eigsh


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
        projection = eigenvectors
        output = torch.matmul(input, projection)
        return output, projection


class PcaLayer(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(
        self,
        input_size,
        output_size,
        initialize_once=False,
        track_gradient=False,
        cov_reg=0,
        sv_reg=0,
    ):
        super().__init__()
        self.PcaOperation = PcaOperation(
            output_size, track_gradient=track_gradient, cov_reg=cov_reg, sv_reg=sv_reg
        )
        self.place_holder = nn.Linear(input_size, output_size, bias=False)
        self.projection_sum = None
        self.counter = 0
        self.initialize_once = initialize_once
        self.initialized = False
        self.track_gradient = track_gradient

    def record_projection(self, input, record_proj=False):
        if record_proj:
            if self.projection_sum is None:
                self.projection_sum = input.clone()
            else:
                self.projection_sum = torch.concat((self.projection_sum, input))

    def forward(
        self, input, initializing, record_proj=False, use_proj_mean=False, **kwargs
    ):
        if self.initialize_once:
            if self.initialized:
                projection = None
                x = self.place_holder(input)
                return x, self.place_holder.weight.T
        if initializing:
            pca_x, projection = self.PcaOperation(input)
            projection = projection.divide(np.sqrt(input.shape[1]))
            with torch.no_grad():
                multiplier = 1
            projection = projection * multiplier
            self.place_holder.weight = nn.Parameter(projection.T, requires_grad=False)
            # self.place_holder.bias = nn.Parameter(torch.add(self.place_holder.bias, self.input_mean))
            self.initialized = True
            if self.track_gradient:
                pca_x = pca_x * multiplier
                return pca_x, projection
            x = self.place_holder(input)
            return x, projection
        else:
            projection = None
            x = self.place_holder(input)
            return x, self.place_holder.weight.T


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


class CustomFunctionModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


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
        corr_matrix = torch.cov(x.t())
        loss = torch.norm(torch.tril(corr_matrix, diagonal=-1), p=self.p_norm) / (
            (corr_matrix.shape[0] - 1) * 2
        )
        return loss


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
        self.pca_layer = PcaLayer(p, r_bar, initialize_once=True)
        relu_nn = [("linear1", nn.Linear(r_bar, width)), ("relu1", nn.LeakyReLU(0.1))]
        for i in range(2, depth):
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
        x = self.relu_stack(x)
        return x


class PCA_NN_PCA(nn.Module):
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
            assert depth >= 2
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(p)
        self.pca_layer = PcaLayer(p, r_bar, initialize_once=True)
        relu_nn = [("linear1", nn.Linear(r_bar, width)), ("relu1", nn.LeakyReLU(0.1))]
        self.activation = nn.LeakyReLU(0.1)
        self.activation_Tanh = nn.Tanh()
        for i in range(2, depth):
            relu_nn.append(("linear{}".format(i), nn.Linear(width, width)))
            relu_nn.append(("relu{}".format(i), nn.LeakyReLU(0.1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_nn))
        self.pca_layer2 = PcaLayer(width, r_bar)
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
        x = self.relu_stack(x)
        x, projection = self.pca_layer2(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        pred = self.last_layer(x)
        return pred


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
        self.batch_norm = nn.BatchNorm1d(p)
        self.pca_layer = PcaLayer(p, r_bar, initialize_once=True)
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
            self.pca_layer2 = PcaLayer(r_bar, r_bar)
        else:
            self.pca_layer2 = PcaLayer(width, r_bar)
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
        x = self.relu_stack(x)
        x, projection = self.pca_layer2(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        x = self.add_nn_stack(x)
        x = self.last_layer(x)
        x = x.reshape(-1, 1)
        return x


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
        self.batch_norm = nn.BatchNorm1d(p)
        self.pca_layer = PcaLayer(p, r_bar, initialize_once=True)
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
        self.pca_layer2 = PcaLayer(add_width * r_bar, r_bar)
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
        x = self.relu_stack(x)
        x = self.replace_layer(x)
        x = self.add_nn_stack(x)
        x, projection = self.pca_layer2(
            x,
            initializing=initializing,
            record_proj=record_proj,
            use_proj_mean=use_proj_mean,
        )
        x = self.last_layer(x)
        x = x.reshape(-1, 1)
        return x
