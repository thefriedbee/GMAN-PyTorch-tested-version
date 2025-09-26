import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALLEST_FLOAT = torch.finfo(torch.float32).min


class conv2d_(nn.Module):
    """
    Borrow this module for a quick implementation of fully connected network.
    Input: [batch_size, channels, (H, W)]
    Output: [batch_size, channels, (H, W1)]
    In reality, input dim is: [batch_size, T, N, D]
    Look at the forward function, D is permuted to the "channels" for a fully connected operation using kernels...
    """
    def __init__(
        self, input_dims, output_dims, 
        kernel_size, stride=(1, 1), padding='SAME', 
        use_bias=True, activation=F.relu, bn_decay=None
    ):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            ks = math.ceil(kernel_size)
            self.padding_size = [ks//2, ks//2]
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(
            input_dims, output_dims, 
            kernel_size, stride=stride, padding=0, bias=use_bias
        )
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    """
    Run a set of conv2d_ functions to use the kernel trick to perform a fully connected network over the last dimension.
    """
    def __init__(
        self, 
        input_dims: int | tuple | list, 
        units: int | tuple | list, 
        activations: nn.Module | tuple[nn.Module] | list[nn.Module], 
        bn_decay: float, 
        use_bias: bool = True
    ) -> None:
        super(FC, self).__init__()
        # type conversion to list
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, 
            output_dims=num_unit, 
            kernel_size=[1, 1], stride=[1, 1], padding='VALID',
            use_bias=use_bias, activation=activation, bn_decay=bn_decay) 
            for input_dim, num_unit, activation 
            in zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    """
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, T, 2]  # dayofweek, timeofday
    T_day:  num of time steps in one day 
        - for hourly data, T_day=24
        - for 15min data, T_day=96
        - for 5min data, T_day=288
    D:      output dims
    return: [batch_size, T, N, D]
    """
    def __init__(self, D_se, D_out, bn_decay, se_num_layers=2, T_day=24):
        super(STEmbedding, self).__init__()
        # three layers of FC to merge different spatial embeddings
        input_dims = [D_se, D_out]
        units = [D_out, D_out]
        activations = [F.relu, None]
        if se_num_layers == 3:
            input_dims = [D_se, D_out, D_out]
            units = [D_out, D_out, D_out]
            activations = [F.tanh, F.relu, None]
        self.FC_se = FC(
            input_dims=input_dims, units=units, 
            activations=activations,bn_decay=bn_decay)
        self.T_day = T_day
        # input_dims = time step per day + days per week=24+7=31
        self.FC_te = FC(
            input_dims=[31, D_out], units=[D_out, D_out], 
            activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, SE, TE):
        T_day = self.T_day
        device = next(self.parameters()).device
        # spatial embedding to [1, 1, N, D]
        SE = SE.unsqueeze(0).unsqueeze(0)
        # SE: [1, 1, N, D]
        SE = self.FC_se(SE)
        # TE: [BS, T, 2]
        # temporal embedding (one hot vector)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7, device=device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T_day, device=device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T_day, T_day)
        # TE: [BS, T, 31]  # 31 = 7 + 24
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        # TE: [BS, T, 1, 31]
        TE = TE.unsqueeze(dim=2)
        # TE: [BS, T, 1, D]
        TE = self.FC_te(TE)
        del dayofweek, timeofday
        # combine SE and TE for spatial-temporal embedding
        STE = SE + TE
        return STE


class spatialAttention(nn.Module):
    """
    spatial attention mechanism
    X:      [BS, T, N, D]
    STE:    [BS, T, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [BS, T, N, D]
    """
    def __init__(self, K: int, d: int, bn_decay: float, mask: bool = False, pivotal_nodes: list[int] | None = None):
        super(spatialAttention, self).__init__()
        D = K * d
        self.K, self.d = K, d
        # self detector mask (only use the other detectors to detect the target detector)
        self.mask = mask
        self.pivotal_nodes = pivotal_nodes
        # input dims is 2D because data and STE both with shapes D are connected together
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        # option to record attention
        self.attention_score = None

    def forward(self, X, STE, get_attention: bool = False):
        device = next(self.parameters()).device
        batch_size = X.shape[0]
        # [BS, T, N, 2D]
        X = torch.cat((X, STE), dim=-1)
        # [BS, T, N, K * d = D]
        query, key, value = self.FC_q(X), self.FC_k(X), self.FC_v(X)
        # between different heads, the forward operations are in parallel,
        # just like the case for different batch sizes...
        # [K * BS, T, N, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # [K * BS, T, N, N]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        # mask attention score (mask to current station id)
        if self.mask:
            # X: [BS, T, N, 2D]
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            # [N, N]
            mask = 1 - torch.eye(num_vertex, device=device)
            # [1, 1, N, N]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            # [K * BS, T, N, N]
            mask = mask.repeat(self.K * batch_size, num_step, 1, 1)
            mask = mask.to(torch.bool)
            # set attention to the minimal score (-inf)
            attention = torch.where(mask, attention, SMALLEST_FLOAT)
        # only pivotal nodes to detect for all nodes (that is, all nodes in the training set)
        if self.pivotal_nodes is not None:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.zeros(num_vertex, num_vertex, device=device)
            mask[:, self.pivotal_nodes] = 1
            # trick: only zero will be converted to False; 
            # all positive values will be converted to True
            if not self.mask:
                mask = mask + torch.eye(num_vertex, device=device)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_step, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, SMALLEST_FLOAT)
        
        attention = F.softmax(attention, dim=-1)
        # [K * BS, T, N, d]
        X = torch.matmul(attention, value)
        # [BS, T, N, K * d = D]
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        if get_attention:
            # record attention to visualize results
            self.attention_score = attention.detach().cpu().numpy()
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    """
    temporal attention mechanism
    X:      [BS, T, N, D]
    STE:    [BS, T, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [BS, T, N, D]
    """
    def __init__(self, K: int, d: int, bn_decay: float, mask: bool = True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.K, self.d, self.mask = K, d, mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        # option for recording attention to visualize attention results...
        self.attention_score = None

    def forward(self, X, STE, get_attention: bool = False):
        batch_size_ = X.shape[0]
        # [BS, T, N, 2D]
        X = torch.cat((X, STE), dim=-1)
        # [BS, T, N, K * d = D]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # print("query.shape", query.shape)
        # print("key.shape", key.shape)
        # print("K", self.K)
        # [K * BS, T, N, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # query: [K * BS, N, T, d]
        # key:   [K * BS, N, d, T]
        # value: [K * BS, N, T, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * BS, N, T, T]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score (only use past to predict the future)
        if self.mask:
            # X: [BS, T, N, 2D]
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            # [T, T]
            mask = torch.ones(num_step, num_step, device=X.device)
            mask = torch.tril(mask)
            # [1, 1, T, T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            # [K * BS, N, T, T]
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, SMALLEST_FLOAT)
        # softmax of attention scores
        # [K * BS, N, T, T]  # last dim's values sum up to 1
        attention = F.softmax(attention, dim=-1)
        if get_attention:
            # record attention to visualize results
            self.attention_score = attention.detach().cpu().numpy()
        # [K * BS, N, T, d]
        X = torch.matmul(attention, value)
        # [K * BS, T, N, d]
        X = X.permute(0, 2, 1, 3)
        # [BS, T, N, K * d = D]
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)
        # [BS, T, N, D]
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    """gated fusion to combine spatial and temporal features into a single feature
    HS:     [BS, T, N, D]
    HT:     [BS, T, N, D]
    D:      output dims
    return: [BS, T, N, D]
    """
    def __init__(self, D: int, bn_decay: float):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K: int, d: int, bn_decay: float,
                 spatial_mask: bool = False, 
                 temporal_mask: bool = True):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay, mask=spatial_mask)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=temporal_mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        # X: [BS, T, N, D]
        # STE: [BS, T, N, D]
        # HS/HT/H: [BS, T, N, D]
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):
    """
    transform attention mechanism (use history to predict future)
    X:        [BS, T_his, N, D]
    STE_his:  [BS, T_his, N, D]
    STE_pred: [BS, T_pred, N, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [BS, T_pred, N, D]
    """
    def __init__(self, K: int, d: int, bn_decay: float):
        super(transformAttention, self).__init__()
        D = K * d
        self.K, self.d = K, d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        # X: [BS, T_his, N, D]
        # STE_his: [BS, T_his, N, D]
        # STE_pred: [BS, T_pred, N, D]
        BS = X.shape[0]
        # [BS, T, N, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * BS, T, N, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # query: [K * BS, N, T_pred, d]
        # key:   [K * BS, N, d, T_his]
        # value: [K * BS, N, T_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * BS, N, T_pred, T_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [BS, T_pred, N, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, BS, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    """GMAN
    X:      [BS, T_his, N]
    TE:     [BS, T_his + T_pred, 2] (time-of-day, day-of-week)
    SE:     [N, K * d]
    T_his:  number of history steps
    T_pred: number of prediction steps
    T:      one day is divided into T steps
    L:      number of STAtt blocks in the encoder/decoder
    K:      number of attention heads
    d:      dimension of each attention head outputs
    return: [BS, T_pred, N]
    """
    def __init__(self, SE, args, bn_decay: float):
        super(GMAN, self).__init__()
        L, K, d = args.L, args.K, args.d
        D = K * d
        self.num_his = args.num_his
        self.SE = SE
        self.STEmbedding = STEmbedding(D, D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, X, TE):
        # X: [BS, T_his, N]
        # TE: [BS, T_his + T_pred, 2] (time-of-day, day-of-week)
        X = torch.unsqueeze(X, -1)
        X = self.FC_1(X)
        # STE: [BS, T, N, D]
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        # encoder
        # X: [BS, T_his, N, D]
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        # transAtt
        # X: [BS, T_pred, N, D]
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        # X: [BS, T_pred, N, D]
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # output
        # X: [BS, T_pred, N, 1]
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        # X: [BS, T_pred, N]
        return torch.squeeze(X, 3)

