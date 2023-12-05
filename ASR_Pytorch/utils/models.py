import numpy as np
import torch as t
import torch.nn.functional as F
from .functions import safe_log, square, Expression


class EEGNet(t.nn.Module):
   
    def __init__(self, F1=8, D=2, F2=None, C=64, T=480, N=3, p_dropout=0.2, activation='relu'):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        activation:   string, either 'elu' or 'relu'
        """
        super(EEGNet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']

        # Prepare Dropout Type
        dropout = t.nn.Dropout

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.activation = (p_dropout, activation)

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # Block 1
        if C==64:
            self.conv1_pad = t.nn.ZeroPad2d((63, 64, 0, 0)) #PhysioNet
            self.conv1 = t.nn.Conv2d(1, F1, (1, 128), bias=False) # PhysioNet
        else:
            self.conv1_pad = t.nn.ZeroPad2d((31, 32, 0, 0))
            self.conv1 = t.nn.Conv2d(1, F1, (1, 64), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv2 = t.nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool1 = t.nn.AvgPool2d((1, 8))
        # self.dropout1 = dropout(p=p_dropout)
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.flatten = Flatten()
        self.fc = t.nn.Linear(F2 * n_features, N, bias=True)

        # initialize weights
        self._initialize_params()

    def forward(self, x, with_stats=False):

        # input dimensions: (s, 1, C, T)
        if x.shape[-1]==1124:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # Block 1
        x = self.conv1_pad(x)
        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        x = self.batch_norm1(x)
        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)            # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)            # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)

        # Classification
        x = self.flatten(x)          # output dim: (s, F2 * (T // 64))
        x = self.fc(x)               # output dim: (s, N)

        if with_stats:
            stats = [('conv1_w', self.conv1.weight.data),
                     ('conv2_w', self.conv2.weight.data),
                     ('sep_conv1_w', self.sep_conv1.weight.data),
                     ('sep_conv2_w', self.sep_conv2.weight.data),
                     ('fc_w', self.fc.weight.data),
                     ('fc_b', self.fc.bias.data)]
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        return self.forward(x, with_stats=True)

    def _initialize_params(self, weight_init=t.nn.init.xavier_uniform_, bias_init=t.nn.init.zeros_):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: t.nn.init inplace function
         - bias_init:   t.nn.init inplace function

        """
        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                weight_init(m.weight)
            if isinstance(m, t.nn.Linear):
                bias_init(m.bias)

        self.apply(init_weight)
    def is_cuda(self):
        is_cuda = False
        for param in self.parameters():
            if param.is_cuda:
                is_cuda = True
        return is_cuda

class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class EEGTCNet(t.nn.Module):
    """
    EEGTCNet

    In order for the baseline to be launched with the same logic as the quantized models, an empty
    quantization scheme and an empty thermostat schedule needs to be configured.
    Use the following templates for the 'net' and 'thermostat' configurations (for the "net" object,
    all params can be omitted to use the default ones):

    "net": {
      "class": "EEGTCNetBaseline",
      "params": {
        "F1": 8,
        "D": 2,
        "F2": 16,
        "C": 22,
        "T": 1125,
        "N": 4,
        "p_dropout": 0.5,
        "activation": "relu",
        "dropout_type": "TimeDropout2D",
      },
      "pretrained": null,
      "loss_fn": {
        "class": "CrossEntropyLoss",
        "params": {}
      }
    }

    "thermostat": {
      "class": "EEGTCNetBaseline",
      "params": {
        "noise_schemes": {},
        "bindings": []
      }
    }
    """

    def __init__(self, F1=8, D=2, F2=None, C=22, T=1125, N=4, activation='relu',
                 Kt = 4, pt = 0.2, Ft = 17, p_dropout=0.5, dropout_type='TimeDropout2D'):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        activation:   string, either 'elu' or 'relu'
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        """
        super(EEGTCNet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = t.nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = t.nn.Dropout2d
        elif dropout_type.lower() == 'timedropout2d':
            dropout = TimeDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.activation = (p_dropout, activation)

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # Block 1
        self.conv1_pad = t.nn.ZeroPad2d((31, 32, 0, 0))
        self.conv1 = t.nn.Conv2d(1, F1, (1, 64), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv2 = t.nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool1 = t.nn.AvgPool2d((1, 8))
        # self.dropout1 = dropout(p=p_dropout)
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.dropout2 = dropout(p=p_dropout)

        # Now the TCN layer
        # First block
        dilation = 1
        self.tcn_upsample = t.nn.Conv1d(in_channels = F2, out_channels = Ft, kernel_size = 1)
        self.tcn_pad1 = t.nn.ConstantPad1d(padding = ((Kt-1) * dilation, 0), value = 0)
        self.tcn_conv1 = t.nn.Conv1d(in_channels = F2, out_channels = Ft, kernel_size = Kt, dilation = dilation)
        self.tcn_batchnorm4 = t.nn.BatchNorm1d(num_features = Ft)
        self.tcn_elu3 = t.nn.ELU()
        self.tcn_dropout3 = t.nn.Dropout(p = pt)
        self.tcn_pad2 = t.nn.ConstantPad1d(padding = ((Kt-1)*dilation, 0), value = 0)
        self.tcn_conv2 = t.nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation)
        self.tcn_batchnorm5 = t.nn.BatchNorm1d(num_features = Ft)
        self.tcn_elu4 = t.nn.ELU()
        self.tcn_dropout4 = t.nn.Dropout(p = pt)
        self.tcn_elu5 = t.nn.ELU()

        # Second block
        dilation = 2
        self.tcn_pad3 = t.nn.ConstantPad1d(padding = ((Kt-1) * dilation, 0), value = 0)
        self.tcn_conv3 = t.nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation)
        self.tcn_batchnorm6 = t.nn.BatchNorm1d(num_features = Ft)
        self.tcn_elu6 = t.nn.ELU()
        self.tcn_dropout5 = t.nn.Dropout(p = pt)
        self.tcn_pad4 = t.nn.ConstantPad1d(padding = ((Kt-1)*dilation,0), value = 0)
        self.tcn_conv4 = t.nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation)
        self.tcn_batchnorm7 = t.nn.BatchNorm1d(num_features = Ft)
        self.tcn_elu7 = t.nn.ELU()
        self.tcn_dropout6 = t.nn.Dropout(p = pt)
        self.tcn_elu8 = t.nn.ELU()

        # Last layer
        self.fc = t.nn.Linear(in_features = Ft, out_features = 4)

        # initialize weights
        self._initialize_params()

    def forward(self, x, with_stats=False):

        # input dimensions: (s, 1, C, T)
        if x.shape[-1]==1124:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2]) #

        # Block 1
        x = self.conv1_pad(x)
        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        x = self.batch_norm1(x)
        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)            # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)            # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)

        # TCN
        # First block
        x = x[:,:,-1,:] #
        res = self.tcn_upsample(x)
        x = self.tcn_pad1(x)
        x = self.tcn_conv1(x)
        x = self.tcn_batchnorm4(x)
        x = self.tcn_elu3(x)
        x = self.tcn_dropout3(x)
        x = self.tcn_pad2(x)
        x = self.tcn_conv2(x)
        x = self.tcn_batchnorm5(x)
        x = self.tcn_elu4(x)
        x = self.tcn_dropout4(x)
        x = self.tcn_elu5(x + res)

        # Second block
        res = self.tcn_pad3(x)
        res = self.tcn_conv3(res)
        res = self.tcn_batchnorm6(res)
        res = self.tcn_elu6(res)
        res = self.tcn_dropout5(res)
        res = self.tcn_pad4(res)
        res = self.tcn_conv4(res)
        res = self.tcn_batchnorm7(res)
        res = self.tcn_elu7(res)
        res = self.tcn_dropout6(res)
        x = self.tcn_elu8(res + x)
        
        # Linear layer to classify
        x = self.fc(x[:, :, -1])
        # x = F.log_softmax(x, dim=1)


        if with_stats:
            stats = [('conv1_w', self.conv1.weight.data),
                     ('conv2_w', self.conv2.weight.data),
                     ('sep_conv1_w', self.sep_conv1.weight.data),
                     ('sep_conv2_w', self.sep_conv2.weight.data),
                     ('tcn_conv1_w', self.tcn_conv1.weight.data),
                     ('tcn_conv2_w', self.tcn_conv2.weight.data),
                     ('tcn_conv3_w', self.tcn_conv3.weight.data),
                     ('tcn_conv4_w', self.tcn_conv4.weight.data),
                     ('fc_w', self.fc.weight.data),
                     ('fc_b', self.fc.bias.data)]
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        return self.forward(x, with_stats=True)

    def _initialize_params(self, weight_init=t.nn.init.xavier_uniform_, bias_init=t.nn.init.zeros_):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: t.nn.init inplace function
         - bias_init:   t.nn.init inplace function

        """
        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                weight_init(m.weight)
            if isinstance(m, t.nn.Linear):
                bias_init(m.bias)

        self.apply(init_weight)

    def is_cuda(self): 
        is_cuda = False
        for param in self.parameters():
            if param.is_cuda:
                is_cuda = True
        return is_cuda

class TimeDropout2d(t.nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """
    def __init__(self, p=0.5, inplace=False):
        """
        See t.nn.Dropout2d for parameters
        """
        super(TimeDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input


class ShallowConvNet(t.nn.Module):
    """
    EEGNet
    """
    def __init__(self, F1=40,filter_time_length=25,F2=None,C=22, T=1125, N=4, p_dropout=0, reg_rate=0.25,
                 activation='relu', constrain_w=False, dropout_type='TimeDropout2D',
                 permuted_flatten=False, n_features=2760):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        reg_rate:     Regularization (L1) of the final linear layer (fc)
                      This parameter is ignored when constrain_w is not asserted
        activation:   string, either 'elu' or 'relu'
        constrain_w:  bool, if True, constrain weights of spatial convolution and final fc-layer
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        permuted_flatten: bool, if True, use the permuted flatten to make the model keras compliant
        """
        super(ShallowConvNet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = t.nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = t.nn.Dropout2d
        elif dropout_type.lower() == 'timedropout2d':
            dropout = TimeDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.F2, self.C, self.T, self.N = (F1, F2, C, T, N)
        self.p_dropout, self.reg_rate, self.activation = (p_dropout, reg_rate, activation)
        self.constrain_w, self.dropout_type = (constrain_w, dropout_type)

        # Number of input neurons to the final fully connected layer
        # n_features = 2760

        # Block 1
        self.conv1 = t.nn.Conv2d(1, F1, (1, filter_time_length),bias=False)
        #self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)

        # Block 2: spatial convolution
        self.conv2 = t.nn.Conv2d(F1, F2, (C, 1), bias=True)
        #self.batch_norm2 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)

        self.activation1 = Expression(square)

        self.pool = t.nn.AvgPool2d((1, 75),stride=(1,15))
        
        self.activation2 = Expression(safe_log)

        self.dropout = dropout(p=p_dropout)

        self.flatten = t.nn.Flatten()

        self.fc = t.nn.Linear(n_features, N, bias=True)


       
        #self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        
    def forward(self, x):

        # reshape vector from (s, C, T) to (s, 1, C, T)
        if x.shape[-1]>=1124:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # input dimensions: (s, 1, C, T)

        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        #x = self.batch_norm1(x)

        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        #x = self.batch_norm2(x)

        x = self.activation1(x)
        x = self.pool(x)
        x = self.activation2(x)            # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout(x)

        # Classification
        x = self.flatten(x)          # output dim: (s, F2 * (T // 64))
        x = self.fc(x)               # output dim: (s, N)

        return x

    def initialize_params(self, weight_init=t.nn.init.xavier_uniform_, bias_init=t.nn.init.zeros_,
                          weight_gain=None, bias_gain=None):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: t.nn.init inplace function
         - bias_init:   t.nn.init inplace function
         - weight_gain: float, if None, don't use gain for weights
         - bias_gain:   float, if None, don't use gain for bias

        """
        # use gain only if xavier_uniform or xavier_normal is used
        weight_params = {}
        bias_params = {}
        if weight_gain is not None:
            weight_params['gain'] = weight_gain
        if bias_gain is not None:
            bias_params['gain'] = bias_gain

        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                weight_init(m.weight, **weight_params)
            if isinstance(m, t.nn.Linear):
                bias_init(m.bias, **bias_params)

        self.apply(init_weight)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

   

    def is_cuda(self):
        is_cuda = False
        for param in self.parameters():
            if param.is_cuda:
                is_cuda = True
        return is_cuda

class ConstrainedConv2d(t.nn.Conv2d):
    """
    Regularized Convolution, where the weights are clamped between two values.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', max_weight=1.0):
        """
        See t.nn.Conv2d for parameters.

        Parameters:
         - max_weight: float, all weights are clamped between -max_weight and max_weight
        """
        super(ConstrainedConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, groups=groups,
                                                bias=bias, padding_mode=padding_mode)
        self.max_weight = max_weight

    def forward(self, input):
        return self.conv2d_forward(input, self.weight.clamp(min=-self.max_weight,
                                                            max=self.max_weight))

class ConstrainedLinear(t.nn.Linear):
    """
    Regularized Linear Transformation, where the weights are clamped between two values
    """

    def __init__(self, in_features, out_features, bias=True, max_weight=1.0):
        """
        See t.nn.Linear for parameters

        Parameters:
         - max_weight: float, all weights are clamped between -max_weight and max_weight
        """
        super(ConstrainedLinear, self).__init__(in_features=in_features, out_features=out_features,
                                                bias=bias)
        self.max_weight = max_weight

    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=-self.max_weight, max=self.max_weight),
                        self.bias)





