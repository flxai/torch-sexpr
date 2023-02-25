import argparse
import click
import hashlib
import logging
import torch
import torchmetrics
import sys

import pyparsing as pp

# Basis for this LUT dict was the output of the following command:
# curl -Ss "https://pytorch.org/docs/stable/optim.html" | htmlq -tp 'h2 a, td p a'
# cf. https://pytorch.org/docs/stable/optim.html#algorithms
TORCH_OPTIMS = {
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SparseAdam": torch.optim.SparseAdam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "NAdam": torch.optim.NAdam,
    "RAdam": torch.optim.RAdam,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
}

TORCH_LOSSES = {
    # PyTorch loss functions
    # cf. https://pytorch.org/docs/stable/nn.html#loss-functions
    "L1Loss": torch.nn.L1Loss,
    "MSELoss": torch.nn.MSELoss,
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "CTCLoss": torch.nn.CTCLoss,
    "NLLLoss": torch.nn.NLLLoss,
    "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
    "GaussianNLLLoss": torch.nn.GaussianNLLLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
    "BCELoss": torch.nn.BCELoss,
    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
    "MarginRankingLoss": torch.nn.MarginRankingLoss,
    "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": torch.nn.MultiLabelMarginLoss,
    "HuberLoss": torch.nn.HuberLoss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "SoftMarginLoss": torch.nn.SoftMarginLoss,
    "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": torch.nn.CosineEmbeddingLoss,
    "MultiMarginLoss": torch.nn.MultiMarginLoss,
    "TripletMarginLoss": torch.nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": torch.nn.TripletMarginWithDistanceLoss,

    # torchmetrics loss functions
    "MeanSquaredLogError": torchmetrics.MeanSquaredLogError,
}

# Basis for this LUT dict was the output of the following command:
# curl -Ss "https://pytorch.org/docs/stable/nn.html" | htmlq -tp 'h2 a, td p a'
TORCH_LAYERS = {
    # Convolution Layers
    "Conv1d": torch.nn.Conv1d,
    "Conv2d": torch.nn.Conv2d,
    "Conv3d": torch.nn.Conv3d,
    "ConvTranspose1d": torch.nn.ConvTranspose1d,
    "ConvTranspose2d": torch.nn.ConvTranspose2d,
    "ConvTranspose3d": torch.nn.ConvTranspose3d,
    "LazyConv1d": torch.nn.LazyConv1d,
    "LazyConv2d": torch.nn.LazyConv2d,
    "LazyConv3d": torch.nn.LazyConv3d,
    "LazyConvTranspose1d": torch.nn.LazyConvTranspose1d,
    "LazyConvTranspose2d": torch.nn.LazyConvTranspose2d,
    "LazyConvTranspose3d": torch.nn.LazyConvTranspose3d,
    "Unfold": torch.nn.Unfold,
    "Fold": torch.nn.Fold,

    # Pooling layers
    "MaxPool1d": torch.nn.MaxPool1d,
    "MaxPool2d": torch.nn.MaxPool2d,
    "MaxPool3d": torch.nn.MaxPool3d,
    "MaxUnpool1d": torch.nn.MaxUnpool1d,
    "MaxUnpool2d": torch.nn.MaxUnpool2d,
    "MaxUnpool3d": torch.nn.MaxUnpool3d,
    "AvgPool1d": torch.nn.AvgPool1d,
    "AvgPool2d": torch.nn.AvgPool2d,
    "AvgPool3d": torch.nn.AvgPool3d,
    "FractionalMaxPool2d": torch.nn.FractionalMaxPool2d,
    "FractionalMaxPool3d": torch.nn.FractionalMaxPool3d,
    "LPPool1d": torch.nn.LPPool1d,
    "LPPool2d": torch.nn.LPPool2d,
    "AdaptiveMaxPool1d": torch.nn.AdaptiveMaxPool1d,
    "AdaptiveMaxPool2d": torch.nn.AdaptiveMaxPool2d,
    "AdaptiveMaxPool3d": torch.nn.AdaptiveMaxPool3d,
    "AdaptiveAvgPool1d": torch.nn.AdaptiveAvgPool1d,
    "AdaptiveAvgPool2d": torch.nn.AdaptiveAvgPool2d,
    "AdaptiveAvgPool3d": torch.nn.AdaptiveAvgPool3d,

    # Padding Layers
    "ReflectionPad1d": torch.nn.ReflectionPad1d,
    "ReflectionPad2d": torch.nn.ReflectionPad2d,
    "ReflectionPad3d": torch.nn.ReflectionPad3d,
    "ReplicationPad1d": torch.nn.ReplicationPad1d,
    "ReplicationPad2d": torch.nn.ReplicationPad2d,
    "ReplicationPad3d": torch.nn.ReplicationPad3d,
    "ZeroPad2d": torch.nn.ZeroPad2d,
    "ConstantPad1d": torch.nn.ConstantPad1d,
    "ConstantPad2d": torch.nn.ConstantPad2d,
    "ConstantPad3d": torch.nn.ConstantPad3d,

    # Non-linear Activations (weighted sum, nonlinearity)
    "ELU": torch.nn.ELU,
    "Hardshrink": torch.nn.Hardshrink,
    "Hardsigmoid": torch.nn.Hardsigmoid,
    "Hardtanh": torch.nn.Hardtanh,
    "Hardswish": torch.nn.Hardswish,
    "LeakyReLU": torch.nn.LeakyReLU,
    "LogSigmoid": torch.nn.LogSigmoid,
    "MultiheadAttention": torch.nn.MultiheadAttention,
    "PReLU": torch.nn.PReLU,
    "ReLU": torch.nn.ReLU,
    "ReLU6": torch.nn.ReLU6,
    "RReLU": torch.nn.RReLU,
    "SELU": torch.nn.SELU,
    "CELU": torch.nn.CELU,
    "GELU": torch.nn.GELU,
    "Sigmoid": torch.nn.Sigmoid,
    "SiLU": torch.nn.SiLU,
    "Mish": torch.nn.Mish,
    "Softplus": torch.nn.Softplus,
    "Softshrink": torch.nn.Softshrink,
    "Softsign": torch.nn.Softsign,
    "Tanh": torch.nn.Tanh,
    "Tanhshrink": torch.nn.Tanhshrink,
    "Threshold": torch.nn.Threshold,
    "GLU": torch.nn.GLU,

    # Non-linear Activations (other)
    "Softmin": torch.nn.Softmin,
    "Softmax": torch.nn.Softmax,
    "Softmax2d": torch.nn.Softmax2d,
    "LogSoftmax": torch.nn.LogSoftmax,
    "AdaptiveLogSoftmaxWithLoss": torch.nn.AdaptiveLogSoftmaxWithLoss,

    # Normalization Layers
    "BatchNorm1d": torch.nn.BatchNorm1d,
    "BatchNorm2d": torch.nn.BatchNorm2d,
    "BatchNorm3d": torch.nn.BatchNorm3d,
    "LazyBatchNorm1d": torch.nn.LazyBatchNorm1d,
    "LazyBatchNorm2d": torch.nn.LazyBatchNorm2d,
    "LazyBatchNorm3d": torch.nn.LazyBatchNorm3d,
    "GroupNorm": torch.nn.GroupNorm,

    # Group Normalization
    "SyncBatchNorm": torch.nn.SyncBatchNorm,
    "InstanceNorm1d": torch.nn.InstanceNorm1d,
    "InstanceNorm2d": torch.nn.InstanceNorm2d,
    "InstanceNorm3d": torch.nn.InstanceNorm3d,
    "LazyInstanceNorm1d": torch.nn.LazyInstanceNorm1d,
    "LazyInstanceNorm2d": torch.nn.LazyInstanceNorm2d,
    "LazyInstanceNorm3d": torch.nn.LazyInstanceNorm3d,
    "LayerNorm": torch.nn.LayerNorm,

    # Recurrent Layers
    "RNNBase": torch.nn.RNNBase,
    "RNN": torch.nn.RNN,
    "LSTM": torch.nn.LSTM,
    "GRU": torch.nn.GRU,
    "RNNCell": torch.nn.RNNCell,
    "LSTMCell": torch.nn.LSTMCell,
    "GRUCell": torch.nn.GRUCell,

    # Transformer Layers
    "Transformer": torch.nn.Transformer,
    "TransformerEncoder": torch.nn.TransformerEncoder,
    "TransformerDecoder": torch.nn.TransformerDecoder,
    "TransformerEncoderLayer": torch.nn.TransformerEncoderLayer,
    "TransformerDecoderLayer": torch.nn.TransformerDecoderLayer,

    # Linear Layers
    "Identity": torch.nn.Identity,
    "Linear": torch.nn.Linear,
    "Bilinear": torch.nn.Bilinear,
    "LazyLinear": torch.nn.LazyLinear,

    # Dropout Layers
    "Dropout": torch.nn.Dropout,
    # "Dropout1d": torch.nn.Dropout1d,
    "Dropout2d": torch.nn.Dropout2d,
    "Dropout3d": torch.nn.Dropout3d,
    "AlphaDropout": torch.nn.AlphaDropout,
    "FeatureAlphaDropout": torch.nn.FeatureAlphaDropout,

    # Sparse Layers
    "Embedding": torch.nn.Embedding,
    "EmbeddingBag": torch.nn.EmbeddingBag,

    # Distance Functions
    "CosineSimilarity": torch.nn.CosineSimilarity,
    "PairwiseDistance": torch.nn.PairwiseDistance,

    # Vision Layers
    "PixelShuffle": torch.nn.PixelShuffle,
    "PixelUnshuffle": torch.nn.PixelUnshuffle,
    "Upsample": torch.nn.Upsample,
    "UpsamplingNearest2d": torch.nn.UpsamplingNearest2d,
    "UpsamplingBilinear2d": torch.nn.UpsamplingBilinear2d,

    # Shuffle Layers
    "ChannelShuffle": torch.nn.ChannelShuffle,

    # DataParallel Layers (multi-GPU, distributed)
    "DataParallel": torch.nn.DataParallel,
    "parallel.DistributedDataParallel": torch.nn.parallel.DistributedDataParallel,

    # Utility Functions
    "Flatten": torch.nn.Flatten,
    "Unflatten": torch.nn.Unflatten,
}


def parse_loss_sexpr(sexpr_loss, losses_lut=TORCH_LOSSES):
    parser = pp.Forward()
    sexp = parser.parse_string(sexpr_loss)
    custom_losses_str = ', '.join([f"'{loss}'" for loss in custom_losses])
    assert isinstance(sexp, list), f"ERROR: Must define loss as sexpr list or use one of {custom_losses_str}."
    params = {}
    loss_last = None
    for s in sexp:
        assert isinstance(s, sexpdata.Symbol) or isinstance(s, list), \
            "ERROR: Expecting parameter to be either sexpdata.Symbol or list."
        if isinstance(s, sexpdata.Symbol):
            v = s.value()
            assert v in losses_lut, f"ERROR: Loss type '{v}' is not allowed, for a list of options see: https://pytorch.org/docs/stable/nn.html#loss-functions"
            assert loss_last is None, f"ERROR: Cannot set multiple loss functions."
            loss_last = v
        elif isinstance(s, list):
            assert len(s) == 2, f"ERROR: Key-value pair has unexpected length: {[_s.value() for _s in s]}"
            assert loss_last is not None, f"ERROR: Loss definition has to start with an loss, not its properties."
            k = s[0].value()
            v = s[1]
            if isinstance(v, sexpdata.Symbol):
                v = v.value()
            params[k] = v
    loss = losses_lut[loss_last](**params)
    return loss


def parse_optimizer_sexpr(sexpr_optimizer, optims_lut=TORCH_OPTIMS, net=None):
    parser = pp.Forward()
    sexp = parser.parse_string(sexpr_optimizer)
    assert isinstance(sexp, list), "ERROR: Must define optimizer as sexpr list."
    params = {}
    optim_last = None
    for s in sexp:
        assert isinstance(s, sexpdata.Symbol) or isinstance(s, list), \
            "ERROR: Expecting parameter to be either sexpdata.Symbol or list."
        if isinstance(s, sexpdata.Symbol):
            v = s.value()
            assert v in optims_lut, f"ERROR: Optimizer type '{v}' is not allowed, for a list of options see: https://pytorch.org/docs/stable/optim.html#algorithms"
            assert optim_last is None, f"ERROR: Cannot set multiple optimizers."
            optim_last = v
        elif isinstance(s, list):
            assert len(s) == 2, f"ERROR: Key-value pair has unexpected length: {[_s.value() for _s in s]}"
            assert optim_last is not None, f"ERROR: Optimizer definition has to start with an optimizer, not its properties."
            k = s[0].value()
            v = s[1]
            if isinstance(v, sexpdata.Symbol):
                v = v.value()
            params[k] = v
    # Apply optimizer to network's parameters only if given
    if net is not None:
        optim = optims_lut[optim_last](net.parameters(), **params)
    else:
        # This gives back a bogus optimizer and just throws no exception if parsed successfully
        optim = optims_lut[optim_last]([torch.empty(1)], **params)
    return optim


def parse_architecture_sexpr(sexpr_architecture, layers_lut=TORCH_LAYERS, in_shape=None, out_shape=None,
                             typecast_list=True, implicit_shapes=False):
    parser = pp.Forward()
    sexp = parser.parse_string(sexpr_architecture)
    assert isinstance(sexp, list), "ERROR: Must define layers as sexpr list."
    layers = []
    params = {}
    layer_last = None
    for s in sexp:
        assert isinstance(s, sexpdata.Symbol) or isinstance(s, list), \
            "ERROR: Expecting parameter to be either sexpdata.Symbol or list."
        if isinstance(s, sexpdata.Symbol):
            v = s.value()
            assert v in layers_lut, f"ERROR: Layer type '{v}' is not allowed, for a list of options see: https://pytorch.org/docs/stable/nn.html"
            # Add last layer with collected properties
            if layer_last is not None:
                # if len(layers) == 0 and implicit_shapes:
                #     # TODO Allow for fixed/implicit out_features
                #     print(layer_last)
                layers.append(layers_lut[layer_last](**params))
            # Flush params for next layer
            params = {}
            # Cache current layer to add properties
            layer_last = v
        elif isinstance(s, list):
            assert len(s) == 2, f"ERROR: Key-value pair has unexpected length: {[_s.value() for _s in s]}"
            assert layer_last is not None, f"ERROR: Architecture definition has to start with a layer, not its properties."
            k = s[0].value()
            v = s[1]
            if isinstance(v, sexpdata.Symbol):
                v = v.value()
            if isinstance(v, list) and typecast_list:
                v = tuple(v)
            # TODO Make 'size' a special dynamic word for in_features/out_features
            params[k] = v
    # Add last layer from cache
    # TODO Allow for fixed/implicit out_features
    layers.append(layers_lut[layer_last](**params))
    net = torch.nn.Sequential(*layers)
    return net


def get_short_class_name(obj):
    return str(obj.__class__).split('.')[-1].split("'")[0].lower()


def get_architecture_id(task, schema, preprocessing, net, optim):
    m = hashlib.sha256()
    m.update(task.encode('utf-8'))
    m.update(str(net).encode('utf-8'))
    net_hash = m.hexdigest()[:6]
    optim_name = get_short_class_name(optim)
    if isinstance(net, str):
        arch_breakdown = net
    else:
        arch_breakdown = '-'.join([get_short_class_name(layer) for layer in list(net.modules())[1:]])
    schema_str = schema.lower().replace('.', '').replace('/', '_')
    if preprocessing is None:
        preprocessing_str = 'none'
    else:
        preprocessing_str = '-'.join([p.lower() for p in preprocessing])
    architecture_id = f'{net_hash}-{task}-{schema_str}-{preprocessing_str}-{optim_name}-{arch_breakdown}'
    return architecture_id


@click.command()
@click.argument("sexpr_loss")
def print_parsed_loss_sexpr(sexpr_loss):
    print(parse_loss_sexpr(sexpr_loss))


@click.command()
@click.argument("sexpr_optimizer")
def print_parsed_optimizer_sexpr(sexpr_optimizer):
    print(parse_optimizer_sexpr(sexpr_optimizer))


@click.command()
@click.argument("sexpr_architecture")
def print_parsed_architecture_sexpr(sexpr_architecture):
    print(parse_architecture_sexpr(sexpr_architecture))
