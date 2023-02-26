import argparse
import click
import functools
import hashlib
import logging
import torch
import torchmetrics
import sexpdata
import sys

# PyTorch optimizers
# cf. https://pytorch.org/docs/stable/optim.html#algorithms
# curl -Ss "https://pytorch.org/docs/stable/_sources/optim.rst.txt" | tac | tac | awk '$0=="Algorithms" { s1=1 } $0=="How to adjust learning rate" { s2=1 } s2==1 && $0=="" { exit } s1==1 && $0 ~ /^ +[[:upper:]]/ { print $1 }' | sed -r 's/(.+)/    "\1",/' | sort
_TORCH_OPTIMS = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
]

# PyTorch loss functions
# cf. https://pytorch.org/docs/stable/nn.html#loss-functions
# curl -Ss "https://pytorch.org/docs/stable/_sources/nn.rst.txt" | tac | tac | awk '$0=="Loss Functions" { s1=1 } $0=="Vision Layers" { s2=1 } s2==1 && $0=="" { exit } s1==1 && $0 ~ /^ +nn\./ { print $1 }' | sed -r 's/nn\.//;s/(.+)/    "\1",/' | sort
_TORCH_LOSSES = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CosineEmbeddingLoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MarginRankingLoss",
    "MSELoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

# TorchMetrics 
# cf. https://github.com/Lightning-AI/metrics
# git clone --depth=1 "https://github.com/Lightning-AI/metrics.git" &> /dev/null && grep -hrI "autoclass:: torchmetrics" | awk '{ print $3 }' | sort | uniq | sed -r 's/torchmetrics\.//;s/(.+)/    "\1",/' | sort
_TORCHMETRICS = [
    "Accuracy",
    "audio.pesq.PerceptualEvaluationSpeechQuality",
    "audio.stoi.ShortTimeObjectiveIntelligibility",
    "AUROC",
    "AveragePrecision",
    "BLEUScore",
    "BootStrapper",
    "CalibrationError",
    "CatMetric",
    "CharErrorRate",
    "CHRFScore",
    "classification.BinaryAccuracy",
    "classification.BinaryAUROC",
    "classification.BinaryAveragePrecision",
    "classification.BinaryCalibrationError",
    "classification.BinaryCohenKappa",
    "classification.BinaryConfusionMatrix",
    "classification.BinaryF1Score",
    "classification.BinaryFBetaScore",
    "classification.BinaryHammingDistance",
    "classification.BinaryHingeLoss",
    "classification.BinaryJaccardIndex",
    "classification.BinaryMatthewsCorrCoef",
    "classification.BinaryPrecision",
    "classification.BinaryPrecisionRecallCurve",
    "classification.BinaryRecall",
    "classification.BinaryRecallAtFixedPrecision",
    "classification.BinaryROC",
    "classification.BinarySpecificity",
    "classification.BinarySpecificityAtSensitivity",
    "classification.BinaryStatScores",
    "classification.MulticlassAccuracy",
    "classification.MulticlassAUROC",
    "classification.MulticlassAveragePrecision",
    "classification.MulticlassCalibrationError",
    "classification.MulticlassCohenKappa",
    "classification.MulticlassConfusionMatrix",
    "classification.MulticlassExactMatch",
    "classification.MulticlassF1Score",
    "classification.MulticlassFBetaScore",
    "classification.MulticlassHammingDistance",
    "classification.MulticlassHingeLoss",
    "classification.MulticlassJaccardIndex",
    "classification.MulticlassMatthewsCorrCoef",
    "classification.MulticlassPrecision",
    "classification.MulticlassPrecisionRecallCurve",
    "classification.MulticlassRecall",
    "classification.MulticlassRecallAtFixedPrecision",
    "classification.MulticlassROC",
    "classification.MulticlassSpecificity",
    "classification.MulticlassSpecificityAtSensitivity",
    "classification.MulticlassStatScores",
    "classification.MultilabelAccuracy",
    "classification.MultilabelAUROC",
    "classification.MultilabelAveragePrecision",
    "classification.MultilabelConfusionMatrix",
    "classification.MultilabelCoverageError",
    "classification.MultilabelExactMatch",
    "classification.MultilabelF1Score",
    "classification.MultilabelFBetaScore",
    "classification.MultilabelHammingDistance",
    "classification.MultilabelJaccardIndex",
    "classification.MultilabelMatthewsCorrCoef",
    "classification.MultilabelPrecision",
    "classification.MultilabelPrecisionRecallCurve",
    "classification.MultilabelRankingAveragePrecision",
    "classification.MultilabelRankingLoss",
    "classification.MultilabelRecall",
    "classification.MultilabelRecallAtFixedPrecision",
    "classification.MultilabelROC",
    "classification.MultilabelSpecificity",
    "classification.MultilabelSpecificityAtSensitivity",
    "classification.MultilabelStatScores",
    "ClasswiseWrapper",
    "CohenKappa",
    "ConcordanceCorrCoef",
    "ConfusionMatrix",
    "CosineSimilarity",
    "CramersV",
    "detection.mean_ap.MeanAveragePrecision",
    "Dice",
    "ExactMatch",
    "ExplainedVariance",
    "ExtendedEditDistance",
    "F1Score",
    "FBetaScore",
    "HammingDistance",
    "HingeLoss",
    "image.ergas.ErrorRelativeGlobalDimensionlessSynthesis",
    "image.fid.FrechetInceptionDistance",
    "image.inception.InceptionScore",
    "image.kid.KernelInceptionDistance",
    "image.lpip.LearnedPerceptualImagePatchSimilarity",
    "JaccardIndex",
    "KendallRankCorrCoef",
    "KLDivergence",
    "LogCoshError",
    "MatchErrorRate",
    "MatthewsCorrCoef",
    "MaxMetric",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanMetric",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinkowskiDistance",
    "MinMaxMetric",
    "MinMetric",
    "multimodal.clip_score.CLIPScore",
    "MultioutputWrapper",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PanopticQuality",
    "PeakSignalNoiseRatio",
    "PearsonCorrCoef",
    "PearsonsContingencyCoefficient",
    "PermutationInvariantTraining",
    "Precision",
    "PrecisionRecallCurve",
    "R2Score",
    "Recall",
    "RetrievalFallOut",
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalPrecisionRecallCurve",
    "RetrievalRecall",
    "RetrievalRPrecision",
    "ROC",
    "SacreBLEUScore",
    "ScaleInvariantSignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SignalDistortionRatio",
    "SignalNoiseRatio",
    "SpearmanCorrCoef",
    "Specificity",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "SQuAD",
    "StatScores",
    "StructuralSimilarityIndexMeasure",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "text.bert.BERTScore",
    "text.infolm.InfoLM",
    "text.perplexity.Perplexity",
    "text.rouge.ROUGEScore",
    "TheilsU",
    "TotalVariation",
    "TranslationEditRate",
    "TschuprowsT",
    "TweedieDevianceScore",
    "UniversalImageQualityIndex",
    "WeightedMeanAbsolutePercentageError",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]

# PyTorch loss functions
# cf. https://pytorch.org/docs/stable/nn.html#loss-functions
# Containing the following categories
#  - Convolution Layers
#  - Pooling layers
#  - Padding Layers
#  - Non-linear Activations (weighted sum, nonlinearity)
#  - Non-linear Activations (other)
#  - Normalization Layers
#  - Group Normalization
#  - Recurrent Layers
#  - Transformer Layers
#  - Linear Layers
#  - Dropout Layers
#  - Sparse Layers
#  - Distance Functions
#  - Vision Layers
#  - Shuffle Layers
#  - DataParallel Layers (multi-GPU, distributed)
#  - Utility Functions
# curl -Ss "https://pytorch.org/docs/stable/_sources/nn.rst.txt" | tac | tac | awk '$0=="Loss Functions" { l=1 } $0=="Vision Layers" { l=0 } $0=="Convolution Layers" { s1=1 } $0=="Utilities" { s2=1 } s2==1 && $0=="" { exit } l==0 && s1==1 && $0 ~ /^ +nn\./ { print $1 }' | sed -r 's/nn\.//;s/(.+)/    "\1",/' | sort
_TORCH_LAYERS = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveLogSoftmaxWithLoss",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AlphaDropout",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CELU",
    "ChannelShuffle",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CosineSimilarity",
    "DataParallel",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "FeatureAlphaDropout",
    "Fold",
    "Flatten",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "GELU",
    "GLU",
    "GroupNorm",
    "GRU",
    "GRUCell",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LazyLinear",
    "LeakyReLU",
    "Linear",
    "LocalResponseNorm",
    "LogSigmoid",
    "LogSoftmax",
    "LPPool1d",
    "LPPool2d",
    "LSTM",
    "LSTMCell",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "MultiheadAttention",
    "PairwiseDistance",
    "parallel.DistributedDataParallel",
    "PixelShuffle",
    "PixelUnshuffle",
    "PReLU",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReLU",
    "ReLU6",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RReLU",
    "SELU",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SyncBatchNorm",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad2d",
]

class SExprParser:
    def __init__(self, fail_summary=False, fail_list=False):
        self.fail_summary = fail_summary
        self.fail_list = fail_list
        self.optims = None
        self.losses = None
        self.layers = None
        self.fails_optims = None
        self.fails_losses = None
        self.fails_layers = None
        self.initial_fail = True

    @staticmethod
    def _getattr_recursive(path):
        return 

    def _extrapolate_module_keys(self, keys, base_module):
        ret = {}
        fails = {}
        for key in keys:
            key_path = key.split('.')
            try:
                abs_path = [sys.modules[base_module]] + key_path
                ret[key] = functools.reduce(lambda a, b: getattr(a, b), abs_path)
            except AttributeError as e:
                name = e.obj.__name__
                if name in fails:
                    fails[name] += [e.name]
                else:
                    fails[name] = []
        return ret, fails

    def report_fails(self, fails):
        if self.fail_summary and len(fails):
            if self.initial_fail:
                print("The following methods could not be imported (your PyTorch/TorchMetrics version might be too old:")
            for fail, failed_names in fails.items():
                count = len(failed_names)
                print(f"Failed imports for '{fail}': {count}")
                for name in failed_names:
                    print(f"  {name}")

    def _load_optims(self):
        if not self.optims:
            self.optims, fails_optims = self._extrapolate_module_keys(_TORCH_OPTIMS, 'torch.optim')
            self.fails_optims = fails_optims
            self.report_fails(self.fails_optims)

    def _load_losses(self):
        if not self.losses:
            losses_torch, fails_losses_torch = self._extrapolate_module_keys(_TORCH_LOSSES, 'torch.nn')
            losses_torchmetrics, fails_losses_torchmetrics = \
                    self._extrapolate_module_keys(_TORCHMETRICS, 'torchmetrics')
            self.losses = losses_torch | losses_torchmetrics
            self.fails_losses = fails_losses_torch | fails_losses_torchmetrics
            self.report_fails(self.fails_losses)

    def _load_layers(self):
        if not self.layers:
            self.layers, fails_layers = self._extrapolate_module_keys(_TORCH_LAYERS, 'torch.nn')
            self.fails_layers = fails_layers
            self.report_fails(self.fails_layers)

    def parse_loss_sexpr(self, sexpr_loss):
        self._load_losses()
        sexp = sexpdata.loads(sexpr_loss)
        assert isinstance(sexp, list), f"ERROR: Must define loss as sexpr list."
        params = {}
        loss_last = None
        for s in sexp:
            assert isinstance(s, sexpdata.Symbol) or isinstance(s, list), \
                "ERROR: Expecting parameter to be either sexpdata.Symbol or list."
            if isinstance(s, sexpdata.Symbol):
                v = s.value()
                assert v in self.losses, f"ERROR: Loss type '{v}' is not allowed, for a list of options see: https://pytorch.org/docs/stable/nn.html#loss-functions"
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
        loss = self.losses[loss_last](**params)
        return loss

    def parse_optimizer_sexpr(self, sexpr_optimizer, net=None):
        self._load_optims()
        sexp = sexpdata.loads(sexpr_optimizer)
        assert isinstance(sexp, list), "ERROR: Must define optimizer as sexpr list."
        params = {}
        optim_last = None
        for s in sexp:
            assert isinstance(s, sexpdata.Symbol) or isinstance(s, list), \
                "ERROR: Expecting parameter to be either sexpdata.Symbol or list."
            if isinstance(s, sexpdata.Symbol):
                v = s.value()
                assert v in self.optims, f"ERROR: Optimizer type '{v}' is not allowed, for a list of options see: https://pytorch.org/docs/stable/optim.html#algorithms"
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
            optim = self.optims[optim_last](net.parameters(), **params)
        # Otherwise give back a bogus optimizer and just throw no exception if parsed successfully
        else:
            optim = self.optims[optim_last]([torch.empty(1)], **params)
        return optim

    def parse_architecture_sexpr(self, sexpr_architecture, typecast_list=True):
        self._load_layers()
        sexp = sexpdata.loads(sexpr_architecture)
        assert isinstance(sexp, list), "ERROR: Must define layers as sexpr list."
        layers = []
        params = {}
        layer_last = None
        for s in sexp:
            assert isinstance(s, sexpdata.Symbol) or isinstance(s, list), \
                "ERROR: Expecting parameter to be either sexpdata.Symbol or list."
            if isinstance(s, sexpdata.Symbol):
                v = s.value()
                assert v in self.layers, f"ERROR: Layer type '{v}' is not allowed, for a list of options see: https://pytorch.org/docs/stable/nn.html"
                # Add last layer with collected properties
                if layer_last is not None:
                    # if len(layers) == 0 and implicit_shapes:
                    #     # TODO Allow for fixed/implicit out_features
                    #     print(layer_last)
                    layers.append(self.layers[layer_last](**params))
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
        layers.append(self.layers[layer_last](**params))
        net = torch.nn.Sequential(*layers)
        return net

    @property
    def failed_count_optims(self):
        return 42

    @property
    def failed_count_losses(self):
        return 42

    @property
    def failed_count_layers(self):
        return 42
        # self.failed_modules_optims = None
        # self.failed_modules_losses = None
        # self.failed_modules_layers = None


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
    try:
        sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
        print(sexpr_parser.parse_loss_sexpr(sexpr_loss))
    except Exception as e:
        print(f"Could not parse optimizer: {e}")


@click.command()
@click.argument("sexpr_optimizer")
def print_parsed_optimizer_sexpr(sexpr_optimizer):
    try:
        sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
        print(sexpr_parser.parse_optimizer_sexpr(sexpr_optimizer))
    except Exception as e:
        print(f"Could not parse optimizer: {e}")


@click.command()
@click.argument("sexpr_architecture")
def print_parsed_architecture_sexpr(sexpr_architecture):
    try:
        sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
        print(sexpr_parser.parse_architecture_sexpr(sexpr_architecture))
    except Exception as e:
        print(f"Could not parse optimizer: {e}")

@click.command()
def print_available_losses():
    sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
    sexpr_parser._load_losses()
    for loss in sexpr_parser.losses:
        print(loss)

@click.command()
def print_available_optimizers():
    sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
    sexpr_parser._load_optims()
    for optimizer in sexpr_parser.optims:
        print(optimizer)

@click.command()
def print_available_layers():
    sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
    sexpr_parser._load_layers()
    for layer in sexpr_parser.layers:
        print(layer)
