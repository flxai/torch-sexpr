# torch-sexpr

Parse architectures, losses and optimizers in PyTorch from S-expressions and instantiate them accordingly.

## Usage

### As a library

The following examples assume that you import the library as follows:

```python
from torch_sexpr import SExprParser
```

#### Architectures

To parse architectures use `parse_architecture`, e.g.:

```python
>>> sexpr_arch = "(Conv2d (in_channels 20) (out_channels 20) (kernel_size 5))"
>>> sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
>>> sexpr_parser.parse_arch_sexpr(sexpr_arch)
```

The available layers for architectures are described in [PyTorch's documentation](https://pytorch.org/docs/stable/nn.html) and `ts.available_layers` respectively.

#### Losses

To parse losses use `parse_loss`, e.g.:

```python
>>> s_expr_loss = "(MSELoss)"
>>> sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
>>> sexpr_parser.parse_loss_sexpr(sexpr_arch)
```

The available layers for architectures are described in [PyTorch's documentation](https://pytorch.org/docs/stable/nn.html#loss-functions) and `ts.available_losses` respectively.
Please also see [additional available losses from TorchMetrics](https://github.com/Lightning-AI/metrics).

#### Optimizers

To parse optimizers use `parse_optimizer`, e.g.:

```python
>>> sexpr_optim = "(Adam (lr 1e-5))"
>>> sexpr_parser = SExprParser(fail_summary=True, fail_list=True)
>>> sexpr_parser.parse_loss_sexpr(sexpr_optim)
```

The available optimizers for architectures are described in [PyTorch's documentation](https://pytorch.org/docs/stable/optim.html#algorithms) and `ts.available_optimizers` respectively.

### Command line interface

This library comes bundled with CLIs that allow for easy use from within the shell.

#### Architectures

You can list all available layers for architectures using:

```console
$ torch-sexpr-list-layers
```

You can parse an architecture serialized as an S-expression using:

```console
$ torch-sexpr-parse-architecture "(Conv2d (in_channels 20) (out_channels 20) (kernel_size 5))"
```

#### Losses

You can list all available layers for architectures using:

```console
$ torch-sexpr-list-losses
```

You can parse an architecture serialized as an S-expression using:

```console
$ torch-sexpr-parse-loss "(MSELoss)"
```

#### Optimizers

You can list all available layers for architectures using:

```console
$ torch-sexpr-list-optimizers
```

You can parse an architecture serialized as an S-expression using:

```console
$ torch-sexpr-parse-optimizer "(Adam (lr 1e-5))"
```

## Installation

The project is hosted on PyPI, so you can install it using common tools.

## Primer on S-expressions

If you've never heard of S-expressions before, you might want to read [their Wikipedia page](https://en.wikipedia.org/wiki/S-expression) as a quick introduction.


## About

I built this library to quickly test out different architectures, losses and optimizers.
Using it in conjunction with CLIs seems like a good fit.
