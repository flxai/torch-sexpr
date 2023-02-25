# torch-sexpr
Parse architectures, losses and optimizers in PyTorch from S-expressions.

## Usage

The following examples assume that you import the library as follows:

```python
import torch_sexpr as ts
```

### Architectures

To parse architectures use `parse_architecture`, e.g.:

```python
>>> s_expr_arch = "(Conv2d (in_channels 20) (out_channels 20) (kernel_size 5))"
>>> ts.parse_architecture(s_expr_arch)
```

The available layers for architectures are described in [PyTorch's documentation](https://pytorch.org/docs/stable/nn.html) and `ts.available_layers` respectively.

### Losses

To parse losses use `parse_loss`, e.g.:

```python
>>> s_expr_loss = "(MSELoss)"
>>> ts.parse_loss(s_expr_loss)
```

The available layers for architectures are described in [PyTorch's documentation](https://pytorch.org/docs/stable/nn.html#loss-functions) and `ts.available_losses` respectively.

### Optimizers

To parse optimizers use `parse_optimizer`, e.g.:

```python
>>> s_expr_optimizer = "(Adam (lr 1e-5))"
>>> ts.parse_optimizer(s_expr_optimizer)
```

The available optimizers for architectures are described in [PyTorch's documentation](https://pytorch.org/docs/stable/optim.html#algorithms) and `ts.available_optimizers` respectively.


## Installation

The project is hosted on PyPI, so you can install it using common tools.


## Primer on S-expressions

If you've never heard of S-expressions before, you might want to read [their Wikipedia page](https://en.wikipedia.org/wiki/S-expression) as a quick introduction.


## About

I built this library to quickly test out different architectures, losses and optimizers.
Using it in conjunction with CLIs seems like a good fit.
