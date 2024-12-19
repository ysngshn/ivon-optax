# A JAX implementation of IVON

This repo contains a [JAX](https://github.com/google/jax) implementation of the IVON optimizer from our paper

**Variational Learning is Effective for Large Deep Networks**\
*Y. Shen\*, N. Daheim\*, B. Cong, P. Nickl, G.M. Marconi, C. Bazan, R. Yokota, I. Gurevych, D. Cremers, M.E. Khan, T. Möllenhoff*\
International Conference on Machine Learning (ICML), 2024 **(spotlight)**

ArXiv: https://arxiv.org/abs/2402.17641 \
Blog: https://team-approx-bayes.github.io/blog/ivon/ \
Tutorial: https://ysngshn.github.io/research/why-ivon/

We also provide [an official PyTorch implementation](https://github.com/team-approx-bayes/ivon) of the IVON optimizer. Experiments in our paper are obtained with the PyTorch implementation and their source code can be found [here](https://github.com/team-approx-bayes/ivon-experiments).

The JAX implementation of IVON is self-contained in a single file [ivon.py](./ivon.py). The main optimizer `ivon.ivon(...)` is implemented as an [Optax](https://github.com/google-deepmind/optax) optimizer (alias of type `optax.GradientTransform`). Also, two functions `ivon.sample_parameters` and `ivon.accumulate_gradients` are provided to obtain posterior samples and accumulate intermediate results in case of multi-sample training, respectively.  

Special thanks to 
- Marco Miani for pointing out some bugs in the implementation.
- [Emanuel Sommer](https://github.com/EmanuelSommer) for raising the concern about the Optax/Flax compatibility issue.

The older legacy version with the same functionalities but different API can be found in the folder [resnet-cifar-legacy](./resnet-cifar-legacy) together with its CIFAR-10 training example.



## Quickstart

### Definition

IVON optimizer can be defined via `ivon.ivon()` and initialized with its `.init()` method.

```python
optimizer = ivon(
        lr, 
        ess, 
        hess_init, 
        beta1, 
        beta2, 
        weight_decay, 
        # ...
)
optstate = optimizer.init(params)
```

Appendix A of our [paper](https://arxiv.org/abs/2402.17641) provides practical guidelines for choosing IVON hyperparameters.

### Usage

IVON requests gradients evaluated at posterior samples, thus it should always 
be used in combination with `ivon.sample_parameters`. 

Additionally, IVON supports multi-sample training for more accurate ELBO loss 
estimation and better training results. This is optional and the function 
`ivon.accumulate_gradients` should be used to collect the intermediate results 
before the final `.update()` call.

In general, a typical training step could be carried out as follows:

```python
train_mcsamples = 1  # 1 sample is good enough, more even better
rngkey, *mc_keys = jax.random.split(rngkey, train_mcsamples + 1)
# Stochastic natural gradient VI with IVON
for i, key in enumerate(mc_keys):
    # draw IVON weight posterior sample
    psample, optstate = ivon.sample_parameters(key, params, optstate)
    # get gradient for this MC sample from feed-forward + backprop
    updates = ff_backprop(psample, inputs, target, ...)
    if i == train_mcsamples - 1:  # last step
        # compute IVON updates
        updates, optstate = optimizer.update(updates, optstate, params)
    else:  # intermediate steps
        # accumulate for current sample
        optstate = ivon.accumulate_gradients(updates, optstate)
params = optax.apply_updates(params, updates)
```

## Installation and dependencies

Simply copy the `ivon.py` file to your project folder and import it.

This file only depends on [`jax`](https://jax.readthedocs.io/en/latest/installation.html) and [`optax`](https://optax.readthedocs.io/en/latest/index.html#installation). Visit their docs to see how they should be installed.

For people working with [`flax`](https://flax.readthedocs.io/en/latest/index.html), we have [a simple Flax usage example adapted from their official documentation](./example_optax_flax.py). 

We also provide an example that has additional dependencies, see below.

## Example: Training ResNet-18 on CIFAR-10

The folder [resnet-cifar](./resnet-cifar) contains an example showing how the IVON optimizer can tackle image classification tasks. This code base is adapted from the [Bayesian-SAM repo](https://github.com/team-approx-bayes/bayesian-sam) with permission from the author. Check out its package dependencies [here](./resnet-cifar/requirements.txt).

### Run training

Go to the [resnet-cifar](./resnet-cifar) folder, run the following commands:

**SGD**
```
python train.py --alpha 0.03 --beta1 0.9 --priorprec 25 --optim sgd --dataset cifar10
```
This should train to around 94.8% test accuracy. 

**IVON**
```
python train.py --alpha 0.5 --beta1 0.9 --beta2 0.99999 --priorprec 25 --optim ivon --hess-init 0.5 --dataset cifar10
```
This should train to around 95.2% test accuracy. 

### Run test

In [resnet-cifar](./resnet-cifar) run the following commands. Adapt `--resultsfolder` to the actual folder which stores the training results.

**SGD**
```
python test.py --resultsfolder results/cifar10_resnet18/sgd/run_0/
```

**IVON**
```
python test.py --resultsfolder results/cifar10_resnet18/ivon/run_0/ --testmc 64
```

Tests should show that IVON with 64-sample Bayesian prediction gets significantly better uncertainty metrics.
