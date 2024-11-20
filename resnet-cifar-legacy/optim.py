from typing import Optional
import jax
import jax.random as jrandom
import jax.numpy as jnp
import copy
from typing import NamedTuple
import ivon
import optax


class TrainState(NamedTuple):
    """
    collects the all the state required for neural network training
    """
    optstate: dict
    netstate: None
    rngkey: None

def build_sgd_optimizer(lossgrad,
                        learningrate : float,
                        momentum : float,
                        wdecay : float): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree_map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate

        (loss, netstate), grad = lossgrad(optstate['w'], trainstate.netstate, minibatch, is_training=True) 

        # momentum
        optstate['gm'] = jax.tree_map(
            lambda gm, g, w: momentum * gm + g + wdecay * w, optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree_map(lambda p, gm: p - learningrate * lrfactor * gm, optstate['w'], optstate['gm'])
    
        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate)

        return newtrainstate, loss

    return init, step


def build_ivon_optimizer(
    lossgrad,
    learningrate: float,
    ess: float,
    hess_init: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    wdecay: float = 1e-4,
    clip_radius: float = float("inf"),
    rescale_lr: bool = True,
    every_k: int = 1,
    axis_name: Optional[str] = None,
):
    optimizer = ivon.ivon(
        learningrate, ess, hess_init, beta1, beta2, wdecay, clip_radius,
        rescale_lr, axis_name)

    def init(weightinit, netstate, rngkey):
        optstate = optimizer.init(weightinit)

        return TrainState(
            optstate={
                'w': weightinit,
                'state': optstate,
            },
            netstate=netstate,
            rngkey=rngkey)

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate
        netstate = trainstate.netstate
        (
            params, ivonoptstate
        ) = optstate['w'], optstate['state']
        n_samples = every_k
        keys = jrandom.split(trainstate.rngkey, n_samples+1)
        rngkey = keys[0]
        grad, loss = None, 0.0
        for skey in keys[1:]:
            psample, noise = optimizer.sampled_params(
                skey, params, ivonoptstate
            )
            (loss, netstate), grad = lossgrad(
                psample, trainstate.netstate, minibatch, is_training=True)
            ivonoptstate = optimizer.accumulate(grad, ivonoptstate, noise)
        grad, ivonoptstate = optimizer.step(ivonoptstate, params)
        grad = jax.tree_map(lambda g: lrfactor * g, grad)
        params = optax.apply_updates(params, grad)
        optstate = {
            'w': params,
            'state': ivonoptstate,
        }

        newtrainstate = trainstate._replace(
            optstate=optstate,
            netstate=netstate,
            rngkey=rngkey,
        )

        return newtrainstate, loss

    return init, step, optimizer.sampled_params
