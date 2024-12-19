from typing import Optional, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree as jtree
from jax import lax
import optax


def randn_like(rng: jrandom.PRNGKey, t):
    tleaves, tdef = jtree.flatten(t)
    keys = jax.random.split(rng, len(tleaves))
    randn = jrandom.normal
    samples = [randn(k, l.shape, l.dtype) for k, l in zip(keys, tleaves)]
    return jtree.unflatten(tdef, samples)


class IVONState(NamedTuple):
    ess: float
    beta1: float
    beta2: float
    weight_decay: float
    momentum: optax.Updates
    hess: optax.Updates
    axis_name: Optional[str] = None
    current_step: int = 0
    grad_acc: Optional[optax.Updates] = None
    nxg_acc: Optional[optax.Updates] = None
    acc_count: int = 0


class IVON(NamedTuple):
    grad_transform: optax.GradientTransformation

    @partial(jax.jit, static_argnums=(0,))
    def init(self, params: optax.Params) -> optax.OptState:
        return self.grad_transform.init(params)

    @staticmethod
    @jax.jit
    def accumulate(
        updates: optax.Updates,
        states: optax.OptState,
        noise: optax.Updates,
    ) -> optax.OptState:
        if noise is None:
            raise ValueError(
                "Missing noise in the IVON state: updated optimizer state"
                " yielded from the `IVON.sampled_params` context required"
            )

        ivonstate = states[0]
        grad_acc, nxg_acc, old_count = ivonstate[-3:]
        if grad_acc is None:
            grad_acc = updates
            nxg_acc = jtree.map(lambda g, n: n * g, updates, noise)
        else:
            grad_acc = jtree.map(lambda a, g: a + g, grad_acc, updates)
            nxg_acc = jtree.map(
                lambda a, g, n: a + n * g, nxg_acc, updates, noise
            )
        ivonstate = IVONState(
            *ivonstate[:-3], grad_acc, nxg_acc, old_count + 1
        )
        states = (ivonstate, *states[1:])
        return states

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        states: optax.OptState,
        params: optax.Params,
    ) -> tuple[optax.Updates, optax.OptState]:
        ivonstate = states[0]
        (
            ess,
            beta1,
            beta2,
            weight_decay,
            momentum,
            hess,
            axis_name,
            current_step,
            grad_acc,
            nxg_acc,
            acc_count,
        ) = ivonstate
        current_step += 1
        avg_grad, avg_nxg = _avg_grad_hess(
            grad_acc, nxg_acc, acc_count, axis_name
        )
        hess = _update_hess(hess, avg_nxg, ess, beta2, weight_decay)
        ivonstate = IVONState(
            ess,
            beta1,
            beta2,
            weight_decay,
            momentum,
            hess,
            axis_name,
            current_step,
            None,
            None,
            0,
        )
        return self.grad_transform.update(
            avg_grad, (ivonstate, *states[1:]), params
        )

    @staticmethod
    def _get_param_sample(
        rng: jax.random.PRNGKey, params: optax.Params, state: IVONState
    ) -> tuple[optax.Params, optax.Updates]:
        rsqrt, ess, weight_decay = lax.rsqrt, state.ess, state.weight_decay
        noise = jtree.map(
            lambda n, h: n * rsqrt(ess * (h + weight_decay)),
            randn_like(rng, params),
            state.hess,
        )
        psample = jtree.map(lambda p, n: p + n, params, noise)
        return psample, noise

    @staticmethod
    @jax.jit
    def sampled_params(
        rng: jax.random.PRNGKey,
        model_params: optax.Params,
        opt_state: optax.OptState,
    ):
        return IVON._get_param_sample(rng, model_params, opt_state[0])


def ivon_init(
    params: optax.Params,
    ess: float,
    hess_init: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    weight_decay: float = 1e-4,
    axis_name: Optional[str] = None,
) -> IVONState:
    zeros_like = jnp.zeros_like
    momentum = jtree.map(zeros_like, params)
    hess = jtree.map(lambda t: jnp.full_like(t, fill_value=hess_init), params)
    return IVONState(
        ess,
        beta1,
        beta2,
        weight_decay,
        momentum,
        hess,
        axis_name,
        0,
        None,
        None,
        0,
    )


def _avg_grad_hess(grad_acc, nxg_acc, acc_count, axis_name):
    avg_grad = jtree.map(lambda g: g / acc_count, grad_acc)
    avg_nxg = jtree.map(lambda h: h / acc_count, nxg_acc)
    if axis_name is not None:
        avg_grad = lax.pmean(avg_grad, axis_name=axis_name)
        avg_nxg = lax.pmean(avg_nxg, axis_name=axis_name)
    return avg_grad, avg_nxg


def _update_momentum(momentum, avg_grad, b1):
    return jtree.map(lambda g, m: b1 * m + (1.0 - b1) * g, avg_grad, momentum)


def _update_hess(hess, avg_nxg, ess, b2, wd):
    nll_hess = jtree.map(lambda a, h: ess * a * (h + wd), avg_nxg, hess)
    square = lax.square
    return jtree.map(
        lambda h, f: b2 * h
        + (1.0 - b2) * f
        + 0.5 * square((1.0 - b2) * (h - f)) / (h + wd),
        hess,
        nll_hess,
    )


def _compute_update(params, hess, momentum, wd, debias):
    return jtree.map(
        lambda p, h, m: (m / debias + wd * p) / (h + wd),
        params,
        hess,
        momentum,
    )


def ivon_update(
    updates: optax.Updates,
    state: IVONState,
    params: Optional[optax.Params] = None,
) -> tuple[optax.Updates, IVONState]:
    if params is None:
        raise ValueError("IVON update requires the `params` argument.")
    (
        ess,
        beta1,
        beta2,
        weight_decay,
        momentum,
        hess,
        axis_name,
        current_step,
    ) = state[:-3]
    momentum = _update_momentum(momentum, updates, beta1)
    debias = 1.0 - beta1**current_step
    updates = _compute_update(params, hess, momentum, weight_decay, debias)
    return updates, IVONState(
        ess,
        beta1,
        beta2,
        weight_decay,
        momentum,
        hess,
        axis_name,
        current_step,
        None,
        None,
        0,
    )


def scale_by_ivon(
    ess: float,
    hess_init: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    weight_decay: float = 1e-4,
    axis_name: Optional[str] = None,
) -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> IVONState:
        return ivon_init(
            params, ess, hess_init, beta1, beta2, weight_decay, axis_name
        )

    return optax.GradientTransformation(init_fn, ivon_update)


def ivon(
    learning_rate: optax.ScalarOrSchedule,
    ess: float,
    hess_init: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    weight_decay: float = 1e-4,
    clip_radius: float = float("inf"),
    rescale_lr: bool = True,
    axis_name: Optional[str] = None,
) -> IVON:
    ivon_transform = scale_by_ivon(
        ess, hess_init, beta1, beta2, weight_decay, axis_name
    )
    if rescale_lr:
        lr_scale = (
            optax.scale_by_learning_rate(learning_rate),
            optax.scale(hess_init + weight_decay),
        )
    else:
        lr_scale = (optax.scale_by_learning_rate(learning_rate),)

    if clip_radius < float("inf"):
        transform = optax.chain(
            ivon_transform,
            optax.clip(clip_radius),
            *lr_scale,
        )
    else:
        transform = optax.chain(ivon_transform, *lr_scale)
    return IVON(transform)
