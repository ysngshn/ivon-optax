# adapted from the example usage from:
# https://flax.readthedocs.io/en/latest/api_reference/flax.training.html
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import ivon


def loss_fn(params, x, y):
    predictions = state.apply_fn({'params': params}, x)
    loss = optax.l2_loss(predictions=predictions, targets=y).mean()
    return loss


if __name__ == "__main__":
    key = jax.random.key(0)
    mc_samples = 3
    x = jnp.ones((1, 2))
    y = jnp.ones((1, 2))
    model = nn.Dense(2)
    variables = model.init(jax.random.key(0), x)
    params = variables['params']
    tx = ivon.ivon(
        learning_rate=1.0,
        ess=1,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx)

    print("Before training step:")
    print(f"loss evaluated at Gaussian posterior mean: {loss_fn(state.params, x, y)}\n")

    for i in range(mc_samples):
        key, skey = jax.random.split(key)
        param_sample, opt_state = ivon.sample_parameters(
            skey, state.params, state.opt_state
        )
        print(f"MC sample iteration {i}:")
        print(f"loss evaluated with Gaussian posterior sample: {loss_fn(param_sample, x, y)}\n")
        grads = jax.grad(loss_fn)(param_sample, x, y)
        if i == mc_samples - 1:
            state = state.replace(opt_state=opt_state)
            state = state.apply_gradients(grads=grads)
        else:
            opt_state = ivon.accumulate_gradients(grads, opt_state)
            state = state.replace(opt_state=opt_state)

    print("After training step:")
    print(f"loss evaluated at Gaussian posterior mean: {loss_fn(state.params, x, y)}")


# Expected output:
# ----------------
# Before training step:
# loss evaluated at Gaussian posterior mean: 3.351468563079834
#
# MC sample iteration 0:
# loss evaluated with Gaussian posterior sample: 5.153180122375488
#
# MC sample iteration 1:
# loss evaluated with Gaussian posterior sample: 1.746554970741272
#
# MC sample iteration 2:
# loss evaluated with Gaussian posterior sample: 3.8970746994018555
#
# After training step:
# loss evaluated at Gaussian posterior mean: 1.5659505128860474
