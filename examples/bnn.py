import matplotlib.pyplot as plt
import numpy as np

import pygrad as gd


class Posterior(gd.distributions.Normal):

    def __init__(
            self,
            size,
            rv: str,
            name: str = 'N'):
        super().__init__(rv=rv, name=name)
        self.loc = gd.Tensor(
            np.zeros(size), dtype=gd.config.dtype, is_variable=True)
        self.lns = gd.Tensor(
            np.zeros(size) - 5, dtype=gd.config.dtype, is_variable=True)

    def forward(self) -> gd.stats.Normal:
        return gd.stats.Normal(loc=self.loc, scale=gd.exp(self.lns))


class Predictor(gd.distributions.Normal):

    def __init__(self, rv='y', name='N'):
        super().__init__(rv=rv, name=name)

    def forward(self, x, w1, b1, w2, b2) -> gd.stats.Normal:
        h = gd.tanh(x @ w1 + b1)
        h = h @ w2 + b2
        return gd.stats.Normal(loc=h, scale=0.1)


if __name__ == "__main__":
    x = np.linspace(-3, 3, 20)[:, None]
    y = np.cos(x) + np.random.uniform(-0.1, 0.1, size=x.shape)
    x_test = np.linspace(-5, 5, 100)[:, None]

    gd.config.dtype = gd.Float32
    prior = (
        gd.distributions.Normal('w1', 'p', loc=np.zeros((1, 10)))
        * gd.distributions.Normal('b1', 'p', loc=np.zeros(10))
        * gd.distributions.Normal('w2', 'p', loc=np.zeros((10, 1)))
        * gd.distributions.Normal('b2', 'p', loc=np.zeros(1))
    )
    py = Predictor('y', 'p')
    p_joint = py * prior
    plt.scatter(x.ravel(), y.ravel())
    for _ in range(10):
        plt.plot(
            x_test.ravel(),
            py(x=x_test, **prior.sample()).loc.data.ravel(),
            color='r')
    plt.show()

    q = (
        Posterior((1, 10), rv='w1', name='q')
        * Posterior(10, rv='b1', name='q')
        * Posterior((10, 1), rv='w2', name='q')
        * Posterior(1, rv='b2', name='q')
    )
    optimizer = gd.optimizers.Adam(q, 0.1)
    for _ in range(1000):
        q.clear()
        p_joint.clear()
        q_sample = q.sample()
        elbo = p_joint.logpdf(
            {'x': gd.Tensor(x), 'y': gd.Tensor(y), **q_sample},
        ) - q.logpdf(q_sample, use_cache=True)
        optimizer.maximize(elbo)
        if optimizer.n_iter % 100 == 0:
            optimizer.learning_rate *= 0.8
    plt.scatter(x.ravel(), y.ravel())
    for _ in range(10):
        plt.plot(
            x_test.ravel(),
            py(x=x_test, **q.sample()).loc.data.ravel(),
            color='r')
    plt.show()
