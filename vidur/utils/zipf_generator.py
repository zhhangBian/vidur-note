import numpy as np

EPS = 1e-8


# 用于生成遵循Zipf分布（也称为Zipf定律或Zipf分布）的随机数生成器
class ZipfGenerator:
    def __init__(
        self, min: int, max: int, theta: float, scramble: bool, seed: int
    ) -> None:
        self._min = min
        self._max = max
        self._items = max - min + 1
        self._theta = theta
        self._zeta_2 = self._zeta(2, self._theta)
        self._alpha = 1.0 / (1.0 - self._theta)
        self._zetan = self._zeta(self._items, self._theta)
        self._eta = (1 - np.power(2.0 / self._items, 1 - self._theta)) / (
            1 - self._zeta_2 / (self._zetan + EPS)
        )
        # 是否对生成的随机数进行额外的随机化处理
        self._scramble = scramble
        self._seed = seed
        self._generator = np.random.RandomState(seed)

    def _zeta(self, count: float, theta: float) -> float:
        return np.sum(1 / (np.power(np.arange(1, count), theta)))

    # 根据Zipf分布生成下一个随机数。它使用累积分布函数（CDF）的逆函数来计算
    def _next(self) -> int:
        u = self._generator.random_sample()
        uz = u * self._zetan

        if uz < 1.0:
            return self._min

        if uz < 1.0 + np.power(0.5, self._theta):
            return self._min + 1

        return self._min + int(
            (self._items) * np.power(self._eta * u - self._eta + 1, self._alpha)
        )

    # 如果 scramble 为 True，则 next 方法会将生成的随机数与种子结合，通过哈希函数进行随机化处理，以增加随机性
    def next(self) -> int:
        retval = self._next()
        if self._scramble:
            retval = self._min + hash(str(retval) + str(self._seed)) % self._items

        return retval
