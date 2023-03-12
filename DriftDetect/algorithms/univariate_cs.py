import numpy as np
from abc import ABC, abstractmethod


class UniVariateCS(ABC):
    def __init__(self, delta):
        self.delta = delta

    @abstractmethod
    def partial_fit(self, X):
        pass

    def detect(self):
        pass

    @abstractmethod
    def reset_state(self):
        pass


class CUMSUM(UniVariateCS):
    def __init__(self, delta=0.001, drift_threshold=5, warning_threshold=3):
        super().__init__(delta)
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.sum = 0
        self.mean = None
        self.variance = None
        self.n = 0
        self.last_mean = None
        self.last_variance = None
        self.last_n = None

    def partial_fit(self, X):
        super().partial_fit(X)
        if self.mean is None:
            self.mean = np.mean(X)
            self.variance = np.var(X)
            self.n = len(X)

        else:
            old_mean = self.mean
            old_variance = self.variance
            old_n = self.n

            new_mean = np.mean(X)
            new_variance = np.var(X)
            new_n = len(X)

            self.mean = (old_mean * old_n + new_mean * new_n) / (old_n + new_n)
            self.n = new_n + old_n
            self.variance = (
                (old_n - 1) * (old_variance + old_mean**2)
                + (new_n - 1) * (new_variance + new_mean**2)
            ) / (self.n - 1) - self.mean

            self.sum += new_n * (new_mean - old_mean) ** 2 / (new_n + old_n)

            self.last_mean = old_mean
            self.last_variance = old_variance
            self.last_n = old_n

    def detect(self):
        if self.last_mean is None or self.last_n == 0:
            return False

        if abs(self.sum) > self.drift_threshold * np.sqrt(
            self.last_n * self.last_variance
        ):
            self.reset_state()
            return True
        elif abs(self.sum) > self.warning_threshold * np.sqrt(
            self.last_n * self.last_variance
        ):
            return True
        return False

    def reset_state(self):
        self.sum = 0
        self.mean = None
        self.variance = None
        self.n = 0
        self.last_mean = None
        self.last_variance = None


class PageHinkley(UniVariateCS):
    def __init__(self, delta=0.005, threshold=5, alpha=1 - 0.0001):
        super().__init__(delta)
        self.threshold = threshold
        self.alpha = alpha
        self.sum = 0
        self.mean = 0
        self.n = 0

    def partial_fit(self, X):
        super().partial_fit(X)
        n_new = len(X)
        mean_new = np.mean(X)
        sum_new = np.sum(np.maximum(0, X - mean_new - self.delta))
        if self.n == 0:
            self.mean = mean_new
            self.sum = sum_new
        else:
            self.mean = ((self.n * self.mean) + (n_new * mean_new)) / (self.n + n_new)
            self.sum += sum_new + (
                (n_new * self.n * (mean_new - self.mean) ** 2) / (self.n + n_new)
            )
        self.n += n_new

    def detect(self):
        if self.n < 2:
            return False
        test_statistic = np.sqrt(self.n) * self.sum
        threshold = np.sqrt(2 * np.log(1 / self.alpha)) + (
            self.threshold / np.sqrt(self.n)
        )
        if test_statistic > threshold:
            self.reset_state()
            return True
        return False

    def reset_state(self):
        self.sum = 0
        self.mean = 0
        self.n = 0
