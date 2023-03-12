import numpy as np
from sklearn.decomposition import PCA
import warnings


class MultiVariateCS:
    def __init__(self, delta):
        self.delta = delta

    def partial_fit(self, X):
        pass

    def detect(self):
        pass

    def reset_state(self):
        pass


class PCA_CS(MultiVariateCS):
    def __init__(
        self,
        pca_n_components=0.99,
        delta=0.001,
        drift_threshold=5,
        warning_threshold=3,
        batch_size=100,
    ):
        super().__init__(delta)
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.batch_size = batch_size
        self.reference_window = []
        self.test_window = []
        self.n = 0
        self.pca_n_components = pca_n_components
        self.pca = PCA(self.pca_n_components)
        self.score = 0

    def partial_fit(self, X):
        super().partial_fit(X)
        self.n += len(X)
        self._update_windows(X)

        if len(self.reference_window) == self.batch_size:
            self._update_pca()
            self._update_score()
        return False

    def detect(self):
        if self.score > self.drift_threshold:
            self.reset_state()
            return True
        return False

    def reset_state(self):
        self.reference_window = []
        self.test_window = []
        self.n = 0
        self.pca = PCA(self.pca_n_components)
        self.score = 0

    def _update_windows(self, X):
        if len(self.test_window) < self.batch_size:
            self.test_window.append(X)

        if len(self.test_window) == self.batch_size:
            self.test_window.append(X)
            oldest_batch = self.test_window[0]
            self.test_window.pop(0)
            self.reference_window.append(oldest_batch)
            if len(self.reference_window) > self.batch_size:
                self.test_window.pop(0)

    def _update_pca(self):
        s1 = np.concatenate(self.reference_window, axis=0)
        self.pca.fit(s1)

    def _update_score(self):
        s1 = np.concatenate(self.reference_window, axis=0)
        s2 = np.concatenate(self.reference_window, axis=0)
        s1_projected = self.pca.transform(s1)
        s2_projected = self.pca.transform(s2)

        divergence = self._divergence(s1_projected, s2_projected)
        self.score = np.max(divergence)

    def _divergence(self, s1, s2):
        u1 = np.mean(s1, axis=0)
        u2 = np.mean(s2, axis=0)
        var1 = np.var(s1, axis=0)
        var2 = np.var(s2, axis=0)

        divergence = (u1 - u2) ** 2 / (2 * var1) + 0.5 * (
            var2 / var1 - 1 - np.log(var2 / var1)
        )

        return divergence


class CUMSUM_CS(MultiVariateCS):
    def __init__(
        self, delta=0.001, drift_threshold=5, warning_threshold=3, batch_size=100
    ):
        super().__init__(delta)
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.batch_size = batch_size
        self.window = []
        self.n = 0
        self.mean = None
        self.variance = None
        self.upper_bound = None
        self.lower_bound = None

    def partial_fit(self, X):
        super().partial_fit(X)
        self.window.append(X)
        self.n += len(X)
        if len(self.window) > self.batch_size:
            self._update_scores()
        return False

    def detect(self):
        if len(self.window) < self.batch_size:
            return False

        if np.any(self.upper_bound - self.mean > self.drift_threshold) or np.any(
            self.mean - self.lower_bound > self.drift_threshold
        ):
            return True

        elif np.any(self.upper_bound - self.mean > self.warning_threshold) or np.any(
            self.mean - self.lower_bound > self.warning_threshold
        ):
            warnings.warn("Warning: possible change point detected.")
            return False

        else:
            return False

    def reset_state(self):
        self.window = []
        self.n = 0
        self.mean = None
        self.variance = None

    def _update_scores(self):
        X = self.window[-1]
        if self.mean is None or self.variance is None:
            self.mean = np.mean(X, axis=0)
            self.variance = np.var(X, axis=0)
            self.upper_bound = np.zeros(X.shape[1])
            self.lower_bound = np.zeros(X.shape[1])

        else:
            old_mean = self.mean
            old_variance = self.variance
            old_n = self.n

            new_mean = np.mean(X, axis=0)
            new_variance = np.var(X, axis=0)
            new_n = len(X)

            self.mean = (old_mean * old_n + new_mean * new_n) / (old_n + new_n)
            self.variance = (
                (old_n - 1) * old_variance
                + old_n * (old_mean**2)
                + (new_n - 1) * new_variance
                + new_n * (new_mean**2)
            ) / (old_n + new_n - 1) - self.mean**2

            z = (X - self.mean) / self.variance
            z_sum = np.sum(z, axis=0)
            self.upper_bound = np.clip(self.upper_bound + z_sum - self.delta, a_min=0)
            self.lower_bound = np.clip(self.lower_bound - z_sum - self.delta, a_min=0)


class EWMA:
    def __init__(
        self, alpha, delta, drift_threshold=5, warning_threshold=3, batch_size=100
    ):
        self.alpha = alpha
        self.delta = delta
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.batch_size = batch_size
        self.window = []
        self.n = 0
        self.ewma = None
        self.cumulative_scores = []

    def partial_fit(self, X):
        self.window.append(X)
        self.n += len(X)
        if len(self.window) > self.batch_size:
            self._update_ewma()
            self._update_scores()
        return False

    def detect(self):
        if len(self.cumulative_scores) > 2 * self.batch_size:
            self.cumulative_scores = self.cumulative_scores[self.batch_size :]
            threshold = np.quantile(self.cumulative_scores, 1 - self.delta)
            scores_over_threshold = np.where(self.cumulative_scores > threshold)[0]
            if len(scores_over_threshold) > self.drift_threshold:
                self.reset_state()
                return True
            elif len(scores_over_threshold) > self.warning_threshold:
                return False
        return False

    def reset_state(self):
        self.window = []
        self.n = 0
        self.ewma = None
        self.cumulative_scores = []

    def _update_ewma(self):
        X = self.window[-1]
        if self.ewma is None:
            self.ewma = X.mean(axis=0)
        else:
            self.ewma = self.alpha * X.mean(axis=0) + (1 - self.alpha) * self.ewma

    def _update_scores(self):
        X = self.window[-1]
        error = np.sum((X - self.ewma) ** 2, axis=1)
        self.cumulative_scores.extend(error)
