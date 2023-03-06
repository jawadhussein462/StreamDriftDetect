import numpy as np
from scipy.stats import norm


class StatisticalBasedMethod:
    def __init__(self, delta=0.001, min_instances=30):
        self.delta = delta
        self.min_instances = min_instances

    def fit(self, X):
        self.data = X
        self.n = 0

    def detect(self, X):
        raise NotImplementedError("Detect method not implemented")


class ADWIN(StatisticalBasedMethod):
    def __init__(self, delta=0.001):
        super().__init__(delta)
        self.width = 0
        self.bucket1 = []
        self.bucket2 = []
        self.last_bucket = None

    def fit(self, X):
        super().fit(X)
        for x in X:
            self.add_element(x)

    def detect(self, X):
        if self.n == 0:
            return False

        change_detected = False
        for x in X:
            self.add_element(x)
            if self.width > 1 and self.bucket2 and self.bucket1:
                p_value = self.calc_p_value()
                if p_value < self.delta:
                    self.bucket1 = self.bucket2
                    self.bucket2 = []
                    self.width = len(self.bucket1)
                    change_detected = True

        return change_detected

    def add_element(self, x):
        self.n += 1
        if self.last_bucket is not None:
            self.last_bucket.append(x)
        else:
            self.bucket1.append(x)
        if len(self.bucket1) > self.width:
            self.last_bucket = self.bucket1
            self.bucket1 = self.bucket2
            self.bucket2 = []
            self.width += 1
        if len(self.bucket2) > 0 and len(self.bucket1) > 0 and self.width > 1:
            mean_bucket2 = np.mean(self.bucket2)
            mean_bucket1 = np.mean(self.bucket1)
            if abs(mean_bucket2 - mean_bucket1) / self.width >= self.delta:
                self.bucket1 = self.bucket2
                self.bucket2 = []
                self.width = len(self.bucket1)
                self.last_bucket = None

    def calc_p_value(self):
        mean_bucket2 = np.mean(self.bucket2)
        mean_bucket1 = np.mean(self.bucket1)
        std_bucket2 = np.std(self.bucket2)
        std_bucket1 = np.std(self.bucket1)
        n1 = len(self.bucket1)
        n2 = len(self.bucket2)
        delta = mean_bucket2 - mean_bucket1
        s = np.sqrt(
            ((n1 - 1) * std_bucket1**2 + (n2 - 1) * std_bucket2**2) / (n1 + n2 - 2)
        )
        z = delta / (s * np.sqrt(1 / n1 + 1 / n2))
        return 2 * (1 - norm.cdf(abs(z)))


class EDDM(StatisticalBasedMethod):
    def __init__(self, delta=0.01):
        super().__init__(delta)
        self.mean = None
        self.variance = None
        self.t = 0
        self.max_t = 0
        self.lambda_ = None
        self.log_likelihoods = []

    def fit(self, X):
        super().fit(X)
        self.mean = np.mean(X)
        self.variance = np.var(X)
        self.t = 1
        self.max_t = 1
        self.lambda_ = 1 / self.variance

    def detect(self, X):
        if self.n == 0:
            return False

        change_detected = False
        for x in X:
            self.n += 1
            self.t += 1
            if self.n == 1:
                continue
            elif self.n == 2:
                self.log_likelihoods.append(self.calc_log_likelihood(x))
                self.max_t = 1
            else:
                log_likelihood = self.calc_log_likelihood(x)
                self.log_likelihoods.append(log_likelihood)
                if log_likelihood < self.delta:
                    change_detected = True
                    self.mean = np.mean(self.data[: self.t])
                    self.variance = np.var(self.data[: self.t])
                    self.lambda_ = 1 / self.variance
                    self.n = 0
                    self.t = 0
                    self.log_likelihoods = []
                elif log_likelihood > self.log_likelihoods[self.max_t - 1]:
                    self.max_t = self.n - 1

        return change_detected

    def calc_log_likelihood(self, x):
        if self.lambda_ is None:
            return 0
        else:
            return np.log(self.lambda_) - self.lambda_ * (x - self.mean) ** 2 / 2
