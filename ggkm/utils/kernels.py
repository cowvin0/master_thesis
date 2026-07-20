import numpy as np


class KernelFunc:

    def __init__(self, kernel, gamma, degree, coef0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _compute_kernel(self, X1, X2):
        kernels = {
            "linear": self.linear_kernel,
            "rbf": self.gaussian_kernel,
            "laplacian": self.laplacian_kernel,
            "exponential": self.exponential_kernel,
            "cauchy": self.cauchy_kernel,
            "sigmoid": self.sigmoid_kernel,
            "polynomial": self.polynomial_kernel,
        }

        if self.kernel not in kernels:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Available kernels: {list(kernels.keys())}"
            )

        return kernels[self.kernel](X1, X2)

    def linear_kernel(self, X1, X2):
        return X1 @ X2.T

    def laplacian_kernel(self, X1, X2):
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        dist = np.sum(diff, axis=2)
        return np.exp(-self.gamma * dist)

    def sigmoid_kernel(self, X1, X2):
        return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)

    def cauchy_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return 1 / (1 + self.gamma * sq_dist)

    def exponential_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        return np.exp(-self.gamma * dist)

    def polynomial_kernel(self, X1, X2):
        return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

    def gaussian_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return np.exp(-self.gamma * sq_dist)
