import jax.numpy as jnp
from typing import Tuple

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: Helps with arithmetic issues.
        :param shape: The shape of the data stream's output.
        """
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: A copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean
        new_object.var = self.var
        new_object.count = self.count
        return new_object


    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)


    def update(self, arr: jnp.ndarray) -> None:
        """
        Update the running statistics with a new batch of data.

        :param arr: A batch of data.
        """
        batch_mean = jnp.mean(arr, axis=0)
        batch_var = jnp.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        

    def update_from_moments(self, batch_mean: jnp.ndarray, batch_var: jnp.ndarray, batch_count: float) -> None:
        """
        Update the running statistics using moments.

        :param batch_mean: The mean of the new batch.
        :param batch_var: The variance of the new batch.
        :param batch_count: The number of samples in the new batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count