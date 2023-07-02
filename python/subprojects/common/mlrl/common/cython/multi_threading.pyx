"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal


cdef class ManualMultiThreadingConfig:
    """
    Allows to configure the multi-threading behavior of a parallelizable algorithm by manually specifying the number of
    threads to be used.
    """

    def get_num_threads(self) -> int:
        """
        Returns the number of threads that are used.

        :return: The number of threads that are used or 0, if all available CPU cores are utilized
        """
        return self.config_ptr.getNumThreads()

    def set_num_threads(self, num_threads: int) -> ManualMultiThreadingConfig:
        """
        Sets the number of threads that should be used.

        :param num_threads: The number of threads that should be used. Must be at least 1 or 0, if all available CPU
                            cores should be utilized
        :return:            A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading
                            behavior
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self
