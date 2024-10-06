from __future__ import annotations


def get_supported_models() -> list[str]:
    """Provides all currently supported models as a list of their names.

    The 'all' specifier is used in reverse mapping as it enables to easily specify all models

    :return: the names of all supported models and 'all' specifier
    """
    # Disable quadratic model, but allow to process already existing profiles with quad model
    return ["all", "constant", "linear", "logarithmic", "quadratic", "power", "exponential"]


def get_supported_nparam_methods() -> list[str]:
    """Provides all currently supported computational methods, to
    estimate the optimal number of buckets, as a list of their names.

    :return: the names of all supported methods
    """
    return ["regressogram", "moving_average", "kernel_regression"]
