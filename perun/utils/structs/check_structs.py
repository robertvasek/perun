from __future__ import annotations


def get_supported_detection_models_strategies() -> list[str]:
    """
    Provides supported detection models strategies to execute
    the degradation check between two profiles with different kinds
    of models. The individual strategies represent the way of
    executing the detection between profiles and their models:

        - best-param: best parametric models from both profiles
        - best-non-param: best non-parametric models from both profiles
        - best-model: best models from both profiles
        - all-param: all parametric models pair from both profiles
        - all-non-param: all non-parametric models pair from both profiles
        - all-models: all models pair from both profiles
        - best-both: best parametric and non-parametric models from both profiles

    :return: the names of all supported degradation models strategies
    """
    return [
        "best-model",
        "best-param",
        "best-nonparam",
        "all-param",
        "all-nonparam",
        "all-models",
        "best-both",
    ]
