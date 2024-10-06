"""Collective package for degradations.

Contains the actual methods in isolate modules, and then a factory module,
containing helper and generic stuff."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
