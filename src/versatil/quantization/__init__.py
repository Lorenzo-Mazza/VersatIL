"""VersatIL quantization bridge to torchao.

This package init must stay import-free: ``versatil.__init__`` imports
``versatil.quantization.torch_patches`` to patch torchao on disk before any
torchao pt2e module is loaded, so importing torchao-dependent submodules here
would crash unpatched installs on Python 3.14.
"""
