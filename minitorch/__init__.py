"""Module Imports for Tensor Library

This script imports various submodules and classes to construct a comprehensive tensor-based library for numerical computations.
It includes testing utilities, optimized operations, and fundamental tensor functionalities.

Modules and Classes:
- `MathTest`, `MathTestVariable`: Testing utilities for mathematical operations.
- `fast_ops`: Contains optimized operations for faster computations.
- `cuda_ops`: CUDA-specific operations for GPU acceleration.
- `tensor_data`, `tensor_functions`, `tensor_ops`: Core tensor data structures, operations, and function implementations.
- `scalar`, `scalar_functions`: Scalar-level computations and utilities.
- `module`, `autodiff`: Modules for building computation graphs and automatic differentiation.
- `datasets`: Pre-defined datasets for machine learning tasks.
- `optim`: Optimization algorithms for training purposes.
- `testing`: Utilities for unit testing and validation of implementations.

Notes
-----
- The `# noqa: F401,F403` directives are used to suppress linting warnings about unused imports and wildcard imports.
- Duplicate imports of `module` and `autodiff` appear to be redundant and may need to be reviewed.
- The `fast_ops` and `cuda_ops` modules are also explicitly imported at the end to ensure their availability.

Usage:
This script is intended to provide a central place for importing all necessary components of the library, allowing other modules or scripts to directly access the desired functionalities without managing individual imports.

"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403
