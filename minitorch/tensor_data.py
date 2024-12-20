from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride

    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    ## TODO: Implement for Task 2.2.
    # raise NotImplementedError("Need to implement for Task 2.2")

    # # Reverse the order of dimensions for easier processing
    # big_index = big_index[::-1]
    # big_shape = big_shape[::-1]
    # shape = shape[::-1]

    # # Initialize out_index with zeros
    # for i in range(len(shape)):
    #     out_index[i] = 0

    # # Process each dimension
    # for i, (bi, bs, s) in enumerate(zip(big_index, big_shape, shape)):
    #     if s > 1:
    #         out_index[i] = bi if bs > 1 else 0

    # # Reverse back the out_index
    # out_index[:] = out_index[::-1]
    # ASSIGN2.2
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None
    # END ASSIGN2.2


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    ## TODO: Implement for Task 2.2.
    # raise NotImplementedError("Need to implement for Task 2.2")

    # # Reverse shapes for easier processing (broadcast from right to left)
    # s1 = list(reversed(shape1))
    # s2 = list(reversed(shape2))

    # # Pad shorter shape with 1s
    # max_len = max(len(s1), len(s2))
    # s1 = s1 + [1] * (max_len - len(s1))
    # s2 = s2 + [1] * (max_len - len(s2))

    # # Compute broadcasted shape
    # result = []
    # for d1, d2 in zip(s1, s2):
    #     if d1 == 1 or d2 == 1 or d1 == d2:
    #         result.append(max(d1, d2))
    #     else:
    #         raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    # # Reverse back and return as tuple
    # return tuple(reversed(result))
    # Module 2 answer
    # ASSIGN2.2
    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
    return tuple(reversed(c_rev))
    # END ASSIGN2.2


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape following NumPy broadcasting rules.

        Args:
        ----
            shape_a (UserShape): The shape of the first array. Can be a tuple or list of integers.
            shape_b (UserShape): The shape of the second array. Can be a tuple or list of integers.

        Returns:
        -------
            UserShape: The broadcasted shape as a tuple of integers.

        Raises:
        ------
            IndexingError: If the shapes cannot be broadcast together according to the broadcasting rules.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert a user-friendly index into a flat position in the underlying storage.

        Args:
        ----
            index (Union[int, UserIndex]): The index to convert.

        Returns:
        -------
            int: The flattened position in the underlying storage corresponding to the given index.

        Raises:
        ------
            IndexingError: If the index is invalid.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate an iterable of all valid indices for the tensor.

        Returns
        -------
            Iterable[UserIndex]: An iterable where each item is a tuple of integers
            representing a valid index into the tensor.

        Yields
        ------
            UserIndex: A tuple of integers representing a valid index.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieve the value at the specified index in the tensor.

        Args:
        ----
            key (UserIndex): The index at which to retrieve the value.

        Returns:
        -------
            float: The value stored at the specified index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at the specified index in the tensor.

        Args:
        ----
            key (UserIndex): The index at which to set the value.
            val (float): The new value to be set at the specified index.

        Returns:
        -------
            None

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        ## TODO: Implement for Task 2.1.
        # raise NotImplementedError("Need to implement for Task 2.1")

        # Original
        # # Create new shape based on the given order
        # new_shape = tuple(self.shape[i] for i in order)

        # # Create new strides based on the given order
        # new_strides = tuple(self._strides[i] for i in order)

        # # Create and return a new TensorData object
        # return TensorData(storage=self._storage, shape=new_shape, strides=new_strides)
        # ASSIGN2.1
        return TensorData(
            self._storage,
            tuple([self.shape[o] for o in order]),
            tuple([self._strides[o] for o in order]),
        )
        # END ASSIGN2.1

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
