# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task 3.1: Parallelization


MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
C:\Users\mlock\Desktop\CS 5781 Machine Learning 
Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (176)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\mlock\Desktop\CS 5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (176) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        ## TODO: Implement for Task 3.1.                                     |
        #raise NotImplementedError("Need to implement for Task 3.1")         |
                                                                             |
        if np.array_equal(in_strides, out_strides) and np.array_equal(       |
            in_shape, out_shape                                              |
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #0
                out[i] = fn(in_storage[i])                                   |
        #Map s                                                               |
        else :                                                               |
            for i in prange(len(out)):---------------------------------------| #1
                # Create thread-local indices inside the parallel loop       |
                out_index = np.empty(MAX_DIMS, np.int32)                     |
                in_index = np.empty(MAX_DIMS, np.int32)                      |
                                                                             |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                o = index_to_position(out_index, out_strides)                |
                j = index_to_position(in_index, in_strides)                  |
                out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py
(196) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py 
(197) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\mlock\Desktop\CS 5781 Machine Learning
Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (231)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\mlock\Desktop\CS 5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (231)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        ## TODO: Implement for Task 3.1.                                   |
        #raise NotImplementedError("Need to implement for Task 3.1")       |
                                                                           |
        if (                                                               |
            np.array_equal(a_strides,b_strides)                            |
            and np.array_equal(a_strides,out_strides)                      |
            and np.array_equal(a_shape,b_shape)                            |
            and np.array_equal(a_shape,out_shape)                          |
        ):                                                                 |
            for i in prange(len(out)):-------------------------------------| #2
                out[i] = fn(a_storage[i],b_storage[i])                     |
                                                                           |
        #Zip S                                                             |
        else :                                                             |
            for i in prange(len(out)):-------------------------------------| #3
                # Thread-local indices                                     |
                out_index = np.empty(MAX_DIMS, np.int32)                   |
                a_index = np.empty(MAX_DIMS, np.int32)                     |
                b_index = np.empty(MAX_DIMS, np.int32)                     |
                                                                           |
                to_index(i, out_shape, out_index)                          |
                o = index_to_position(out_index, out_strides)              |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                j = index_to_position(a_index, a_strides)                  |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                k = index_to_position(b_index, b_strides)                  |
                out[o] = fn(a_storage[j], b_storage[k])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py 
(258) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py
(259) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py
(260) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\mlock\Desktop\CS 5781 Machine Learning
Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (294)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\mlock\Desktop\CS 5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (294)
------------------------------------------------------------------------|loop #ID
    def _reduce(                                                        |
        out: Storage,                                                   |
        out_shape: Shape,                                               |
        out_strides: Strides,                                           |
        a_storage: Storage,                                             |
        a_shape: Shape,                                                 |
        a_strides: Strides,                                             |
        reduce_dim: int,                                                |
    ) -> None:                                                          |
        ## TODO: Implement for Task 3.1.                                |
        #raise NotImplementedError("Need to implement for Task 3.1")    |
                                                                        |
                                                                        |
                                                                        |
        #Reduce S                                                       |
        out_index = np.zeros(MAX_DIMS, np.int32)------------------------| #4
        reduce_size = a_shape[reduce_dim]                               |
                                                                        |
        # Parallelize the outer loop over output elements               |
        for i in prange(len(out)):--------------------------------------| #5
            # Thread-local indices                                      |
            out_index = np.empty(MAX_DIMS, np.int32)                    |
            local_index = np.empty(MAX_DIMS, np.int32)                  |
                                                                        |
            to_index(i, out_shape, out_index)                           |
            o = index_to_position(out_index, out_strides)               |
                                                                        |
            # Copy indices to local                                     |
            for j in range(len(out_shape)):                             |
                local_index[j] = out_index[j]                           |
                                                                        |
            # Sequential reduction                                      |
            for s in range(reduce_size):                                |
                local_index[reduce_dim] = s                             |
                j = index_to_position(local_index, a_strides)           |
                out[o] = fn(out[o], a_storage[j])                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py
(315) is hoisted out of the parallel loop labelled #5 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\mlock\Desktop\CS 
5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py
(316) is hoisted out of the parallel loop labelled #5 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: local_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\mlock\Desktop\CS 5781 Machine Learning
Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (335)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\mlock\Desktop\CS 5781 Machine Learning Engineering\workspace\mod3-CSK0227\minitorch\fast_ops.py (335)
--------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                          |
    out: Storage,                                                                     |
    out_shape: Shape,                                                                 |
    out_strides: Strides,                                                             |
    a_storage: Storage,                                                               |
    a_shape: Shape,                                                                   |
    a_strides: Strides,                                                               |
    b_storage: Storage,                                                               |
    b_shape: Shape,                                                                   |
    b_strides: Strides,                                                               |
) -> None:                                                                            |
    """NUMBA tensor matrix multiply function.                                         |
                                                                                      |
    Should work for any tensor shapes that broadcast as long as                       |
                                                                                      |
    ```                                                                               |
    assert a_shape[-1] == b_shape[-2]                                                 |
    ```                                                                               |
                                                                                      |
    Optimizations:                                                                    |
                                                                                      |
    * Outer loop in parallel                                                          |
    * No index buffers or function calls                                              |
    * Inner loop should have no global writes, 1 multiply.                            |
                                                                                      |
                                                                                      |
    Args:                                                                             |
    ----                                                                              |
        out (Storage): storage for `out` tensor                                       |
        out_shape (Shape): shape for `out` tensor                                     |
        out_strides (Strides): strides for `out` tensor                               |
        a_storage (Storage): storage for `a` tensor                                   |
        a_shape (Shape): shape for `a` tensor                                         |
        a_strides (Strides): strides for `a` tensor                                   |
        b_storage (Storage): storage for `b` tensor                                   |
        b_shape (Shape): shape for `b` tensor                                         |
        b_strides (Strides): strides for `b` tensor                                   |
                                                                                      |
    Returns:                                                                          |
    -------                                                                           |
        None : Fills in `out`                                                         |
                                                                                      |
    """                                                                               |
    # # Basic compatibility check #added                                              |
    # assert a_shape[-1] == b_shape[-2]                                               |
                                                                                      |
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                          |
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                          |
                                                                                      |
    ## TODO: Implement for Task 3.2.                                                  |
    #raise NotImplementedError("Need to implement for Task 3.2")                      |
                                                                                      |
                                                                                      |
    #Multi S                                                                          |
    # Basic compatibility check                                                       |
    assert a_shape[-1] == b_shape[-2]                                                 |
                                                                                      |
    # Get batch strides (0 if not batched)                                            |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                            |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                            |
                                                                                      |
    # Get key dimensions                                                              |
    batch_size = max(out_shape[0], 1)    # Number of batches                          |
    M = out_shape[1]                     # Rows in output                             |
    N = out_shape[2]                     # Cols in output                             |
    K = a_shape[-1]                      # Shared dimension (cols in A, rows in B)    |
                                                                                      |
    # Parallel over both batch and M dimensions for better utilization                |
    for batch in prange(batch_size):--------------------------------------------------| #6
        for row in range(M):                                                          |
            for col in range(N):                                                      |
                a_pos = (                                                             |
                    batch * a_batch_stride + row * a_strides[-2]                      |
                )                                                                     |
                b_pos = (                                                             |
                    batch * b_batch_stride + col * b_strides[-1]                      |
                )                                                                     |
                                                                                      |
                acc = 0.0                                                             |
                for _ in range(K):                                                    |
                    acc += (                                                          |
                        a_storage[a_pos] * b_storage[b_pos]                           |
                    )                                                                 |
                    a_pos += a_strides[-1]                                            |
                    b_pos += b_strides[-2]                                            |
                                                                                      |
                out_pos = (                                                           |
                    batch * out_strides[0]                                            |
                    + row * out_strides[-2]                                           |
                    + col * out_strides[-1]                                           |
                )                                                                     |
                                                                                      |
                out[out_pos] = acc                                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #6).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

