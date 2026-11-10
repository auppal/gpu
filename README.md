# Computer Architecture and Design ECE 4535

## GPUs and CUDA Programming

November 10, 2026
Presented by Ahsen Uppal

This guest lecture covers GPU history, the architectural differences between
CPUs and GPUs, and the CUDA programming model. The lab uses the original course
examples and extends the vector example into a basic matrix multiplication.

## Lecture resources

The primary resource is Yonghong Yan's CSCE 569 Parallel Computing course at the
University of South Carolina. Follow the material for classes 23 and 24.

- [CSCE 569 course page](https://passlab.github.io/CSCE569/)
- [GPU Architectures and Programming, Part 1](https://passlab.github.io/CSCE569/notes/lecture_GPUArchCUDA01.pdf)
- [GPU Architectures and Programming, Part 2](https://passlab.github.io/CSCE569/notes/lecture_GPUArchCUDA02-CUDAMem.pdf)

## Lab overview

1. Download, build, and inspect the hello example
2. Run vector addition and explore thread indexing
3. Adapt the vector example to multiply square matrices |
4. Check results and explain the architecture |

The source files come directly from the original Yan's CSCE 569 course.
The matrix exercise adds a row-column dot product and a matching result check.

## Download the course examples

Log in to `orion.ece.seas.gwu.edu` with your account. Replace
`YOUR_USERNAME` with your username.

```sh
ssh YOUR_USERNAME@orion.ece.seas.gwu.edu
```

Use the CUDA `PATH` or module settings supplied in class, then check the compiler.

```sh
nvcc --version
```

Create a working directory and download three original course files.

```sh
mkdir -p gpu-lab
cd gpu-lab
curl -fLO https://passlab.github.io/CSCE569/resources/gpu_code_examples/hello-1.cu
curl -fLO https://passlab.github.io/CSCE569/resources/gpu_code_examples/vectorAdd.cu
curl -fLO https://passlab.github.io/CSCE569/resources/gpu_code_examples/common.h
```

| File | Use in this lab |
| --- | --- |
| `hello-1.cu` | Introduces kernel launches and thread IDs |
| `vectorAdd.cu` | Supplies vector addition and the host program used for the matrix exercise |
| `common.h` | Supplies the course's `CHECK` macro for CUDA error reporting |

The complete directory also contains matrix and other examples for further
reading. [Original GPU examples](https://passlab.github.io/CSCE569/resources/gpu_code_examples/)

## Part 1 — Identify the threads

Build and run the original hello example.

```sh
nvcc hello-1.cu -o hello-1
./hello-1
```

The original program launches 10 blocks with 10 threads per block. Predict the
number of times it prints `Hello World!`.

Change `num_blocks` to 2 and `num_threads` to 4. Replace the `printf` inside
`hellokernel` with the following statement, then rebuild and run.

```cpp
printf("block=%u thread=%u global=%u\n",
       blockIdx.x, threadIdx.x,
       blockIdx.x * blockDim.x + threadIdx.x);
```

Find eight thread records with global indices from 0 through 7. Their printed
order can vary.

| CUDA variable | Meaning |
| --- | --- |
| `blockIdx.x` | This block's index in the grid |
| `blockDim.x` | The number of threads along the block's x dimension |
| `threadIdx.x` | This thread's index within its block |

Explain why the global index includes both a block index and a thread index.
Predict the output count if the launch uses three blocks of four threads.

## Part 2 — Explore the original vector example

Open `vectorAdd.cu`. Its `vectorAdd` kernel assigns one output element to each
valid thread and computes `C[i] = A[i] + B[i]`.

### Add error reporting using the course helper

The original program assigns CUDA return codes to `err`. To make failures visible,
add these includes after the existing `cuda_runtime.h` include.

```cpp
#include <stdlib.h>
#include <math.h>
#include "common.h"
```

After each statement of the form `err = cuda...;`, add `CHECK(err);`. The course
helper will display an error and stop if a CUDA operation fails. For example,

```cpp
err = cudaMalloc((void **)&d_A, size);
CHECK(err);
```

After the kernel launch and its `cudaGetLastError()` check, also wait for kernel
completion so execution errors are reported before checking the result.

```cpp
err = cudaGetLastError();
CHECK(err);
CHECK(cudaDeviceSynchronize());
```

These reporting edits will carry over when you copy this file for Part 3.

### Build, run, and vary the launch

```sh
nvcc -O2 vectorAdd.cu -o vectorAdd
./vectorAdd
```

The original program uses `numElements = 50000` and `threadsPerBlock = 256`.
A successful run prints `Test PASSED`.

1. Find the expression that calculates `blocksPerGrid`. Explain why it rounds up.
2. Change `numElements` to 1000, rebuild, and run again.
3. Calculate the number of blocks, total launched threads, and threads outside
   the valid array range.
4. Change `threadsPerBlock` to 128, rebuild, and repeat your calculation.
5. Restore `threadsPerBlock` to 256 for the matrix exercise.

Each run should print `Test PASSED`. These examples take their sizes from the
source code, so rebuild after editing those values.

Find the five host-side stages in `main`.

1. Allocate GPU memory.
2. Copy the input arrays to the GPU.
3. Launch the kernel.
4. Copy the result back to the CPU.
5. Release GPU memory.

## Part 3 — Extend vector addition to matrix multiplication

Make a working copy of the vector program.

```sh
cp vectorAdd.cu matrixLab.cu
```

Each square matrix will occupy a flat array of `MATRIX_N * MATRIX_N` elements.
The existing memory allocations and transfers can therefore serve the matrix
exercise. Keep the original source's copyright notice with your working copy.

For a hand calculation, use the following matrices.

| Matrix | Row 0 | Row 1 |
| --- | --- | --- |
| A | 1, 2 | 3, 4 |
| B | 5, 6 | 7, 8 |
| C = A × B | 19, 22 | 43, 50 |

For example, `C[0, 0] = 1*5 + 2*7 = 19`. Calculate `C[0, 1]` yourself. The
program retains the original example's random input initialization and checks
its own inputs against a CPU calculation.

### Set the matrix size

Add the following after the includes in `matrixLab.cu`.

```cpp
#ifndef MATRIX_N
#define MATRIX_N 16
#endif
```

In `main`, replace the assignment to `numElements` with the following.

```cpp
int numElements = MATRIX_N * MATRIX_N;
```

The build commands below can override `MATRIX_N` to test different sizes.

### Replace the GPU kernel

Replace the entire `vectorAdd` kernel with this scaffold. Complete the three
marked expressions.

```cpp
__global__ void matrixMul(float *A, const float *B, float *C,
                          int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        int row = 0; // TODO: derive the row from i and MATRIX_N.
        int col = 0; // TODO: derive the column from i and MATRIX_N.
        float sum = 0.0f;

        for (int k = 0; k < MATRIX_N; ++k)
        {
            sum += 0.0f; // TODO: multiply A[row,k] by B[k,col].
        }

        C[i] = sum;
    }
}
```

In row-major storage, the element at row `r`, column `c` has offset
`r * MATRIX_N + c`. Integer division of a flat index by `MATRIX_N` gives its
row. The remainder gives its column.

Each thread computes one whole dot product. The loop over `k` runs inside the
thread.

In `main`, rename the kernel call to match the new function.

```cpp
matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

The one-dimensional launch assigns threads to flat output indices. The kernel
converts each index into a matrix row and column. This reuses the launch geometry
from the vector example.

The original `vectorAdd_cpu` function near the top of the file is unused by this
program. Remove that function from `matrixLab.cu`.

### Replace the result check

In `main`, replace the verification loop immediately following the device-to-host
copy with the following CPU reference calculation. This code is supplied for you
to copy into your program.

```cpp
for (int i = 0; i < numElements; ++i)
{
    int row = i / MATRIX_N;
    int col = i % MATRIX_N;
    double expected = 0.0;

    for (int k = 0; k < MATRIX_N; ++k)
    {
        expected += (double)h_A[row * MATRIX_N + k]
                  * h_B[k * MATRIX_N + col];
    }

    if (!isfinite(h_C[i]) ||
        fabs(h_C[i] - expected) > 1e-4 * (1.0 + fabs(expected)))
    {
        fprintf(stderr, "Mismatch at C[%d,%d]: got %g, expected %g\n",
                row, col, h_C[i], expected);
        exit(EXIT_FAILURE);
    }
}
```

The existing `Test PASSED` message follows this check. You can also change the
opening status message to identify the matrix exercise.

```cpp
printf("[Matrix multiplication: %d by %d]\n", MATRIX_N, MATRIX_N);
```

### Build and check the matrix program

Complete the three kernel expressions, then run these tests. Each command pair
builds a particular matrix size and runs that executable.

```sh
nvcc -O2 -DMATRIX_N=2 matrixLab.cu -o matrixLab
./matrixLab

nvcc -O2 -DMATRIX_N=16 matrixLab.cu -o matrixLab
./matrixLab

nvcc -O2 -DMATRIX_N=17 matrixLab.cu -o matrixLab
./matrixLab
```

Each run should print `Test PASSED`. The width-17 case has 289 outputs and
exercises the boundary check when using 256 threads per block.

## Part 4 — Explain the architecture

Answer these questions in a few sentences each.

1. For width 17 and 256 threads per block, how many blocks and threads launch?
   How many threads own valid output elements?
2. Why can each matrix thread compute its result without coordinating updates
   to C with other threads?
3. Give one reason a GPU can help with a large matrix product, and one reason
   a tiny product may receive little benefit.

If you finish early, test width 64 or describe how you could express the same
output mapping using a two-dimensional block and grid.

```sh
nvcc -O2 -DMATRIX_N=64 matrixLab.cu -o matrixLab
./matrixLab
```

## Submission

Submit your modified `hello-1.cu`, `vectorAdd.cu`, and `matrixLab.cu`, successful
output for the vector and matrix tests, and answers to the three architecture
questions.

## Further reading

The original matrix assignment provides additional work with shared memory,
cuBLAS, and performance analysis. Its `matmul_base` function shows the sequential
row-column dot product used in this lab.

- [Original matrix source](https://passlab.github.io/CSCE569/resources/gpu_code_examples/matmul.cu)
- [Original Assignment 4](https://passlab.github.io/CSCE569/Assignment_4/)
- [NVIDIA introduction to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
