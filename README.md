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

1. Open a Google Colab notebook and connect to a GPU.
2. Download, build, and inspect the original hello example.
3. Run vector addition and explore thread indexing.
4. Adapt the vector example to multiply square matrices.
5. Save your source files, results, and explanations.

The source files come directly from Yonghong Yan's CSCE 569 course. The matrix
exercise adds a row-column dot product and a matching result check.

## 1. Create your Colab notebook

Google Colab is a notebook that runs code on a remote computer through your web
browser. For this lab, that computer will have an NVIDIA GPU. You will write CUDA
source files and compile them using `nvcc`.

1. Sign in to your Google account and open [Google Colab](https://colab.research.google.com/).
2. Choose **New notebook**. If a notebook is already open, use **File → New notebook**.
3. Click the notebook name near the top left and rename it to
   `YourName_GPU_Lab.ipynb`.
4. Choose **Runtime → Change runtime type**.
5. Select a **GPU** under **Hardware accelerator**. Choose **T4 GPU** if that
   option is available. Keep **Python 3** as the runtime language.
6. Click **Save**, then **Connect** near the top right if a connection is needed.
   Wait for the runtime to connect.

Google provides GPU access subject to availability. If Colab cannot allocate a
GPU, tell the instructor.

### How to use notebook cells

A notebook contains **code cells** and **text cells**.

- Click **+ Code** to add a code cell.
- Paste a complete notebook-cell block from these instructions into that cell.
  Copy the contents inside the code box.
- Click the triangular run button on the left of the cell, or press
  **Shift+Enter**, to execute it.
- Wait for execution to finish. Results appear immediately below the cell.
- Use **+ Text** to record observations and answers.

Run cells in the order given. Several cells begin with `%%bash`. This tells Colab
to run the cell as Linux shell commands. Cells beginning with `%%writefile` save
the following lines into a source file. Put either command on the very first
line of its cell.

Blocks labeled **CUDA replacement** below belong inside the relevant source-code
cell. The surrounding instructions identify which part of the file to replace.

### Check the GPU and compiler

Add a code cell, paste this complete block, and run it.

```bash
%%bash
set -e
nvidia-smi
nvcc --version
```

You should see a table identifying the NVIDIA GPU and a version report from the
CUDA compiler. Record the GPU model in a text cell. If either command reports
an error, ask the instructor for help before continuing.

The line `set -e` stops a shell cell when one of its commands fails.

## 2. Download the original course files

Add and run this code cell. It creates `/content/gpu-lab` on the Colab computer
and downloads the three files needed for the lab. The file-existence check lets
you rerun this cell while preserving files you have already edited.

```bash
%%bash
set -e
mkdir -p /content/gpu-lab
cd /content/gpu-lab

for filename in hello-1.cu vectorAdd.cu common.h
do
    if [ ! -f "$filename" ]; then
        curl -fsSL "https://passlab.github.io/CSCE569/resources/gpu_code_examples/$filename" -o "$filename.download"
        mv "$filename.download" "$filename"
    fi
done

ls -l hello-1.cu vectorAdd.cu common.h
```

| File | Use in this lab |
| --- | --- |
| `hello-1.cu` | Introduces kernel launches and thread IDs |
| `vectorAdd.cu` | Supplies vector addition and the host program used for the matrix exercise |
| `common.h` | Supplies the course's `CHECK` macro for CUDA error reporting |

Each compile cell below includes `cd /content/gpu-lab` because each `%%bash`
cell starts a new shell.

## 3. Edit and run the hello example

### Put the complete source into an editable notebook cell

Add and run this display cell.

```python
!cat /content/gpu-lab/hello-1.cu
```

Then create your **hello source cell**.

1. Click **+ Code**.
2. Type `%%writefile /content/gpu-lab/hello-1.cu` on its first line.
3. Copy the **entire source file** from the display cell's output and paste it
   below that first line. Include the headers, kernel, and `main` function.
4. Run the source cell. Colab should report that it is writing or overwriting
   `hello-1.cu`.

Keep this source cell as the place where you edit the hello program. Its full
contents are saved with your notebook. Running it writes the current version
into the file that the compiler will read.

### Compile and run

Add a code cell after the source cell, paste this block, and run it.

```bash
%%bash
set -e
cd /content/gpu-lab
nvcc -arch=native hello-1.cu -o hello-1
./hello-1
```

The first command compiles the source. The second runs the resulting program.
The flag `-arch=native` generates code for the GPU assigned to your Colab runtime.

The original program launches 10 blocks with 10 threads per block. Predict the
number of times it prints `Hello World!`, then check the output.

### Display each thread's identity

In your hello source cell, change `num_blocks` to 2 and `num_threads` to 4.
Replace the `printf` inside `hellokernel` with this **CUDA replacement**.

```cuda
printf("block=%u thread=%u global=%u\n",
       blockIdx.x, threadIdx.x,
       blockIdx.x * blockDim.x + threadIdx.x);
```

Run the **source cell first**, then run the **compile-and-run cell** again.
Repeat those two steps after every source edit throughout the lab.

Find eight thread records with global indices from 0 through 7. Their printed
order can vary.

| CUDA variable | Meaning |
| --- | --- |
| `blockIdx.x` | This block's index in the grid |
| `blockDim.x` | The number of threads along the block's x dimension |
| `threadIdx.x` | This thread's index within its block |

In a text cell, explain why the global index includes both a block index and a
thread index. Predict the output count for three blocks of four threads.

## 4. Explore the original vector example

Add and run this display cell.

```python
!cat /content/gpu-lab/vectorAdd.cu
```

Create a **vector source cell** using the same procedure as for hello. Its first
line must be `%%writefile /content/gpu-lab/vectorAdd.cu`. Paste the complete
original file below that line, retaining its source notice.

The `vectorAdd` kernel assigns one output element to each valid thread and
computes `C[i] = A[i] + B[i]`.

### Add error reporting using the course helper

Make the following edits in the vector source cell.

Add these **CUDA replacement lines** immediately after the existing
`#include <cuda_runtime.h>` line.

```cuda
#include <stdlib.h>
#include <math.h>
#include "common.h"
```

The original program assigns CUDA return codes to `err`. After each statement
of the form `err = cuda...;`, add `CHECK(err);`. The course helper displays an
error and stops if a CUDA operation fails. For example, an allocation and its
check should look like this.

```cuda
err = cudaMalloc((void **)&d_A, size);
CHECK(err);
```

Immediately after the kernel launch, the existing `cudaGetLastError()` line and
the checks should look like this **CUDA replacement**.

```cuda
err = cudaGetLastError();
CHECK(err);
CHECK(cudaDeviceSynchronize());
```

The synchronization waits for the GPU computation to finish and reports execution
errors. Run the vector source cell to save your edits.

### Build, run, and vary the launch

Add and run this compile-and-run cell.

```bash
%%bash
set -e
cd /content/gpu-lab
nvcc -O2 -arch=native vectorAdd.cu -o vectorAdd
./vectorAdd
```

The original program uses `numElements = 50000` and `threadsPerBlock = 256`.
A successful run prints `Test PASSED`.

1. Find the expression that calculates `blocksPerGrid`. Explain why it rounds up.
2. In the vector source cell, change `numElements` to 1000. Run the source cell,
   then the compile-and-run cell.
3. Calculate the number of blocks, total launched threads, and threads outside
   the valid array range.
4. Change `threadsPerBlock` to 128. Run both cells and repeat your calculation.
5. Restore `threadsPerBlock` to 256. Run both cells once more.

Each run should print `Test PASSED`. Record the counts and whether each run passed
in a text cell before moving to the next configuration.

| Vector length | Threads per block | Blocks | Total threads | Extra threads | Passed? |
| --- | --- | --- | --- | --- | --- |
| 50000 | 256 | | | | |
| 1000 | 256 | | | | |
| 1000 | 128 | | | | |

Find the five host-side stages in `main`.

1. Allocate GPU memory.
2. Copy the input arrays to the GPU.
3. Launch the kernel.
4. Copy the result back to the CPU.
5. Release GPU memory.

## 5. Extend vector addition to matrix multiplication

Create a new **matrix source cell**.

1. Copy the complete contents of your edited vector source cell into a new
   code cell.
2. Change its first line to `%%writefile /content/gpu-lab/matrixLab.cu`.
3. Make the CUDA edits below within this new source cell.

Each square matrix will occupy a flat array of `MATRIX_N * MATRIX_N` elements.
The existing memory allocations and transfers can serve the matrix exercise.

For a hand calculation, use the following matrices.

| Matrix | Row 0 | Row 1 |
| --- | --- | --- |
| A | 1, 2 | 3, 4 |
| B | 5, 6 | 7, 8 |
| C = A × B | 19, 22 | 43, 50 |

For example, `C[0, 0] = 1*5 + 2*7 = 19`. Calculate `C[0, 1]` yourself and record
it in a text cell. The program retains the original random input initialization
and checks its own inputs against a CPU calculation.

### Set the matrix size

Add this **CUDA replacement** after the includes in the matrix source cell.

```cuda
#ifndef MATRIX_N
#define MATRIX_N 16
#endif
```

In `main`, replace the assignment to `numElements` with the following.

```cuda
int numElements = MATRIX_N * MATRIX_N;
```

The compile cells below override `MATRIX_N` to test different sizes. Use
`threadsPerBlock = 256` for this exercise.

### Replace the GPU kernel

Replace the entire `vectorAdd` kernel with this **CUDA replacement**. Complete
the three expressions marked `TODO`.

```cuda
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

Each thread computes one whole dot product. The loop over `k` runs inside that
thread.

In `main`, rename the kernel call to match the new function.

```cuda
matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

The one-dimensional launch assigns threads to flat output indices. The kernel
converts each index into a matrix row and column.

Remove the unused `vectorAdd_cpu` function near the top of the matrix source
cell.

### Replace the result check

In `main`, replace the verification loop immediately following the device-to-host
copy with this **CUDA replacement**. This CPU reference calculation is supplied
for you to copy into the matrix source cell.

```cuda
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

The existing `Test PASSED` message follows this check. You can also replace the
opening status message with the following.

```cuda
printf("[Matrix multiplication: %d by %d]\n", MATRIX_N, MATRIX_N);
```

After completing the three kernel expressions and all edits above, run the
matrix source cell to write `matrixLab.cu`.

### Build and check the matrix program

Create **three separate code cells** for the following tests. Keeping their
outputs in separate cells records all three results in your notebook.

Test the width-2 case.

```bash
%%bash
set -e
cd /content/gpu-lab
nvcc -O2 -arch=native -DMATRIX_N=2 matrixLab.cu -o matrixLab
./matrixLab
```

Test the width-16 case.

```bash
%%bash
set -e
cd /content/gpu-lab
nvcc -O2 -arch=native -DMATRIX_N=16 matrixLab.cu -o matrixLab
./matrixLab
```

Test the width-17 case.

```bash
%%bash
set -e
cd /content/gpu-lab
nvcc -O2 -arch=native -DMATRIX_N=17 matrixLab.cu -o matrixLab
./matrixLab
```

Each test should print `Test PASSED`. The width-17 case has 289 outputs and
exercises the boundary check. If you edit the matrix source again, run its source
cell first, then rerun these three test cells.

## 6. Explain the architecture

Add a text cell and answer these questions in a few sentences each.

1. For width 17 and 256 threads per block, how many blocks and threads launch?
   How many threads own valid output elements?
2. Why can each matrix thread compute its result without coordinating updates
   to C with other threads?
3. Give one reason a GPU can help with a large matrix product, and one reason
   a tiny product may receive little benefit.

If you finish early, test width 64 or describe how to express the same output
mapping using a two-dimensional block and grid.

```bash
%%bash
set -e
cd /content/gpu-lab
nvcc -O2 -arch=native -DMATRIX_N=64 matrixLab.cu -o matrixLab
./matrixLab
```

## 7. Save and download your work

Your notebook stores the source-code cells, text answers, and saved outputs in
Google Drive. The files under `/content` belong to the temporary runtime and
can disappear when that runtime is deleted. Keep the complete source in your
three `%%writefile` cells so it can be recreated.

1. Run each final source cell and its tests, and finish your text answers.
2. Add and run the following Python code cell to download a ZIP containing
   your source files.
3. Choose **File → Save** and allow the notebook to finish saving.
4. Use **File → Download → Download .ipynb** to download a notebook copy.
   The menu may label the last choice simply **.ipynb**.

```python
from pathlib import Path
from zipfile import ZipFile
from google.colab import files

lab_directory = Path("/content/gpu-lab")
archive_path = Path("/content/gpu_lab_submission.zip")

with ZipFile(archive_path, "w") as archive:
    for filename in ["hello-1.cu", "vectorAdd.cu", "matrixLab.cu", "common.h"]:
        archive.write(lab_directory / filename, arcname=filename)

files.download(str(archive_path))
```

If the browser asks for download permission, allow this download.

Submit the downloaded notebook and `gpu_lab_submission.zip`. Your notebook
should include the GPU model, vector observation table, successful matrix-test
outputs, and answers to the architecture questions.

## Common problems

| What you see | What to do |
| --- | --- |
| GPU unavailable or `nvidia-smi` fails | Check **Runtime → Change runtime type** and select a GPU. Tell the instructor if Colab cannot allocate one. |
| `nvcc: command not found` | Show the instructor the environment-check output. |
| A Python syntax error on CUDA or shell code | Check the cell's first line. Use `%%writefile` for a complete CUDA source file and `%%bash` for the provided shell cells. |
| Source file missing | Run the download cell and the relevant `%%writefile` source cell. |
| The program still behaves like the previous version | Run the edited source cell, then its compile-and-run cell. |
| Compiler errors | Read the first reported error and its source line. Check the surrounding braces and the replacement location. |
| Matrix mismatch | Recheck the row, column, and product expressions. Run the source cell and tests again after correcting them. |
| The runtime reconnects with an empty working directory | Rerun the environment and download cells, then your saved source cells and compile cells. |

## Further reading

The original matrix assignment provides additional work with shared memory,
cuBLAS, and performance analysis. Its `matmul_base` function shows the sequential
row-column dot product used in this lab.

- [Original GPU examples](https://passlab.github.io/CSCE569/resources/gpu_code_examples/)
- [Original matrix source](https://passlab.github.io/CSCE569/resources/gpu_code_examples/matmul.cu)
- [Original Assignment 4](https://passlab.github.io/CSCE569/Assignment_4/)
- [NVIDIA introduction to CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [NVIDIA compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [IPython notebook commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)
