# CUDA: multiplication matrix

## Task

1. Create algoritm for multiplication matrix with CUDA.

2. Compare executable time between GPU (CUDA) and CPU.

3. Reseach work for created algoritm for CUDA.

## Uses

1. Compile executable program

    ```console
        make mm_cuda
    ```

2. Run program

    ```console
        ./mm_cuda
    ```

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | Intel® Core™ i7-8750H CPU @ 2.20GHz (Turbo Boost  4.10 GHz) × 12 |
| RAM  | 16 GB DDR4 |
| GPU  | GeForce GTX 1060 with Max-Q Design/PCIe/SSE2 |
| OS type | 64-bit  |

## Reseach

"SIZE" is mean size of multiplier size.
"CPU" is mean time for ijk-algoritm.
"CUDA" is mean time for created algoritm for GPU.
All time is average time in seconds.
Multipliers have float type for elements.

| SIZE  | CPU | CUDA |
|-------|---------|----|
| 128x128 |0.026391000|0.000690112|
| 256x256 |0.109841000|0.003494624|
| 512x512 |0.846972000|0.024500128|
| 1024x1024 |7.995466000|0.181961700|
| 2048x2048 |160.313562000|1.194571257|
