import sys
from numba import cuda

@cuda.jit(device=True)
def getGlobalId(dimensionGrid):
    if dimensionGrid == 1:
        global_id = (cuda.blockIdx.x * cuda.blockDim.x) + cuda.threadIdx.x
    elif dimensionGrid == 2:
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        total_width = cuda.gridDim.x * cuda.blockDim.x
        global_id = y * total_width + x
    return global_id

@cuda.jit
def kernel(dimensionGrid):
    print("Thread ID x :", cuda.threadIdx.x, "Thread ID y :", cuda.threadIdx.y, "Thread ID z :", cuda.threadIdx.z,
          "Global ID :", getGlobalId(dimensionGrid))


def run(blocksPerGrid, threadsPerBlock, dimensionGrid):
    print("Starting", sys._getframe().f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    kernel[blocksPerGrid, threadsPerBlock](dimensionGrid)
    cuda.synchronize()

# 2D Grid, 4 2D blocks of 14 threads
run((2, 2), (2, 7), 2)