import sys

from numba import cuda


@cuda.jit
def printCoordinates():
    global_id = cuda.grid(1)
    print("Thread ID x :", cuda.threadIdx.x, "Thread ID y :", cuda.threadIdx.y, "Thread ID z :", cuda.threadIdx.z,
          "Global ID :", global_id)


def executePrintCoordinates(blocksPerGrid, threadsPerBlock):
    print("Starting", sys._getframe().f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    printCoordinates[blocksPerGrid, threadsPerBlock]()
    cuda.synchronize()


# 1D Grid, 1 1D block of 1 thread
executePrintCoordinates(1, 1)
# 1D Grid, 1 1D block of 16 threads
executePrintCoordinates(1, 16)
# 1D Grid, 2 1D block of 1 thread
executePrintCoordinates(2, 1)
# 1D Grid, 1 3D block of 16 threads
executePrintCoordinates(1, (4, 2, 2))
