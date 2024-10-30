from numba import cuda
import numba as nb
import sys


@cuda.jit
def printCoordinates():
    global_id = cuda.grid(1)
    print("Thread ID x :", cuda.threadIdx.x, "Thread ID y :", cuda.threadIdx.y, "Thread ID z :", cuda.threadIdx.z, "Global ID :", global_id)

def question1():
    # 1 block of 1 thread
    threadsPerBlock = 1
    blocksPerGrid = 1
    print("Starting", sys._getframe().f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    printCoordinates[blocksPerGrid, threadsPerBlock]()
    cuda.synchronize()



if __name__ == '__main__':
    question1()