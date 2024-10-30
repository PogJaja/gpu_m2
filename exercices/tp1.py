import math
import sys

import numpy as np
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

@cuda.jit
def fillTab(tab):
    gx = cuda.grid(1)
    if gx < len(tab):
        tab[gx] = cuda.threadIdx.x

#Taille du tableau
N=500
#Taille des blocks
TB=32
#Taille de la grille
Grid=math.ceil(N/TB)
#Création du tableau
tab = np.empty(N, dtype=np.uint32)

d_tab = cuda.to_device(tab)
fillTab[Grid, TB](d_tab)
cuda.synchronize()
tab = d_tab.copy_to_host()
print(tab)

# 2D Grid, 4 2D blocks of 14 threads
#run((2, 2), (2, 7), 2)