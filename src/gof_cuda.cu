#include "gof_cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <algorithm>
#include <iostream>
#include <format>

void createDeviceMap(const bMap_t& h_map, bMap_t& d_map) {
    d_map = {};
    d_map.x = h_map.x; d_map.y = h_map.y;
    const size_t size = d_map.x * d_map.y;

    cudaMalloc(&d_map.board, d_map.x * d_map.y);
    cudaMalloc(&d_map.board2, d_map.x * d_map.y); 
    
    cudaMemcpy(d_map.board, h_map.board, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_map.board2, h_map.board2, size, cudaMemcpyHostToDevice);

    return;
}

//WARNING: assumes that h_map has allocated enough space
void cpMapFromDevice(bMap_t& h_map, const bMap_t& d_map) {
    const size_t size = d_map.x * d_map.y;

    cudaMemcpy(h_map.board, d_map.board, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_map.board2, d_map.board2, size, cudaMemcpyDeviceToHost);

    return;
}

void freeDeviceMap(bMap_t& _map) {
    cudaFree(_map.board);
    cudaFree(_map.board2);

    return;
}

inline __device__ bool& mapAt(bMap_t& _map, size_t x, size_t y) {
    return _map.board[x + y * _map.x];
}

inline __device__ bool& mapAt2(bMap_t& _map, size_t x, size_t y) {
    return _map.board2[x + y * _map.x];
}

__global__ void d_gof(bMap_t _map) {
    const int32_t id_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t id_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_x >= _map.x || id_y >= _map.y) return;

    int8_t neighbors = 0;
    int8_t cell = mapAt(_map, id_x, id_y);
    
    for(int32_t x = -1; x < 2; x++)
        for (int32_t y = -1; y < 2; y++) {
            if (x == y && y == 0) continue;

            if(id_x + x >= 0 && id_x + x < _map.x && 
                id_y + y >= 0 && id_y + y < _map.y)
                neighbors += mapAt(_map, id_x + x, id_y + y);
        }

    if (neighbors == 3) cell = 1;
    else if (cell && neighbors == 2) cell = 1;
    else cell = 0;

    mapAt2(_map, id_x, id_y) = cell;
	return;
}

__global__ void d_gof_shared(bMap_t _map) {
    //max block size is 1024 threads -> we can get maximum of 32 side length with 34 necessary cell accesses -> we need 1024 + 4 * 32 + 4 = 1156 booleans (maximum)
	//WARNING: NOT 0 INITIALIZED COZ DYNAMIC
    extern __shared__ bool sharedCells[];  
    
    //local and global coords
    const int32_t lx = threadIdx.x, ly = threadIdx.y;
    const int32_t gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t gy = threadIdx.y + blockDim.y * blockIdx.y;
    
    const uint32_t toClean = std::ceil((blockDim.x+2.0) * (blockDim.y+2.0) / (blockDim.x * blockDim.y)); 
    const uint32_t startPoint = (lx + ly*blockDim.x) * toClean;
    for(uint32_t i = 0; i < toClean; i++) {
        if(startPoint + i >= ((blockDim.x+2) * (blockDim.y+2)))
            break;
        sharedCells[startPoint + i] = 0;
    }

    __syncthreads();

    if(gx >= _map.x || gy >= _map.y)
        return;
    
    const bool x0 = lx == 0, y0 = ly == 0,
          xm = lx == blockDim.x-1, ym = ly == blockDim.y-1;
    //edge x=0, x=max etc
    const bool ex0 = x0 && gx != 0, exm = xm && gx < _map.x-1, 
          ey0 = y0 && gy != 0, eym = ym && gy < _map.y-1;

    const int32_t rowSize = blockDim.x+2;

    sharedCells[lx+1 + (ly+1)*rowSize] = mapAt(_map, gx, gy);
    
    if(ex0)
        sharedCells[0 + (ly+1)*rowSize] = mapAt(_map, gx-1, gy);
    if(exm)
        sharedCells[lx+2 + (ly+1)*rowSize] = mapAt(_map, gx+1, gy);
    if(ey0)
        sharedCells[lx+1 + 0*rowSize] = mapAt(_map, gx, gy-1);
    if(eym)
        sharedCells[lx+1 + (ly+2)*rowSize] = mapAt(_map, gx, gy+1);
    
    if(ex0 && ey0)
        sharedCells[0] = mapAt(_map, gx-1, gy-1);
     if(ex0 && eym)
        sharedCells[0 + (ly+2) * rowSize] = mapAt(_map, gx-1, gy+1);
    if(exm && eym)
        sharedCells[lx+2 + (ly+2) * rowSize] = mapAt(_map, gx+1, gy+1);
    if(exm && ey0)
        sharedCells[lx+2 + 0 * rowSize] = mapAt(_map, gx+1, gy-1);

    __syncthreads();
    
    int8_t neighbors = 0;
    int8_t cell = sharedCells[lx+1 + (ly+1)*rowSize];

    for(int32_t x = -1; x < 2; x++)
        for (int32_t y = -1; y < 2; y++) {
            if (x == y && y == 0) continue;
            neighbors += sharedCells[1+lx+x + (1+ly+y)*rowSize];
        }

    if (neighbors == 3) cell = 1;
    else if (cell && neighbors == 2) cell = 1;
    else cell = 0;

    mapAt2(_map, gx, gy) = cell;

	return;
}

__host__ float iterateGof(bMap_t& d_map, size_t _threads_x, size_t _threads_y, size_t _iterations) {
    dim3 dim = { (uint32_t)_threads_x, (uint32_t)_threads_y, 1 };
    dim3 blocks = { (uint32_t)ceil(float(d_map.x) / _threads_x), (uint32_t)ceil(float(d_map.y) / _threads_y), 1 };

    float res = 0, ms; //ms
    cudaEvent_t ev1,ev2;
    cudaEventCreate(&ev1,0);
    cudaEventCreate(&ev2,0);

    //std::cout<<std::format("d:{},{} b:{},{}\n", dim.x,dim.y,blocks.x,blocks.y);
    for (size_t i = 0; i < _iterations; i++) {
        cudaEventRecord(ev1);
        d_gof<<<blocks, dim>>>(d_map);
        cudaEventRecord(ev2);
        cudaEventSynchronize(ev2);
        cudaEventElapsedTime(&ms, ev1, ev2);
        res += ms;
        //new state is stored in board2 -> we swap these
        std::swap(d_map.board, d_map.board2);
    }
    //if(cudaError_t e = cudaGetLastError(); e != cudaSuccess) 
    //    std::cout<<std::format("ERROR {}\n", (int)e);

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    return res;
}


__host__ void iterateGof_shared(bMap_t& d_map, size_t _threads_x, size_t _threads_y, size_t _iterations) {
    dim3 dim = { (uint32_t)_threads_x, (uint32_t)_threads_y, 1 };
    dim3 blocks = { (uint32_t)ceil(double(d_map.x) / _threads_x), (uint32_t)ceil(double(d_map.y) / _threads_y), 1 };
    size_t sharedMemSize = (_threads_x+2) * (_threads_y+2);

    //std::cout<<std::format("d:{},{},{} b:{},{},{}\n", dim.x,dim.y,dim.z,blocks.x,blocks.y,blocks.z);
    for (size_t i = 0; i < _iterations; i++) {
        d_gof_shared<<<blocks, dim, sharedMemSize>>>(d_map);
        //new state is stored in board2 -> we swap these
        std::swap(d_map.board, d_map.board2);
    }
    //if(cudaError_t e = cudaGetLastError(); e != cudaSuccess) 
    //    std::cout<<std::format("ERROR {}\n", (int)e);

    return;
}

__host__ float cudaGof(bMap_t& _map, size_t _threads_x, size_t _threads_y, size_t _iterations) {
    if (_threads_x * _threads_y > 1024) return 0;
    
    bMap_t d_map = {};
    createDeviceMap(_map, d_map);
    float res = iterateGof(d_map, _threads_x, _threads_y, _iterations);

    cpMapFromDevice(_map, d_map);
    freeDeviceMap(d_map);
	return res;
}

__host__ void cudaGof_shared(bMap_t& _map, size_t _threads_x, size_t _threads_y, size_t _iterations) {
    if (_threads_x * _threads_y > 1024) return;
    
    bMap_t d_map = {};
    createDeviceMap(_map, d_map);
    iterateGof_shared(d_map, _threads_x, _threads_y, _iterations);
   
    cpMapFromDevice(_map, d_map);
    freeDeviceMap(d_map);

    return;
}
