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
   
    // global coords of first thread in block 
    const int32_t g0x = blockDim.x * blockIdx.x - 1;
    const int32_t g0y = blockDim.y * blockIdx.y - 1;

    //local and global coords
    const int32_t lx = threadIdx.x, ly = threadIdx.y;
    const int32_t gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t gy = threadIdx.y + blockDim.y * blockIdx.y;

    const int32_t rowSize = blockDim.x+2;
    const uint32_t owned = std::ceil((blockDim.x+2.0) * (blockDim.y+2.0) / (blockDim.x * blockDim.y)); 
    const uint32_t startPoint = (lx + ly*blockDim.x) * owned;

    for(uint32_t i = 0; i < owned; i++) {
        uint32_t curr = startPoint + i;
        if(curr >= ((blockDim.x+2) * (blockDim.y+2)))
            break;
       
        int32_t curr_gx = g0x + (curr % rowSize);
        int32_t curr_gy = g0y + (curr / rowSize);        
        if(curr_gx < 0 || curr_gy < 0
           || curr_gx >= _map.x || curr_gy >= _map.y)
            sharedCells[curr] = 0;
        else
            sharedCells[curr] = mapAt(_map, curr_gx, curr_gy);
    }
    
    if(gx >= _map.x || gy >= _map.y)
        return;

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

    float res = 0; //ms
    cudaEvent_t ev1,ev2;
    cudaEventCreate(&ev1,0);
    cudaEventCreate(&ev2,0);

    cudaEventRecord(ev1);
    for (size_t i = 0; i < _iterations; i++) {
        d_gof<<<blocks, dim>>>(d_map);
        //new state is stored in board2 -> we swap these
        std::swap(d_map.board, d_map.board2);
    }
    cudaEventRecord(ev2);
    cudaEventSynchronize(ev2);
    cudaEventElapsedTime(&res, ev1, ev2);

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    return res;
}


__host__ float iterateGof_shared(bMap_t& d_map, size_t _threads_x, size_t _threads_y, size_t _iterations) {
    dim3 dim = { (uint32_t)_threads_x, (uint32_t)_threads_y, 1 };
    dim3 blocks = { (uint32_t)ceil(double(d_map.x) / _threads_x), (uint32_t)ceil(double(d_map.y) / _threads_y), 1 };
    size_t sharedMemSize = (_threads_x+2) * (_threads_y+2);

    float res = 0; //ms
    cudaEvent_t ev1,ev2;
    cudaEventCreate(&ev1,0);
    cudaEventCreate(&ev2,0);

    cudaEventRecord(ev1);
    for (size_t i = 0; i < _iterations; i++) {
        d_gof_shared<<<blocks, dim, sharedMemSize>>>(d_map);
        //new state is stored in board2 -> we swap these
        std::swap(d_map.board, d_map.board2);
    }
    cudaEventRecord(ev2);
    cudaEventSynchronize(ev2);
    cudaEventElapsedTime(&res, ev1, ev2);

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    return res;
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

__host__ float cudaGof_shared(bMap_t& _map, size_t _threads_x, size_t _threads_y, size_t _iterations) {
    if (_threads_x * _threads_y > 1024) return 0;
    
    bMap_t d_map = {};
    createDeviceMap(_map, d_map);
    float res = iterateGof_shared(d_map, _threads_x, _threads_y, _iterations);
   
    cpMapFromDevice(_map, d_map);
    freeDeviceMap(d_map);

    return res;
}
