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

//unoptimized
__global__ void d_gof(bMap_t _map) {
    //todo:
    //we compute one thread per one cell
    //max block size is 1024 threads -> we can get maximum of 32 side length with 34 necessary cell accesses -> we need 1024 + 4 * 32 + 4 = 1156 booleans (maximum)
	//__shared__ bool cells[1156];  //potentially make this dynamic size in kernel launch parameters
    size_t id_x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t id_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_x >= _map.x || id_y >= _map.y) return;

    int8_t neighbors = 0;
    int8_t cell = mapAt(_map, id_x, id_y);
    
    for(int8_t x = -1; x < 2; x++)
        for (int8_t y = -1; y < 2; y++) {
            if (x == y && y == 0) continue;

            if(id_x + x >= 0 && id_x + x < _map.x && 
                id_y + y >= 0 && id_y + y < _map.y)
                neighbors += mapAt(_map, id_x + x, id_y + y);
        }

    if (neighbors == 3) cell = 1;
    else if (cell && neighbors == 2) cell = 1;
    else cell = 0;

	//printf("thread %lld %lld\n", id_x, id_y);
    mapAt2(_map, id_x, id_y) = cell;
    //__syncthreads();
	return;
}

//one iteration good only since we free map afterwards
__host__ void iterateGof(bMap_t& d_map, size_t _threads, size_t _iterations) {
    dim3 dim = { (uint32_t)_threads, (uint32_t)_threads, 1 };
    dim3 blocks = { (uint32_t)ceil(float(d_map.x) / _threads), (uint32_t)ceil(float(d_map.y) / _threads), 1 };

    for (size_t i = 0; i < _iterations; i++) {
        d_gof<<<dim, blocks>>>(d_map);
        //new state is stored in board2 -> we swap these
        std::swap(d_map.board, d_map.board2);
    }
    
    return;
}

__host__ void cudaGof(bMap_t& _map, size_t _threads, size_t _iterations) {
    if (_threads > 32) return;
    
    bMap_t d_map = {};
    createDeviceMap(_map, d_map);
    iterateGof(d_map, _threads, _iterations);
	//d_gof<<<dim, blocks>>>(d_map);
    //new state is stored in board2 -> we swap these
    std::swap(d_map.board, d_map.board2);

    cpMapFromDevice(_map, d_map);
    freeDeviceMap(d_map);
	return;
}