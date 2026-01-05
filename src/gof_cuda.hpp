#pragma once 

#include <cuda.h>
#include <device_launch_parameters.h>
#include "gof.hpp"
#include <cuda_runtime.h>

__host__ float cudaGof_shared(bMap_t& _map, size_t _threads_x, size_t _threads_y, size_t _iterations);
__host__ float cudaGof(bMap_t& _map, size_t _threads_x, size_t _threads_y, size_t _iterations);
