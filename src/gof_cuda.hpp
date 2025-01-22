#include <cuda.h>
#include <device_launch_parameters.h>
#include "gof.hpp"
#include <cuda_runtime.h>

__host__ void cudaGof(bMap_t& _map, size_t _threads, size_t _iterations);