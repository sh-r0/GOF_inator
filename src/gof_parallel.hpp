#pragma once

#include <iostream>
#include "gof.hpp"

enum scheduleType : uint8_t {
	SCHEDULE_TYPE_DYNAMIC,
	SCHEDULE_TYPE_STATIC,
	SCHEDULE_TYPE_GUIDED,

	SCHEDULE_TYPE_MAX
};

void parGof(bMap_t& _map, size_t _threads, scheduleType _type, size_t _chunkSize);
