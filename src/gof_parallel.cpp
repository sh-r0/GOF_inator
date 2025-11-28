#include <format>
#include "gof_parallel.hpp"
#include <omp.h>

void parGof(bMap_t& _map, size_t _threads, scheduleType _type, size_t _chunkSize) {	
	const int size = _map.x * _map.y;
	const int size_x = _map.x, size_y = _map.y;

    if(_chunkSize == 0)
        _chunkSize = size/_threads;

	switch (_type) {
		case SCHEDULE_TYPE_DYNAMIC: {
#pragma omp parallel for schedule(dynamic, _chunkSize) num_threads(_threads)
			for (int i = 0; i < size; i++) {
				int x = i % size_x, y = i / size_x;
				int8_t neighbors = 0;
				int8_t cell = _map.at(x,y);
                    
				for (int j = -1; j < 2; j++)
					for (int k = -1; k < 2; k++) {
						if ((k == j) && (j == 0)) continue;

						int tx = x + j;
						int ty = y + k;
						if (tx >= 0 && tx < size_x && ty >= 0 && ty < size_y)
							neighbors += _map.at(tx,ty);
					}

				if (neighbors == 3) cell = 1;
				else if (cell && (neighbors == 2)) cell = 1;
				else cell = 0;

				_map.at2(x,y) = cell;
			}
		} break;
		case SCHEDULE_TYPE_STATIC: {
#pragma omp parallel for schedule(static, _chunkSize) num_threads(_threads)
			for (int i = 0; i < size; i++) {
				int x = i % size_x, y = i / size_x;
				int8_t neighbors = 0;
				int8_t cell = _map.at(x, y);

				for (int j = -1; j < 2; j++)
					for (int k = -1; k < 2; k++) {
						if ((k == j) && (j == 0)) continue;

						int tx = x + j;
						int ty = y + k;
						if (tx >= 0 && tx < size_x && ty >= 0 && ty < size_y)
							neighbors += _map.at(tx, ty);
					}

				if (neighbors == 3) cell = 1;
				else if (cell && (neighbors == 2)) cell = 1;
				else cell = 0;

				_map.at2(x, y) = cell;
			}
		} break;
		case SCHEDULE_TYPE_GUIDED: {
#pragma omp parallel for schedule(guided, _chunkSize) num_threads(_threads)
			for (int i = 0; i < size; i++) {
				int x = i % size_x, y = i / size_x;
				int8_t neighbors = 0;
				int8_t cell = _map.at(x, y);

				for (int j = -1; j < 2; j++)
					for (int k = -1; k < 2; k++) {
						if ((k == j) && (j == 0)) continue;

						int tx = x + j;
						int ty = y + k;
						if (tx >= 0 && tx < size_x && ty >= 0 && ty < size_y)
							neighbors += _map.at(tx, ty);
					}

				if (neighbors == 3) cell = 1;
				else if (cell && (neighbors == 2)) cell = 1;
				else cell = 0;

				_map.at2(x, y) = cell;
			}
		} break;
		default:
			break;
	}
	std::swap(_map.board, _map.board2);

	return;
}
