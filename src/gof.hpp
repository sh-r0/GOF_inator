#pragma once
#include <array>
#include <cstdint>
#include <cuda.h>
//#include <bitset>

struct bMap_t {
	size_t x, y;
	bool* board, *board2;

	inline bool& at(size_t _x, size_t _y) {
		return board[_x + _y * y];
	}

	inline bool& at2(size_t _x, size_t _y) {
		return board2[_x + _y * y];
	}

	void init(size_t _x, size_t _y) {
		x = _x; y = _y;
		board = (bool*)malloc(x * y);
		board2 = (bool*)malloc(x * y);

		return;
	}
};

void randomizeMap(bMap_t& _map);
void printMap(const bMap_t& _map);