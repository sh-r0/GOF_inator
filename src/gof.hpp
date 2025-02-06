#pragma once
#include <array>
#include <cstdint>
#include <vector>
//#include <cuda.h>
//#include <bitset>

struct bMap_t {
	size_t x, y;
	bool* board, *board2;

	inline bool& at(size_t _x, size_t _y) {
		return board[_x + _y * x];
	}

	inline bool& at2(size_t _x, size_t _y) {
		return board2[_x + _y * x];
	}
};

struct mapRecord_t {
    size_t x, y;
    std::vector<bool*> boardStates;

	inline bool& at(size_t _index, size_t _x, size_t _y) {
		return boardStates[_index][_x + _y * y];
	}
};

void cleanMap(bMap_t& _map);
void initMap(bMap_t& _map, size_t _x, size_t _y);
bMap_t cloneMap(const bMap_t& _map);
bool compareMap(bMap_t& _map1, bMap_t& _map2);
void randomizeMap(bMap_t& _map);
void every2ndMap(bMap_t& _map);
void tPatternMap(bMap_t& _map);
void printMap(const bMap_t& _map);

void cleanMapRecord(mapRecord_t& _record);
void initMapRecord(mapRecord_t& _record, bMap_t& _map);
void pushMapState(mapRecord_t& _record, bMap_t& _map);
