#include "gof.hpp"
#include <cstddef>
#include <cstring>
#include <iostream>

void cleanMap(bMap_t& _map){
    if(_map.board){
        free(_map.board);
        _map.board = nullptr;
    }
    if(_map.board2){
        free(_map.board2);
        _map.board2 = nullptr;
    }

    return;
}

void initMap(bMap_t& _map, size_t _x, size_t _y) {
    cleanMap(_map);

    _map.x = _x;
    _map.y = _y;

    _map.board = (bool*)malloc(_x * _y);
    _map.board2 = (bool*)malloc(_x * _y);

    return;
}

void randomizeMap(bMap_t& _map) {
    for (size_t i = 0; i < _map.x * _map.y; i++)
        _map.board[i] = rand() % 2;

    return;
}

void every2ndMap(bMap_t& _map) {
    for(size_t i = 0; i < _map.x * _map.y; i++)
        _map.board[i] = i % 2;

    return;
} 

void tPatternMap(bMap_t& _map) {
    for(size_t i = 0; i < _map.x*_map.y;i++) {
        if(i/_map.x < _map.y/3 || (i%_map.x > _map.x/3 && i%_map.x < 2*_map.x/3))
            _map.board[i] = 1;
        else _map.board[i] = 0;
    }

    return;
}

void printMap(const bMap_t& _map) {
    const size_t size = _map.x * _map.y;
    for (int i = 0; i < size; i++) {
        fputc(_map.board[i] ? '-' : ' ', stdout);
        fputc(i % _map.y == (_map.y - 1) ? '\n' : '|', stdout);
    }

    return;
}

void cleanMapRecord(mapRecord_t& _record) {
    _record.x = 0; _record.y = 0;
    for(auto state : _record.boardStates)
        free(state);
    
    _record.boardStates.clear();
    return;
}

void initMapRecord(mapRecord_t& _record, bMap_t& _map) {
    _record.x = _map.x; _record.y = _map.y;

    return;
}

bMap_t cloneMap(const bMap_t& _map) {
    bMap_t res {};
    res.x = _map.x;
    res.y = _map.y;
    
    const size_t size = res.x * res.y;
    res.board = (bool*)malloc(size);
    res.board2 = (bool*)malloc(size);

    memcpy(res.board, _map.board, size);

    return res; 
}

bool compareMap(bMap_t& _map1, bMap_t& _map2) {
    if(_map1.x != _map2.x || _map1.y != _map2.y)
        return false;

    const size_t size = _map1.x * _map1.y;
    for(size_t i = 0; i < size; i++) 
        if(_map1.board[i] != _map2.board[i]) 
            return false;

    return true;
}

void pushMapState(mapRecord_t& _record, bMap_t& _map) {
    #ifdef _DEBUG
    if(_record.x != _map.x || _record.y != _map.y) {
        std::cerr<<"mismatched map and record sizes!\n";
        return;
    }
    #endif

    const size_t mapSize = _record.x * _record.y;
    _record.boardStates.push_back((bool*)(malloc(mapSize)));
    memcpy(_record.boardStates.back(), _map.board, mapSize);

    return;
}
