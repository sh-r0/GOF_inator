#include "gof.hpp"

void randomizeMap(bMap_t& _map) {
    const size_t size = _map.x * _map.y;
    _map.board = (bool*)malloc(_map.x * _map.y);

    for (size_t x = 0; x < _map.x * _map.y; x++)
        _map.board[x] = rand() % 2;

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