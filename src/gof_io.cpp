#include "gof_io.hpp"
#include "gof.hpp"

#include <cstddef>
#include <iostream>
#include <format>
#include <cstdio>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

/*
	GOF native format is specified as such:
	first 8 bytes - map.x
	second 8 bytes - map.y
	rest - bytes representing each cell of the map
*/

constexpr const char* nativeExtension_c = ".gof";

void loadNative(const std::filesystem::path& _path, bMap_t& _map) {
	FILE* inputFile = fopen(_path.string().c_str(), "rb");
	fseek(inputFile, 0, SEEK_END);
	size_t fSize = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);

	fread(&_map.x, sizeof(_map.x), 1, inputFile);
	fread(&_map.y, sizeof(_map.y), 1, inputFile);

	size_t mapSize = _map.x * _map.y;
	if (fSize != (mapSize + 2 * sizeof(_map.x))) {
		std::cout << std::format("unexpected filesize!\nexpected {} found {}\n", _map.x * _map.y + 2 * sizeof(_map.x), fSize);
		_map.x = 0; _map.y = 0;
		_map.board = nullptr;
		return;
	}

	_map.board = (bool*)malloc(mapSize);
	_map.board2 = (bool*)malloc(mapSize);

	(void)fread(_map.board, 1, mapSize, inputFile);

	return;
}

void loadImage(const std::filesystem::path& _path, bMap_t& _map) {
	int x = 0, y = 0, ch = 0;
	uint8_t* data = stbi_load(_path.string().c_str(), &x, &y, &ch, 1);
	if (!data) {
		_map.x = 0; _map.y = 0;
		_map.board = nullptr;
		return;
	}

	size_t mapSize = x * y;
	_map.x = x; _map.y = y;
	_map.board = (bool*)malloc(mapSize);
	_map.board2 = (bool*)malloc(mapSize);

	for (size_t i = 0; i < mapSize; i++)
		if (data[i] == 0) data[i] = 1;
		else data[i] = 0;

	memcpy(_map.board, data, mapSize);

	free(data);
	return;
}

void loadMap(const std::filesystem::path& _path, bMap_t& _map) {
    cleanMap(_map);

    if (_path.extension() == nativeExtension_c)
        loadNative(_path, _map);
    else 
        loadImage(_path, _map);

	return;
}

void saveImage(const std::filesystem::path& _path, bMap_t& _map) {
	size_t mapSize = _map.x * _map.y;
	uint8_t* data = (uint8_t*)malloc(mapSize);
	for (size_t i = 0; i < mapSize; i++) 
		data[i] = _map.board[i] ? 0 : 255;
	
	stbi_write_png(_path.string().c_str(), _map.x, _map.y, 1, data, _map.x);

	free(data);
	return;
}

void saveNative(const std::filesystem::path& _path, bMap_t& _map) {
	FILE* outputFile = fopen(_path.string().c_str(), "wb");

	fwrite(&_map.x, sizeof(_map.x), 1, outputFile);
	fwrite(&_map.y, sizeof(_map.y), 1, outputFile);

	size_t mapSize = _map.x * _map.y;
	(void)fwrite(_map.board, 1, mapSize, outputFile);

	return;
}

void saveMap(const std::filesystem::path& _path, bMap_t& _map) {
	if (_path.extension().string() == nativeExtension_c)
		saveNative(_path, _map);
	else
		saveImage(_path, _map);

	return;
}

void makeVid(const std::filesystem::path& _pattern, const std::filesystem::path& _path) {
	const std::string ffmpegCommand = std::format("ffmpeg -framerate 1 -i {} -c:v libx264 -r 30 -pix_fmt yuv420p {}", _pattern.string(), _path.string());
	(void)system(ffmpegCommand.c_str());

	return;
}

void saveRecord(const std::filesystem::path& _path, mapRecord_t& _record) {
    bMap_t tmp = {};
    tmp.x = _record.x; tmp.y = _record.y;

    system("mkdir tmp");
    for(size_t i = 0; i < _record.boardStates.size(); i++) {
        tmp.board = _record.boardStates[i]; 
        saveImage(std::format("tmp/frame_{}.png", i), tmp);
    }

    makeVid("tmp/frame_%d.png", _path);
    
    //remove tmp/*
    system("rm -rf tmp");

    return;
}
