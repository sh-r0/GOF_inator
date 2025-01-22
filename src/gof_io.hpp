#pragma once
#include "gof.hpp"
#include <string>
#include <filesystem>

void loadMap(const std::filesystem::path& _path, bMap_t& _map);
void saveMap(const std::filesystem::path& _path, bMap_t& _map);
void makeVid(const std::filesystem::path& _pattern, const std::filesystem::path& _path);