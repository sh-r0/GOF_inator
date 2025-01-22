#include <iostream>
#include <format>
#include <omp.h>
#include <chrono>

#include "gof_parallel.hpp"
#include "gof_cuda.hpp"
#include "gof_io.hpp"

int32_t main() {
    bMap_t map = {};
    map.init(32,32);
    randomizeMap(map);
    //loadMap("map1.png", map);
    //saveMap("map2.png", map);

    //printMap(map);
    //saveMap("./blob.gof", map);
    std::chrono::time_point<std::chrono::steady_clock> tp1, tp2;
    
    for (size_t i = 0; i < 10; i++) {
        parGof(map, 25, SCHEDULE_TYPE_DYNAMIC);
        saveMap(std::format("map_frame{}.png", i), map);
    }
    makeVid("map_frame%d.png", "vid.mp4");

    /*
    for (size_t x = 0; x < 10; x++) {
        //tp1 = std::chrono::steady_clock::now();

        //cudaGof(map, 20, 10);
        for (size_t i = 0; i < 10; i++) {
            parGof(map, 25, SCHEDULE_TYPE_DYNAMIC);
            //printMap(map);
            //std::cout << "\n\n\n";
        }
        //tp2 = std::chrono::steady_clock::now();
        //std::cout << std::format("done in {}micro s/{}ms\n", std::chrono::duration_cast<std::chrono::microseconds>(tp2-tp1).count(), std::chrono::duration_cast<std::chrono::milliseconds>(tp2-tp1).count());
    }
    */
    //printMap(map);

    return 0;
}