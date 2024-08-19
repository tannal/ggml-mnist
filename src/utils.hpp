#pragma once

#include <vector>
#include <iostream>

inline void print_image(const std::vector<float>& image) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            float pixel = image[i * 28 + j];
            if (pixel < 0.2) std::cout << " ";
            else if (pixel < 0.4) std::cout << ".";
            else if (pixel < 0.6) std::cout << "o";
            else if (pixel < 0.8) std::cout << "O";
            else std::cout << "@";
        }
        std::cout << "\n";
    }
}