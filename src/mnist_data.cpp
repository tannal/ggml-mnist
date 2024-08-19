#include "mnist_data.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>

MNISTData::MNISTData(const std::string& images_file, const std::string& labels_file) {
    load_images(images_file);
    load_labels(labels_file);

    if (images.size() != labels.size()) {
        throw std::runtime_error("Number of images and labels do not match");
    }
}

void MNISTData::load_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file");
    }

    images.resize(num_images, std::vector<float>(num_rows * num_cols));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
}

void MNISTData::load_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0, num_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_labels, sizeof(num_labels));

    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file");
    }

    labels.resize(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }
}