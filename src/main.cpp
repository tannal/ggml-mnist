#include "mnist_model.hpp"
#include "mnist_data.hpp"
#include "utils.hpp"
#include <iostream>
#include <chrono>

int main() {
    try {
        MNISTData train_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        MNISTData test_data("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

        MNISTModel model;

        // Training
        std::cout << "Training model...\n";
        auto start = std::chrono::high_resolution_clock::now();

        model.train(train_data, 10, 0.01f);  // 10 epochs, learning rate 0.01

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Training completed in " << diff.count() << " seconds\n";

        // Testing
        std::cout << "Testing model...\n";
        float accuracy = model.test(test_data);
        std::cout << "Test accuracy: " << (accuracy * 100.0f) << "%\n";

        // Inference example
        std::vector<float> sample_image = test_data.get_image(0);
        int predicted_label = model.infer(sample_image);
        int true_label = test_data.get_label(0);

        std::cout << "Sample inference:\n";
        std::cout << "Predicted label: " << predicted_label << "\n";
        std::cout << "True label: " << true_label << "\n";

        // Print the image
        print_image(sample_image);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}