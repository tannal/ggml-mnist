#include "mnist_model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

MNISTModel::MNISTModel() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };

    ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("Failed to initialize GGML context");
    }

    weights1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 784, 128);
    bias1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    weights2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 10);
    bias2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);

    // Initialize weights and biases with random values
    ggml_set_f32(weights1, 0.1f);
    ggml_set_f32(bias1, 0.1f);
    ggml_set_f32(weights2, 0.1f);
    ggml_set_f32(bias2, 0.1f);
}

MNISTModel::~MNISTModel() {
    ggml_free(ctx);
}

void MNISTModel::train(const MNISTData& data, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < data.size(); ++i) {
            const std::vector<float>& image = data.get_image(i);
            std::vector<float> target(10, 0.0f);
            target[data.get_label(i)] = 1.0f;

            std::vector<float> output(10);
            forward(image, output);
            backward(image, target);

            // Update weights and biases
            // (This is a simplified version, you might want to implement proper optimization algorithms)
            for (int j = 0; j < 784 * 128; ++j) {
                float* w = (float*)weights1->data + j;
                *w -= learning_rate * ggml_get_f32_1d(weights1, j);
            }
            for (int j = 0; j < 128; ++j) {
                float* b = (float*)bias1->data + j;
                *b -= learning_rate * ggml_get_f32_1d(bias1, j);
            }
            for (int j = 0; j < 128 * 10; ++j) {
                float* w = (float*)weights2->data + j;
                *w -= learning_rate * ggml_get_f32_1d(weights2, j);
            }
            for (int j = 0; j < 10; ++j) {
                float* b = (float*)bias2->data + j;
                *b -= learning_rate * ggml_get_f32_1d(bias2, j);
            }
        }
    }
}

float MNISTModel::test(const MNISTData& data) {
    int correct = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        const std::vector<float>& image = data.get_image(i);
        int true_label = data.get_label(i);
        int predicted_label = infer(image);
        if (predicted_label == true_label) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / data.size();
}

int MNISTModel::infer(const std::vector<float>& image) {
    std::vector<float> output(10);
    forward(image, output);
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

void MNISTModel::forward(const std::vector<float>& input, std::vector<float>& output) {

}

void MNISTModel::backward(const std::vector<float>& input, const std::vector<float>& target) {
    // Implement backpropagation here
    // This is a placeholder and should be implemented properly
}