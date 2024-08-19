#include "mnist_model.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
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
    // Create input tensor
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 784);
    memcpy(x->data, input.data(), input.size() * sizeof(float));

    // First layer
    struct ggml_tensor * layer1 = ggml_add(ctx, ggml_mul_mat(ctx, weights1, x), bias1);
    struct ggml_tensor * layer1_act = ggml_relu(ctx, layer1);

    // Second layer
    struct ggml_tensor * layer2 = ggml_add(ctx, ggml_mul_mat(ctx, weights2, layer1_act), bias2);
    struct ggml_tensor * probs = ggml_soft_max(ctx, layer2);

    // Create computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, probs);

    // Create computation plan
    struct ggml_cplan plan = ggml_graph_plan(gf, GGML_DEFAULT_N_THREADS);

    // Execute computation
    ggml_graph_compute(gf, &plan);

    // Copy results to output vector
    for (int i = 0; i < 10; i++) {
        output[i] = ggml_get_f32_1d(probs, i);
    }
}

void MNISTModel::backward(const std::vector<float>& input, const std::vector<float>& target) {
    // Create a new compute graph for the backward pass
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // Get the input tensor and copy the input data
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 784);
    memcpy(x->data, input.data(), ggml_nbytes(x));

    // Get the weights and biases
    struct ggml_tensor* fc1_weight = ggml_get_tensor(ctx, "fc1_weight");
    struct ggml_tensor* fc1_bias = ggml_get_tensor(ctx, "fc1_bias");
    struct ggml_tensor* fc2_weight = ggml_get_tensor(ctx, "fc2_weight");
    struct ggml_tensor* fc2_bias = ggml_get_tensor(ctx, "fc2_bias");


    // Forward pass
    struct ggml_tensor* fc1 = ggml_mul_mat(ctx, fc1_weight, x);
    fc1 = ggml_add(ctx, fc1, fc1_bias);
    fc1 = ggml_relu(ctx, fc1);

    struct ggml_tensor* fc2 = ggml_mul_mat(ctx, fc2_weight, fc1);
    fc2 = ggml_add(ctx, fc2, fc2_bias);

    struct ggml_tensor* probs = ggml_soft_max(ctx, fc2);

    // Compute loss
    struct ggml_tensor* target_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
    memcpy(target_tensor->data, target.data(), ggml_nbytes(target_tensor));
    
    struct ggml_tensor* loss = ggml_cross_entropy_loss(ctx, probs, target_tensor);

    // Add tensors to the graph
    ggml_build_forward_expand(gf, loss);

    // Compute gradients

    // Update weights and biases
    float learning_rate = 0.01f;


    // Free the graph
}