#pragma once

#include "mnist_data.hpp"
#include <ggml.h>
#include <vector>
#include <memory>

class MNISTModel {
public:
    MNISTModel();
    ~MNISTModel();

    void train(const MNISTData& data, int epochs, float learning_rate);
    float test(const MNISTData& data);
    int infer(const std::vector<float>& image);

private:
    struct ggml_context* ctx;
    struct ggml_tensor* weights1;
    struct ggml_tensor* bias1;
    struct ggml_tensor* weights2;
    struct ggml_tensor* bias2;

    void forward(const std::vector<float>& input, std::vector<float>& output);
    void backward(const std::vector<float>& input, const std::vector<float>& target);
};