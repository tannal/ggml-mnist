#include <gtest/gtest.h>
#include "mnist_model.hpp"
#include "mnist_data.hpp"

TEST(MNISTModelTest, Initialization) {
    ASSERT_NO_THROW({
        MNISTModel model;
    });
}

TEST(MNISTModelTest, InferenceShape) {
    MNISTModel model;
    std::vector<float> dummy_image(784, 0.5f);
    int label = model.infer(dummy_image);
    ASSERT_GE(label, 0);
    ASSERT_LT(label, 10);
}

TEST(MNISTModelTest, Training) {
    MNISTModel model;
    MNISTData data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    ASSERT_NO_THROW({
        model.train(data, 1, 0.01f);
    });
}

TEST(MNISTModelTest, Testing) {
    MNISTModel model;
    MNISTData data("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    float accuracy = model.test(data);
    ASSERT_GE(accuracy, 0.0f);
    ASSERT_LE(accuracy, 1.0f);
}