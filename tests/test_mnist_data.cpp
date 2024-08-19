#include <gtest/gtest.h>
#include "mnist_data.hpp"

TEST(MNISTDataTest, LoadTrainingData) {
    ASSERT_NO_THROW({
        MNISTData train_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        ASSERT_EQ(train_data.size(), 60000);
    });
}

TEST(MNISTDataTest, LoadTestData) {
    ASSERT_NO_THROW({
        MNISTData test_data("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
        ASSERT_EQ(test_data.size(), 10000);
    });
}

TEST(MNISTDataTest, ImageShape) {
    MNISTData data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    ASSERT_EQ(data.get_image(0).size(), 784);
}

TEST(MNISTDataTest, LabelRange) {
    MNISTData data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    for (size_t i = 0; i < data.size(); ++i) {
        int label = data.get_label(i);
        ASSERT_GE(label, 0);
        ASSERT_LT(label, 10);
    }
}