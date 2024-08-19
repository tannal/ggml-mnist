# src/CMakeLists.txt
add_library(mnist_lib
    mnist_model.cpp
    mnist_data.cpp
)

target_include_directories(mnist_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mnist_lib PUBLIC ggml)

add_executable(mnist_train_infer main.cpp)
target_link_libraries(mnist_train_infer PRIVATE mnist_lib)


# 创建mnist-common库
add_library(mnist-common mnist-common.cpp)
target_link_libraries(mnist-common PUBLIC ggml)

# 创建mnist-eval可执行文件
add_executable(mnist-eval mnist-eval.cpp)
target_link_libraries(mnist-eval PRIVATE ggml mnist-common)

# 创建mnist-train可执行文件
add_executable(mnist-train mnist-train.cpp)
target_link_libraries(mnist-train PRIVATE ggml mnist-common)