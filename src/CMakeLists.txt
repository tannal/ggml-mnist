# src/CMakeLists.txt
add_library(mnist_lib
    mnist_model.cpp
    mnist_data.cpp
)

target_include_directories(mnist_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mnist_lib PUBLIC ggml)

add_executable(mnist_train_infer main.cpp)
target_link_libraries(mnist_train_infer PRIVATE mnist_lib)