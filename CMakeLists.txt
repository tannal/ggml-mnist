# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(ggml_mnist_project VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Find threads package
find_package(Threads REQUIRED)

# Add GGML as a subdirectory
add_subdirectory(external/ggml)

# Add src directory
add_subdirectory(src)

# Add tests directory
add_subdirectory(tests)