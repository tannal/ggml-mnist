# tests/CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(mnist_tests
  test_mnist_model.cpp
  test_mnist_data.cpp
)

add_executable(test_ggml 
  test_ggml.cpp
)
target_link_libraries(test_ggml
  PRIVATE
  gtest_main
  mnist_lib
)


target_link_libraries(mnist_tests
  PRIVATE
  gtest_main
  mnist_lib
)

include(GoogleTest)
gtest_discover_tests(mnist_tests)