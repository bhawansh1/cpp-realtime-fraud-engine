cmake_minimum_required(VERSION 3.20)
project(VeloceML VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for maximum performance & warnings
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -funroll-loops)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Core VeloceML Library
add_library(veloce_ml STATIC
    src/math/Matrix.cpp
    src/data/Dataset.cpp
    src/models/LogisticRegression.cpp
    src/models/DecisionTree.cpp
    src/metrics/Metrics.cpp
    src/server/HttpServer.cpp
)

target_include_directories(veloce_ml PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 1. CLI Training & Evaluation Pipeline
add_executable(veloce_train src/cli/main_train.cpp)
target_link_libraries(veloce_train PRIVATE veloce_ml)

# 2. Production Microservice (REST & UI Server)
add_executable(veloce_server src/server/main_server.cpp)
target_link_libraries(veloce_server PRIVATE veloce_ml)

# 3. Unit & Integration Test Suite
add_executable(veloce_tests
    tests/test_main.cpp
    tests/test_matrix.cpp
    tests/test_logistic.cpp
    tests/test_tree.cpp
)
target_link_libraries(veloce_tests PRIVATE veloce_ml)

# 4. Latency & Throughput Benchmark Suite
add_executable(veloce_benchmark benchmarks/benchmark_cpp.cpp)
target_link_libraries(veloce_benchmark PRIVATE veloce_ml)
