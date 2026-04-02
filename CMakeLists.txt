cmake_minimum_required(VERSION 3.18)
project(LlucsBrain LANGUAGES CXX)

# Desenvolvido por Llucs

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suporte a OpenMP para paralelismo em CPU
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/core_engine.cpp
    src/stdp_learning.cpp
)

add_executable(LlucsBrain ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(LlucsBrain PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(LlucsBrain PRIVATE stdc++fs)
