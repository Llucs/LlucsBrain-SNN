cmake_minimum_required(VERSION 3.18)
project(LlucsBrain LANGUAGES CXX)

# Desenvolvido por Llucs

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tentar encontrar CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --expt-relaxed-constexpr")
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA encontrado. Compilando com suporte a GPU.")
else()
    message(STATUS "CUDA não encontrado. Compilando apenas para CPU.")
endif()

# Suporte a OpenMP para paralelismo em CPU
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_definitions(-DUSE_OPENMP)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/core_engine.cu
    src/stdp_learning.cu
)

add_executable(LlucsBrain ${SOURCES})

if(CMAKE_CUDA_COMPILER)
    set_target_properties(LlucsBrain PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
    )
    target_link_libraries(LlucsBrain PRIVATE curand)
else()
    # Se não houver CUDA, tratar arquivos .cu como .cpp
    set_source_files_properties(src/core_engine.cu src/stdp_learning.cu PROPERTIES LANGUAGE CXX)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(LlucsBrain PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(LlucsBrain PRIVATE stdc++fs)
