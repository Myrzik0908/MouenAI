cmake_minimum_required(VERSION 3.20)
project(Mouen_Engine VERSION 1.0 LANGUAGES C)

option(ENABLE_TRAINING "Build with support for MOUEN_TRAINING mode" OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimization for Clang / LLVM
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    if(WIN32)
        add_compile_options(/utf-8 /O2 /fp:fast -Wno-unused-command-line-argument -Wno-pass-failed -Wno-error)
    else()
        add_compile_options(-O3 -march=native -ffast-math -Wno-unused-command-line-argument -Wno-pass-failed -Wno-error)
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# --- DEFINING LIBRARY SOURCE FILES ---
set(LIB_SOURCES
    MouenAI/dataset.c
    MouenAI/dataset.h
    MouenAI/tokenizer.c
    MouenAI/tokenizer.h
    MouenAI/engine.c
    MouenAI/engine.h
)

# Building a library
add_library(MouenAI SHARED ${LIB_SOURCES})

set_target_properties(MouenAI PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
set_target_properties(MouenAI PROPERTIES PREFIX "")

# Calculate the absolute path to the folder containing your source files
get_filename_component(REAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/MouenAI" ABSOLUTE)
message(STATUS "!!! ABSOLUTE PATH TO THE SOURCE FILES: ${REAL_SOURCE_DIR}")

# Configuring include paths for the library
target_include_directories(MouenAI PUBLIC "${REAL_SOURCE_DIR}")
target_include_directories(MouenAI PRIVATE "${REAL_SOURCE_DIR}")
target_include_directories(MouenAI PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(MouenAI PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# --- MANUAL OPENMP CONFIGURATION FOR WINDOWS + CLANG-CL ---
if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(MouenAI PRIVATE /openmp)
    get_filename_component(LLVM_BIN_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
    find_library(LIBOMP_LIB NAMES libomp libomp.lib PATHS "${LLVM_BIN_DIR}/../lib" NO_DEFAULT_PATH)

    if(LIBOMP_LIB)
        target_link_libraries(MouenAI PRIVATE "${LIBOMP_LIB}")
    else()
        target_link_libraries(MouenAI PRIVATE libomp)
    endif()

    set(DYNAMIC_OMP_PATH "${LLVM_BIN_DIR}/libomp.dll")
    if(EXISTS "${DYNAMIC_OMP_PATH}")
        add_custom_command(TARGET MouenAI POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DYNAMIC_OMP_PATH}"
            "$<TARGET_FILE_DIR:MouenAI>/"
        )
    endif()
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(MouenAI PRIVATE OpenMP::OpenMP_C)
endif()

target_compile_definitions(MouenAI PRIVATE $<$<CONFIG:Debug>:DEBUG>)

if(ENABLE_TRAINING)
    target_compile_definitions(MouenAI PRIVATE MOUEN_TRAINING=1 TRAINING=1)
endif()