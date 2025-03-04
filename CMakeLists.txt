cmake_minimum_required(VERSION 3.10)
project(MazeWalker)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project directory setup
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(SFML_DIR D:/downloads/SFML-2.5.1/lib/cmake/SFML)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Include directories
include_directories(${PROJECT_ROOT}/include)

# Find all .cpp files in the src directory
file(GLOB SOURCES ${PROJECT_ROOT}/src/*.cpp)

# Add executable
add_executable(MazeWalker ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} sfml-graphics sfml-window sfml-system)

# copy all DLL files and python files needed
file(GLOB DEP_FILES "${PROJECT_ROOT}/dll_files/*.dll" "${PROJECT_ROOT}/image_to_maze/*.py" "${PROJECT_ROOT}/fonts/*.ttf")
foreach(DEP_FILE ${DEP_FILES})
    file(COPY ${DEP_FILE} DESTINATION ${PROJECT_ROOT}/build/Debug)
    message(STATUS "Copied DLL file: ${DEP_FILE}")
endforeach()