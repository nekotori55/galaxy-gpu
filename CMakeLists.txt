cmake_minimum_required(VERSION 3.13)
project(galaxy_gpu)
set(CMAKE_CXX_STANDARD 20)
set(CUDA_ARCHITECTURES native)

enable_language(CUDA)

set(SOURCE_FILES src/main.cu src/vector2.h )

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED) 


add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} GLUT::GLUT OpenGL::GLU OpenGL)
