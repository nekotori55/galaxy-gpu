cmake_minimum_required(VERSION 3.13)
project(galaxy_gpu)
set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES main.cpp vector2.h)

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} GLUT::GLUT OpenGL::GLU OpenGL)
