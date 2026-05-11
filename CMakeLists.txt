cmake_minimum_required(VERSION 3.16)
project(DecisionTree VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(dt_lib STATIC src/decision_tree.c)
target_include_directories(dt_lib PUBLIC src)
target_link_libraries(dt_lib PUBLIC m)

add_executable(robot_dt src/main.c)
target_link_libraries(robot_dt PRIVATE dt_lib)

enable_testing()
add_subdirectory(tests)
