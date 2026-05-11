cmake_minimum_required(VERSION 3.16)
project(DecisionTree VERSION 1.0 LANGUAGES C)

# стандарт C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# предупреждения компилятора
add_compile_options(-Wall -Wextra -Wpedantic)

# библиотека с основным кодом — используется и программой, и тестами
add_library(dt_lib STATIC src/decision_tree.c)
target_include_directories(dt_lib PUBLIC src)
target_link_libraries(dt_lib PUBLIC m)

# основной бинарник
add_executable(robot_dt src/main.c)
target_link_libraries(robot_dt PRIVATE dt_lib)

# тесты
enable_testing()
add_subdirectory(tests)

# опция coverage: cmake -S . -B build -DCOVERAGE=ON
option(COVERAGE "Включить gcov покрытие" OFF)
if(COVERAGE)
    target_compile_options(dt_lib PRIVATE --coverage -O0 -g)
    target_link_options(dt_lib PRIVATE --coverage)
endif()
