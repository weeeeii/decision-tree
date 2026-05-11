cmake_minimum_required(VERSION 3.16)

add_executable(test_runner test_all.c)
target_link_libraries(test_runner PRIVATE dt_lib)

add_test(NAME AllTests COMMAND test_runner)
set_tests_properties(AllTests PROPERTIES
    PASS_REGULAR_EXPRESSION "0 failed"
)
