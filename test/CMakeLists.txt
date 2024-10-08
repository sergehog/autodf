cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)

add_executable(compiletime_autodf_test compiletime_autodf_test.cpp)
add_test(NAME compiletime_autodf_test COMMAND compiletime_autodf_test)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        option(CODE_COVERAGE_REPORT "Check code-coverage for test targets" OFF)
        if(CODE_COVERAGE_REPORT)
            set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
            include(CodeCoverage)
            append_coverage_compiler_flags()

            setup_target_for_coverage_gcovr_html(
                NAME autodf_test_coverage
                EXECUTABLE ctest --verbose
                DEPENDENCIES compiletime_autodf_test
                BASE_DIRECTORY "../"
                EXCLUDE "test"
            )
        endif (CODE_COVERAGE_REPORT)
endif ()