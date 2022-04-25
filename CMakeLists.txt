cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(neural_network LANGUAGES CXX)
enable_testing()

find_package(Boost 1.56 REQUIRED COMPONENTS program_options)

include(Sanitizers.cmake)

include_directories(include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

OPTION(ENABLE_TSAN "enable Tsan" OFF)
IF(ENABLE_TSAN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
ENDIF()

add_executable(test_network "./source/test_network.cpp" "./source/network.cpp" "./source/auxiliary_functions.cpp")
add_executable(train_network "./source/train_network.cpp" "./source/network.cpp" "./source/auxiliary_functions.cpp")

set_property(TARGET test_network PROPERTY CXX_STANDARD 20)
set_property(TARGET train_network PROPERTY CXX_STANDARD 20)

target_link_libraries(test_network PRIVATE Boost::program_options Threads::Threads)
target_link_libraries(train_network PRIVATE Boost::program_options Threads::Threads)

install(TARGETS test_network train_network DESTINATION bin)
install(DIRECTORY demo_data DESTINATION bin)
install(PROGRAMS demo DESTINATION bin)
