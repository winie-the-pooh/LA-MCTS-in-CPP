cmake_minimum_required(VERSION 3.15)
project(LA_MCTS)


set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE "Release")

####################### Mine #########################
set(SRC LA-MCTS.cpp LA-MCTS.h functions.cpp functions.h node.cpp node.h tracker.h tracker.cpp Macros.h
        Classifier.cpp Classifier.h)
add_library(LA-MCTS STATIC ${SRC})
target_link_libraries(LA-MCTS thundersvm gplib eigen)
####################### Mine #########################

####################### OpenMP #########################
add_subdirectory(openmp)
####################### OpenMP #########################

####################### Eigen #########################
add_subdirectory(eigen-3.3.7)
####################### Eigen #########################

####################### thundersvm #########################
include_directories(thundersvm-0.3.4/include)
add_subdirectory(thundersvm-0.3.4)
target_include_directories(thundersvm PRIVATE openmp/runtime/src/include)
target_link_libraries(thundersvm eigen omp)
####################### thundersvm #########################

####################### GPR #########################
include_directories(GPR/include)
add_subdirectory(GPR)
target_link_libraries(gplib eigen omp)
####################### GPR #########################


add_executable(LA_MCTS main.cpp)
target_link_libraries(LA_MCTS LA-MCTS gplib thundersvm eigen)