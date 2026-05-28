cmake_minimum_required(VERSION 3.18)

set(PAROSOL_SOURCES
    src/GReader.cpp
    src/GWriter.cpp
    src/JacobiSmoother.cpp
    src/Toolbox.cpp
    src/AsciiImage.cpp
    src/HDF5Image.cpp
    src/AsciiImageMirrored.cpp
    src/fem.cpp
    src/est_ev.cpp
    src/Chebyshev.cpp
    src/PCGSolver.cpp
    src/main.cpp)

add_executable(parosol ${PAROSOL_SOURCES})
target_compile_features(parosol PRIVATE cxx_std_11)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS CXX)

target_include_directories(parosol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${HDF5_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(parosol PRIVATE
    ${HDF5_LIBRARIES}
    MPI::MPI_CXX)
