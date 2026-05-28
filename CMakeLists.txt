cmake_minimum_required(VERSION 3.18)
project(parosol_py LANGUAGES CXX)

add_subdirectory(src/parosol_native)

install(TARGETS parosol
        RUNTIME DESTINATION parosol_py/bin)
