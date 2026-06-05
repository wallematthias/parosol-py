cmake_minimum_required(VERSION 3.18)
project(parosol_py LANGUAGES CXX)

add_subdirectory(src/parosol_native)

install(TARGETS parosol
        RUNTIME DESTINATION parosol_py/bin)

foreach(PAROSOL_MPI_RUNTIME openmpi msmpi)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/parosol_py/bin/${PAROSOL_MPI_RUNTIME}")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/parosol_py/bin/${PAROSOL_MPI_RUNTIME}"
                DESTINATION parosol_py/bin)
        file(GLOB PAROSOL_MPI_RUNTIME_PROGRAMS
             "${CMAKE_CURRENT_SOURCE_DIR}/src/parosol_py/bin/${PAROSOL_MPI_RUNTIME}/bin/*"
             "${CMAKE_CURRENT_SOURCE_DIR}/src/parosol_py/bin/${PAROSOL_MPI_RUNTIME}/*.exe")
        if(PAROSOL_MPI_RUNTIME_PROGRAMS)
            if(PAROSOL_MPI_RUNTIME STREQUAL "openmpi")
                install(PROGRAMS ${PAROSOL_MPI_RUNTIME_PROGRAMS}
                        DESTINATION parosol_py/bin/${PAROSOL_MPI_RUNTIME}/bin)
            else()
                install(PROGRAMS ${PAROSOL_MPI_RUNTIME_PROGRAMS}
                        DESTINATION parosol_py/bin/${PAROSOL_MPI_RUNTIME})
            endif()
        endif()
    endif()
endforeach()
