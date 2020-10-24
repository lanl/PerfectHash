cmake_minimum_required (VERSION 3.1)
project (PerfectHash)

if (DEVICE_DETECT_DEBUG)
   add_definitions(-DDEVICE_DETECT_DEBUG=1)
endif (DEVICE_DETECT_DEBUG)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif() 

find_package(OpenCL)
if (OpenCL_FOUND)
   add_definitions(-DHAVE_OPENCL)
   set(HAVE_CL_DOUBLE ON CACHE BOOL "Have OpenCL Double")
   set(NO_CL_DOUBLE OFF)
   include_directories(${OpenCL_INCLUDE_DIRS})
   #message("OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS}")
   #message("OpenCL_LIBRARIES ${OpenCL_LIBRARIES}")
endif (OpenCL_FOUND)

add_subdirectory(kdtree)

# Adds build target of sort with source code files
add_executable(sort sort.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(sort ${OpenCL_LIBRARIES} m)

# Adds build target of sort2d with source code files
add_executable(sort2d sort2d.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(sort2d ${OpenCL_LIBRARIES} m)

# Adds build target of remap with source code files
add_executable(remap remap.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(remap ${OpenCL_LIBRARIES} kdtree m)

# Adds build target of remap2d with source code files
add_executable(remap2d remap2d.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(remap2d ${OpenCL_LIBRARIES} kdtree m)

# Adds build target of neigh with source code files
add_executable(neigh neigh.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(neigh ${OpenCL_LIBRARIES} kdtree m)

# Adds build target of neigh2d with source code files
add_executable(neigh2d neigh2d.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(neigh2d ${OpenCL_LIBRARIES} kdtree m)

# Adds build target of table with source code files
add_executable(table table.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(table ${OpenCL_LIBRARIES} m)

# Adds build target of tablelarge with source code files
add_executable(tablelarge tablelarge.c gpu.c timer.c gpu.h timer.h)
target_link_libraries(tablelarge ${OpenCL_LIBRARIES} m)

# Cleanup
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
       	"CMakeCache.txt;Makefile;cmake_install.cmake;ipo_out.optrpt")
