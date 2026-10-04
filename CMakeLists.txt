cmake_minimum_required(VERSION 3.28)
# 本機硬體：RTX 5060 Ti (sm_120) + RTX 3060 Ti (sm_86)。必須在 project() 之前設定，否則 CMake 會填入編譯器預設值。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 86)
endif()
project(miroxen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                      $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo -Xcompiler=/utf-8")

# 所有 src/ 下的程式碼（main.cpp 除外）組成核心靜態庫；新增檔案不需改這裡。
file(GLOB_RECURSE MX_SRC CONFIGURE_DEPENDS src/*.cpp src/*.cu)
list(FILTER MX_SRC EXCLUDE REGEX ".*/src/main\\.cpp$")
add_library(miroxen_core STATIC ${MX_SRC})
target_include_directories(miroxen_core PUBLIC src third_party)
target_link_libraries(miroxen_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
set_target_properties(miroxen_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
if(WIN32)
  target_link_libraries(miroxen_core PUBLIC ws2_32)
endif()

add_executable(miroxen src/main.cpp)
target_link_libraries(miroxen PRIVATE miroxen_core)

# tests/*_test.cpp|cu 各自成為一個可執行檔並註冊到 ctest。
enable_testing()
file(GLOB MX_TESTS CONFIGURE_DEPENDS tests/*_test.cpp tests/*_test.cu)
foreach(t ${MX_TESTS})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_link_libraries(${name} PRIVATE miroxen_core)
  add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
