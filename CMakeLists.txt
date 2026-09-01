cmake_minimum_required(VERSION 3.16)

project(Sem_iBoW3D VERSION 1.0.0 LANGUAGES CXX)

option(SEM_IBOW3D_BUILD_APP "Build the Sem-iBoW3D command line application" ON)
option(SEM_IBOW3D_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB QUIET)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

if(TARGET TBB::tbb)
  set(SEM_IBOW3D_TBB_TARGET TBB::tbb)
elseif(TBB_FOUND AND TBB_LIBRARIES)
  set(SEM_IBOW3D_TBB_TARGET ${TBB_LIBRARIES})
else()
  set(TBB_ROOT "" CACHE PATH "Optional TBB root used when find_package(TBB) is unavailable")
  set(TBB_INCLUDE_DIR "" CACHE PATH "Optional TBB include directory")
  set(TBB_LIBRARY_DIR "" CACHE PATH "Optional TBB library directory")
  if(TBB_ROOT AND NOT TBB_INCLUDE_DIR)
    set(TBB_INCLUDE_DIR "${TBB_ROOT}/include" CACHE PATH "Optional TBB include directory" FORCE)
  endif()
  if(TBB_ROOT AND NOT TBB_LIBRARY_DIR)
    set(TBB_LIBRARY_DIR "${TBB_ROOT}/lib" CACHE PATH "Optional TBB library directory" FORCE)
  endif()
  find_path(SEM_IBOW3D_TBB_INCLUDE_DIR tbb/tbb.h
    HINTS ${TBB_INCLUDE_DIR} ${TBB_ROOT}/include
  )
  find_library(SEM_IBOW3D_TBB_LIBRARY NAMES tbb
    HINTS ${TBB_LIBRARY_DIR} ${TBB_ROOT}/lib ${TBB_ROOT}/build/my_tbb_build_release
  )
  if(NOT SEM_IBOW3D_TBB_INCLUDE_DIR OR NOT SEM_IBOW3D_TBB_LIBRARY)
    message(FATAL_ERROR
      "TBB was not found. Install libtbb-dev/oneTBB or pass -DTBB_ROOT=/path/to/tbb.")
  endif()
  add_library(sem_ibow3d_tbb UNKNOWN IMPORTED)
  set_target_properties(sem_ibow3d_tbb PROPERTIES
    IMPORTED_LOCATION "${SEM_IBOW3D_TBB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SEM_IBOW3D_TBB_INCLUDE_DIR}"
  )
  get_filename_component(SEM_IBOW3D_TBB_RPATH "${SEM_IBOW3D_TBB_LIBRARY}" DIRECTORY)
  set(SEM_IBOW3D_TBB_TARGET sem_ibow3d_tbb)
endif()

set(LIBCXX_RUNTIME_DIR "" CACHE PATH "Optional directory containing libc++.so.1 and libc++abi.so.1 for binary Open3D SDKs")

add_library(sem_ibow3d_core
  src/DataBase.cpp
  src/Data_IO.cpp
  src/FeatureContainer.cpp
  src/LabelContainer.cpp
  src/TimeSave.cpp
  src/iBoW3D.cpp
)

target_include_directories(sem_ibow3d_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(sem_ibow3d_core
  PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

if(SEM_IBOW3D_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "SEM_IBOW3D_ENABLE_SANITIZERS requires GCC or Clang")
  endif()
  target_compile_options(sem_ibow3d_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(sem_ibow3d_core PUBLIC -fsanitize=address,undefined)
endif()

target_compile_definitions(sem_ibow3d_core PUBLIC GLEW_NO_GLU)

target_link_libraries(sem_ibow3d_core
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Threads::Threads
    ${SEM_IBOW3D_TBB_TARGET}
)

if(TARGET Open3D::Open3D)
  target_link_libraries(sem_ibow3d_core PUBLIC Open3D::Open3D)
else()
  target_include_directories(sem_ibow3d_core PUBLIC ${Open3D_INCLUDE_DIRS})
  target_link_libraries(sem_ibow3d_core PUBLIC ${Open3D_LIBRARIES})
endif()

if(LIBCXX_RUNTIME_DIR)
  find_library(SEM_IBOW3D_LIBCXX NAMES c++ libc++.so.1
    HINTS "${LIBCXX_RUNTIME_DIR}" NO_DEFAULT_PATH)
  find_library(SEM_IBOW3D_LIBCXXABI NAMES c++abi libc++abi.so.1
    HINTS "${LIBCXX_RUNTIME_DIR}" NO_DEFAULT_PATH)
  if(NOT SEM_IBOW3D_LIBCXX OR NOT SEM_IBOW3D_LIBCXXABI)
    message(FATAL_ERROR
      "LIBCXX_RUNTIME_DIR was set, but libc++.so.1 or libc++abi.so.1 was not found there.")
  endif()
  target_link_libraries(sem_ibow3d_core PUBLIC ${SEM_IBOW3D_LIBCXX} ${SEM_IBOW3D_LIBCXXABI})
  target_link_directories(sem_ibow3d_core PUBLIC "${LIBCXX_RUNTIME_DIR}")
  target_link_options(sem_ibow3d_core PUBLIC "LINKER:-rpath-link,${LIBCXX_RUNTIME_DIR}")
  list(APPEND SEM_IBOW3D_EXTRA_RPATHS "${LIBCXX_RUNTIME_DIR}")
endif()

if(SEM_IBOW3D_TBB_RPATH)
  target_link_directories(sem_ibow3d_core PUBLIC "${SEM_IBOW3D_TBB_RPATH}")
  target_link_options(sem_ibow3d_core PUBLIC "LINKER:-rpath-link,${SEM_IBOW3D_TBB_RPATH}")
  list(APPEND SEM_IBOW3D_EXTRA_RPATHS "${SEM_IBOW3D_TBB_RPATH}")
endif()

if(SEM_IBOW3D_EXTRA_RPATHS)
  set_target_properties(sem_ibow3d_core PROPERTIES
    BUILD_RPATH "${SEM_IBOW3D_EXTRA_RPATHS}"
    INSTALL_RPATH "${SEM_IBOW3D_EXTRA_RPATHS}"
  )
endif()

if(SEM_IBOW3D_BUILD_APP)
  add_executable(sem_ibow3d apps/sem_ibow3d.cpp)
  target_link_libraries(sem_ibow3d PRIVATE sem_ibow3d_core)
  if(SEM_IBOW3D_EXTRA_RPATHS)
    set_target_properties(sem_ibow3d PROPERTIES
      BUILD_RPATH "${SEM_IBOW3D_EXTRA_RPATHS}"
      INSTALL_RPATH "${SEM_IBOW3D_EXTRA_RPATHS}"
    )
  endif()
endif()

if(BUILD_TESTING)
  add_executable(sem_ibow3d_input_contracts tests/test_input_contracts.cpp)
  target_link_libraries(sem_ibow3d_input_contracts PRIVATE sem_ibow3d_core)
  add_test(
    NAME input_contracts
    COMMAND sem_ibow3d_input_contracts "${CMAKE_CURRENT_BINARY_DIR}/input_contracts_fixture"
  )

  if(SEM_IBOW3D_BUILD_APP)
    add_test(NAME cli_help COMMAND sem_ibow3d --help)
    set_tests_properties(cli_help PROPERTIES
      PASS_REGULAR_EXPRESSION "--keypoint-num;--feature-dim;--init-words-num;--words-num-add;--near-num;--gap-num;--ransac-n;--semantic-num"
      FAIL_REGULAR_EXPRESSION "--lambda-label"
    )

    add_test(
      NAME cli_rejects_init_pcd_num_one
      COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/expect_command_failure.py"
              "--init-pcd-num must be at least 2"
              "$<TARGET_FILE:sem_ibow3d>"
              --dataset KITTI --sequence 00 --init-pcd-num 1
    )

    add_test(
      NAME cli_rejects_near_num_zero
      COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/expect_command_failure.py"
              "--near-num must be positive"
              "$<TARGET_FILE:sem_ibow3d>"
              --dataset KITTI --sequence 00 --near-num 0
    )

    add_test(
      NAME cli_rejects_removed_lambda_label
      COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/expect_command_failure.py"
              "Unknown argument: --lambda-label"
              "$<TARGET_FILE:sem_ibow3d>"
              --dataset KITTI --sequence 00 --lambda-label 1.0
    )

    add_test(
      NAME tiny_registration_ransac
      COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/run_tiny_registration_smoke.py"
              "$<TARGET_FILE:sem_ibow3d>"
              "${CMAKE_CURRENT_BINARY_DIR}/tiny_registration_ransac"
              ransac
    )
    if(SEM_IBOW3D_ENABLE_SANITIZERS)
      set_tests_properties(tiny_registration_ransac PROPERTIES DISABLED TRUE)
    endif()

    add_test(
      NAME tiny_registration_fgr
      COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/run_tiny_registration_smoke.py"
              "$<TARGET_FILE:sem_ibow3d>"
              "${CMAKE_CURRENT_BINARY_DIR}/tiny_registration_fgr"
              fgr
    )
    if(SEM_IBOW3D_ENABLE_SANITIZERS)
      set_tests_properties(tiny_registration_fgr PROPERTIES DISABLED TRUE)
    endif()
  endif()
endif()
