cmake_minimum_required(VERSION 3.20)
project(AyumuCurlingSimulator LANGUAGES CXX)

# --- C++20 必須（<bit> の popcount / countr_zero 等のため）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 主要ソース（必要に応じて追加/調整）
set(SRC_FAST
    fast_simulator.cpp
    # 他に .cpp があればここに列挙
)

# --- ライブラリ化（コアを共有）
add_library(fast_sim STATIC ${SRC_FAST})
target_include_directories(fast_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (MSVC)
  target_compile_options(fast_sim PRIVATE /W4 /permissive- /utf-8 /bigobj)
else()
  target_compile_options(fast_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- サンプル実行ファイル（任意）
add_executable(myapp sample.cpp)
target_link_libraries(myapp PRIVATE fast_sim)

# ===================== pybind11 (Python拡張) =====================
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Python拡張モジュール (.pyd)
pybind11_add_module(py_fast_simulator py_fast_simulator.cpp)
target_link_libraries(py_fast_simulator PRIVATE fast_sim)

# 追加の定義/ライブラリが必要ならここに追記
# target_compile_definitions(fast_sim PRIVATE MYDEF=1)
# target_link_libraries(fast_sim PRIVATE ws2_32)
