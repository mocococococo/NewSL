cmake_minimum_required(VERSION 3.16)
project(culc_module LANGUAGES CXX)

# 推奨: C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- pybind11 を取得（ネット接続が必要）----
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ---- Python拡張モジュールをビルド ----
pybind11_add_module(culc_module src/add.cpp)

# もし Python が複数ある環境なら、明示することも可能
# set(Python_EXECUTABLE "C:/Path/To/python.exe")
# set_target_properties(culc_module PROPERTIES Python3_EXECUTABLE "${Python_EXECUTABLE}")
