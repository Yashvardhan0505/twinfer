# twinfer build
#   make cpu                 CPU reference engine (any machine)
#   make cuda                CUDA engine, kernel tests and benchmarks (needs nvcc)
#   make cuda ARCH=sm_75     pick a GPU arch (default: native = the GPU in this machine)
#   make test                CPU-side tests (tokenizer + CPU engine vs numpy reference)
#   make test-cuda           CUDA kernel + end-to-end tests on a real GPU
#   make test-emu            CUDA kernels run under the CPU emulator (no GPU needed)

CXX      ?= g++
NVCC     ?= nvcc
ARCH     ?= native
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -fopenmp
NVFLAGS  ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall
BUILD    := build

COMMON_SRC := common/model_files.cpp common/tokenizer.cpp common/platform.cpp
COMMON_HDR := $(wildcard common/*.h)
CUDA_HDR   := $(wildcard cuda/*.h cuda/*.cuh cuda/kernels/*.cuh)

.PHONY: all cpu cuda test test-cuda test-emu bench clean

all: cpu

cpu: $(BUILD)/twinfer-cpu

$(BUILD)/twinfer-cpu: cpu/main.cpp cpu/cpu_model.cpp cpu/cpu_model.h $(COMMON_SRC) $(COMMON_HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ cpu/main.cpp cpu/cpu_model.cpp $(COMMON_SRC)

$(BUILD)/test-common: tests/test_common.cpp cpu/cpu_model.cpp cpu/cpu_model.h $(COMMON_SRC) $(COMMON_HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_common.cpp cpu/cpu_model.cpp $(COMMON_SRC)

cuda: $(BUILD)/twinfer-cuda $(BUILD)/test-kernels $(BUILD)/bench-kernels

$(BUILD)/twinfer-cuda: cuda/main.cu cuda/cuda_model.cu $(CUDA_HDR) $(COMMON_SRC) $(COMMON_HDR)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ cuda/main.cu cuda/cuda_model.cu $(COMMON_SRC)

$(BUILD)/test-kernels: cuda/test_kernels.cu $(CUDA_HDR) $(COMMON_HDR)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ cuda/test_kernels.cu

$(BUILD)/bench-kernels: cuda/bench_kernels.cu $(CUDA_HDR) $(COMMON_HDR)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ cuda/bench_kernels.cu -lcublas

test: cpu $(BUILD)/test-common
	python3 tests/run_tests.py --cpu

test-cuda: cpu cuda
	./$(BUILD)/test-kernels
	python3 tests/run_tests.py --cpu --cuda

test-emu: cpu
	python3 tests/cuda_emu/run_emulated.py

bench: cuda
	./$(BUILD)/bench-kernels

clean:
	rm -rf $(BUILD)
