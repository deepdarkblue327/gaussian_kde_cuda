CXX=nvcc
CXXFLAGS=-std=c++11 -O3 -x cu -Wno-deprecated-gpu-targets -arch sm_20

all: a3

clean:
	rm -rf a3
