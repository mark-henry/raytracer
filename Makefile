NVFLAGS= -arch=compute_20 -code=sm_20

all:
	nvcc $(NVFLAGS) parallelized.cu -o raytrace