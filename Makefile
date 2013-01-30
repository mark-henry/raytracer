NVFLAGS= -arch=compute_20 -code=sm_20

raytrace: parallelized.cu
	nvcc $(NVFLAGS) $^ .cu -o raytrace

run: raytrace
	./raytrace
	eog awesome.tga