NVFLAGS= -g -arch=compute_20 -code=sm_20

raytrace: parallelized.cu Image.cpp
	nvcc $(NVFLAGS) $^ -o raytrace

run: raytrace
	./raytrace
	eog awesome.tga &