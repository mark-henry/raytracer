NVFLAGS= -g -arch=compute_20 -code=sm_20

raytrace: parallelized.cu Image.cpp
	nvcc $(NVFLAGS) $^ -o raytrace

raytrace-debug: parallelized.cu Image.cpp
	nvcc $(NVFLAGS) $^ -o raytrace-debug -DDEBUG

debug: raytrace-debug
	./raytrace-debug
	eog awesome.tga &

run: raytrace
	./raytrace
	eog awesome.tga &