NVFLAGS= -arch=compute_20 -code=sm_20

raytrace: parallelized.cu Image.cpp
	nvcc $(NVFLAGS) $^ -o raytrace

run: parallelized.cu Image.cpp
	nvcc $(NVFLAGS) $^ -o raytrace
	./raytrace
	eog awesome.tga &

debug: parallelized.cu Image.cpp
	nvcc $(NVFLAGS) $^ -o raytrace -DDEBUG
	./raytrace
	eog awesome.tga &
