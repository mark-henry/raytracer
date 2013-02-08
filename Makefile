CC=nvcc
CFLAGS= -O3 -g -c -DGL_GLEXT_PROTOTYPES
LDFLAGS= -lGL -lGLU -lglut
NVFLAGS= -O3 -c -arch=compute_20 -code=sm_20

OBJS = callbacksPBO.o tracer.o simpleGLmain.o simplePBO.o

realtime: $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@
	
callbacksPBO.o: callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

tracer.o: tracer.cu
	$(CC) $(NVFLAGS) -o $@ $<

simpleGLmain.o: simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

run: realtime
	./realtime