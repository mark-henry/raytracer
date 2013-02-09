CC=nvcc
CFLAGS= -c -DGL_GLEXT_PROTOTYPES -O3
NVFLAGS= -c -O3 -arch=compute_20 -code=sm_20 -Xptxas -dlcm=ca
LDFLAGS= -lGL -lGLU -lglut

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