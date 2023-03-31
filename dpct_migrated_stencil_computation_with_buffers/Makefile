C_SRCS = ./fd-source-buffer.dp.cpp


CFLAGS = -O3 -w -arch=sm_60
LIBS = -lm 

all: stencil_code

stencil_code:
	dpcpp -o $@ $(CFLAGS) $(C_SRCS) $(LIBS)

clean:
	rm -f stencil_code
