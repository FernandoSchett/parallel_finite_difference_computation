C_SRCS = ./fd-source-code.cu


CFLAGS = -O3 -w --ftz=false --prec-div=true --fmad=false -arch=sm_60
LIBS = -lm 

all: stencil_code

stencil_code:
	nvcc -o $@ $(CFLAGS) $(C_SRCS) $(LIBS)

clean:
	rm -f stencil_code