C_SRCS = ./fd-source-code.dp.cpp


CFLAGS = -O3 -w -march=native
LIBS = -lm 

all:stencil_code

stencil_code:
	dpcpp -o $@ $(CFLAGS) $(C_SRCS) $(LIBS)

clean:
	rm -f stencil_code
