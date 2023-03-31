C_SRCS = ./src/fd-code.cu


CFLAGS = -I"./lib/include" -O3 -w --ftz=false --prec-div=true --fmad=false -arch=sm_60
LDFLAGS = -L"./lib"
LIBS = -lm -lsource

all: rtm_code

rtm_code:
	nvcc -o $@ $(CFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)

clean:
	rm -f rtm_code