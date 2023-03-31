C_SRCS = ./src/fd-code-buffers.dp.cpp


CFLAGS = -I"./lib/include" -O3 -w -arch=sm_60
LDFLAGS = -L"./lib"
LIBS = -lm -lsource

all: rtm_code

rtm_code:
	dpcpp -o $@ $(CFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)

clean:
	rm -f rtm_code
