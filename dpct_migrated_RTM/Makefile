C_SRCS = ./src/fd-code.dp.cpp


CFLAGS = -I"./lib/include" -O3 -w -march=native
LDFLAGS = -L"./lib"
LIBS = -lm -lsource

all: rtm_code

rtm_code:
	dpcpp -o $@ $(CFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)

clean:
	rm -f rtm_code
