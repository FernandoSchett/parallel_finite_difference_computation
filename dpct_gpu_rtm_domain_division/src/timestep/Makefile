CWPROOT = ../../lib/cwp

# source files.
SRC = fd.c

OBJ = $(SRC:.c=.o)

OUT = ../../lib/libtimestep.a

# include directories
INCLUDES = -I../../include/timestep -I../../include/util -I$(CWPROOT)/include

# C compiler flags
CFLAGS = -O3 -Wall -Wextra -pedantic -fpermissive -w -fpermissive

# compiler
CC = g++

.SUFFIXES: .c

default: $(OUT)

.c.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

clean:
	rm -f $(OBJ) $(OUT)

