CWPROOT = ../../lib/cwp

# source files.
SRC = taper.c

OBJ = $(SRC:.c=.o)

OUT = ../../lib/libboundary.a

# include directories
INCLUDES = -I../../include/boundary -I$(CWPROOT)/include

# C compiler flags
CFLAGS = -O3 -Wall -Wextra -pedantic -fpermissive

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

