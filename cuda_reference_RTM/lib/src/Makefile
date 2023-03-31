# source files.
SRC = functions.c

OBJ = $(SRC:.c=.o)

OUT = ../libsource.a

# include directories
INCLUDES = -I../include 

# C compiler flags
CFLAGS = -O3 -Wall -Wextra -pedantic

# compiler
CC = gcc

.SUFFIXES: .c

default: $(OUT)

.c.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

clean:
	rm -f $(OBJ) $(OUT)

