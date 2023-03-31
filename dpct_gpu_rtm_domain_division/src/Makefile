CWPROOT = ../lib/cwp

IDIR = ../include
_DIRS = timestep source boundary

CC = g++
CFLAGS = -w -I$(IDIR) $(patsubst %,-I$(IDIR)/%,$(_DIRS)) -I$(CWPROOT)/include -L$(CWPROOT)/lib -L../lib/ -O3 -Wall -Wextra -pedantic -fopenmp

ODIR = .
LIBS = $(patsubst %,-l%,$(_DIRS)) -lsu -lpar -lcwp -lm
BIN = ./rtm_main

_DEPS = 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = rtm_main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(CFLAGS)

rtm_main: $(OBJ)
	for dir in ${_DIRS}; do \
		cd ./$$dir; $(MAKE); cd ..; \
        done
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)
	mv $(BIN) ../build/rtm_main
	$(CC) -o mod_main mod_main.cpp $(CFLAGS) $(LIBS)
	mv mod_main ../build/mod_main

.PHONY: clean allclean

allclean:
	for dir in ${_DIRS}; do \
                cd ./$$dir; $(MAKE) clean; cd ..; \
        done
	rm -f $(ODIR)/*.o core*

clean:
	rm -f $(ODIR)/*.o core*
	rm -f ../build/rtm_main
	rm -f ../build/mod_main

