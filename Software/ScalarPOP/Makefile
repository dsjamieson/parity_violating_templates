CXX = mpicxx
CXXERRFLAGS = -fmax-errors=1 -Werror
CXXFLAGS = -O3 -Wall -fopenmp -std=c++20 $(CXXERRFLAGS)
LIB = -lfftw3_mpi -lfftw3_omp -lfftw3 -lgsl -lgslcblas -lhdf5
INC = -I./include 
BIN = bin

SRC = $(wildcard src/*.cpp)
SRC += $(wildcard src/*/*.cpp)
SRC += $(wildcard src/*/*/*.cpp)
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))

default: all

all: $(OBJ)
	$(CXX) $(CXXFLAGS) -o bin/scalar_pop  $^ $(INC) $(LIB)

build/%.o: src/%.cpp Makefile include/*
	$([${OBJ}])
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@ $(LIB)

phony:
clean:
	find build -type f -not -name '.*' -delete
	find bin -type f -not -name '.*' -delete
