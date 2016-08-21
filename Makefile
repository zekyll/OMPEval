CXXFLAGS += -O3 -std=c++11 -Wall -Wpedantic

ifdef SYSTEMROOT
    CXXFLAGS += -lpthread
else
    CXXFLAGS += -pthread
endif

ifeq ($(SSE4),1)
	CXXFLAGS += -msse4.2
endif

SRCS := $(wildcard omp/*.cpp)
OBJS := ${SRCS:.cpp=.o}

all: lib/ompeval.a test

lib:
	mkdir lib

lib/ompeval.a: $(OBJS) | lib
	ar rcs $@ $^

test: test.cpp benchmark.cpp lib/ompeval.a
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	$(RM) test test.exe lib/ompeval.a $(OBJS)
