CXXFLAGS += -O3 -std=c++11 -pthread -Wall
ifeq ($(SSE4),1)
	CXXFLAGS += -msse4.2
endif
SRCS := $(wildcard omp/*.cpp)
OBJS := ${SRCS:.cpp=.o}

all: lib/ompeval.a test

lib/ompeval.a: $(OBJS)
	ar rcs lib/ompeval.a $(OBJS)

test: test.cpp lib/ompeval.a
	$(CXX) $(CXXFLAGS) -o test test.cpp lib/ompeval.a

clean:
	$(RM) test lib/ompeval.a $(OBJS)
