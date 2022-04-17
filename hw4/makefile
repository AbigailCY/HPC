CXX = nvcc

#need to change to -arch=sm_35 if Compute capability < 6.0
CXXFLAGS = -Xcompiler -fopenmp -arch=sm_61 -std=c++11

# turning off auto-vectorization since this can make hand-vectorized code slower
#CXXFLAGS += -fno-tree-vectorize

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
