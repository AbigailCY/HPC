CFLAGS = -march=native -O3 -fopenmp
CXXFLAGS = -march=native -O3 -fopenmp -std=c++11
CXXFLAGS2 = -march=native -O2 -fopenmp -std=c++11
EXE =  MMult1  omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp val_test01_solved val_test02_solved

all: $(EXE)

val_test01_solved: val_test01_solved.cpp
	g++ val_test01_solved.cpp -o val_test01_solved
val_test02_solved: val_test02_solved.cpp
	g++ val_test02_solved.cpp -o val_test02_solved

MMult1: MMult1.cpp
	g++ MMult1.cpp -o MMult1 $(CXXFLAGS2)

omp_solved2: omp_solved2.c
	g++ omp_solved2.c -o omp_solved2 $(CFLAGS)
omp_solved3: omp_solved3.c
	g++ omp_solved3.c -o omp_solved3 $(CFLAGS)
omp_solved4: omp_solved4.c
	g++ omp_solved4.c -o omp_solved4 $(CFLAGS)
omp_solved5: omp_solved5.c
	g++ omp_solved5.c -o omp_solved5 $(CFLAGS)
omp_solved6: omp_solved6.c
	g++ omp_solved6.c -o omp_solved6 $(CFLAGS)


jacobi2D-omp: jacobi2D-omp.cpp
	g++ jacobi2D-omp.cpp -o jacobi2D-omp $(CXXFLAGS)
gs2D-omp: gs2D-omp.cpp
	g++ gs2D-omp.cpp -o gs2D-omp $(CXXFLAGS)


clean:
	rm $(EXE)$