#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

long jacobi2D(double *u, double *u0, double *f, long n);

int main(int argc, char** argv) {

    long n = atoi(argv[1]);

    double* u = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
    double* u0 = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
    double* f = (double*) aligned_malloc(n*n * sizeof(double));

    for (long i = 0; i < n+2; i++) {
        for (long j = 0; j < n+2; j++) {
            u[i+j*(n+2)] = 0;
            u0[i+j*(n+2)] = 0;
            if (i < n && j < n) {
                f[i+j*n] = 1;
            }
        }
    }


    Timer t;
    t.tic();
    long iter = jacobi2D(u, u0, f, n);
    double time = t.toc();

    printf("Jacobi time: %8f\n", time); 

    aligned_free(u);
    aligned_free(u0);
    aligned_free(f);

}


long jacobi2D(double *u, double *u0, double *f, long n) {
    double norm;
    double err;
    long iter = 0;
    // h = 1/(n+1)
    do {
        norm = 0;
        iter += 1;

        // update
        #pragma omp parallel for collapse(2)
        for (long i = 1; i < n+1; i++) {
            for (long j = 1; j < n+1; j++) {
                u0[i*(n+2)+j] = (f[(i-1)*n+j-1]/(n+1)/(n+1)+u[(i-1)*(n+2)+j] + u[i*(n+2)+j-1] + u[(i+1)*(n+2)+j] + u[i*(n+2)+j+1])/4.0;
            }           
        }
        // copy u0 to u
        double* utemp = u;
        u = u0;
        u0 = utemp;

        // compute error norm
        for (long i = 1; i < n+1; i++) {
            for (long j = 1; j < n+1; j++) {
                err = f[(i-1)*n+j-1] - (4*u[i*(n+2)+j]-u[(i-1)*(n+2)+j] - u[i*(n+2)+j-1] - u[(i+1)*(n+2)+j] - u[i*(n+2)+j+1])*(n+1)*(n+1);
                norm = std::max(fabs(err),norm);
            }
        }

    } while ((norm > 1e-6) && (iter < 5000));
    
    printf("iter %d; norm %8f; ",iter ,  norm);
    return iter;
    // for (long i = 0; i < n+2; i++) {
    //     for (long j = 0; j < n+2; j++) {
    //         printf("%10f ", u[i*(n+2)+j]);
    //     }
    //     printf("\n");
    // }
}