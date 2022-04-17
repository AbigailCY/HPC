#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h>
#include <assert.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

#define THREADS_PER_BLOCK 32
// export OMP_NUM_THREADS=10 to change number of threads

__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

double jacobi2D(double *u, double *u0, double *f, long n);

__global__ void jacobi_update_kernel(double *u, double *u0, const double *f, long n, double *norm);
void jacobi_kernel(double *u, double *u0, double *f, long n, double *norm_d, double *norm);



int main(int argc, char** argv) {

    long n = argv[1] ? atoi(argv[1]):256;
    
    double* u = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
    double* u_h = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
    double* u0 = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
    double* f = (double*) aligned_malloc(n*n * sizeof(double));
    double* norm = (double*) aligned_malloc(sizeof(double));
    double norm_ref;

    for (long i = 0; i < n+2; i++) {
        for (long j = 0; j < n+2; j++) {
            u[i+j*(n+2)] = 0;
            u0[i+j*(n+2)] = 0;
            u_h[i+j*(n+2)] = 0;
            if (i < n && j < n) {
                f[i+j*n] = 1;
            }
        }
    }
    norm[0]=0;

    double *u_d, *u0_d, *f_d, *norm_d;
    checkCuda(cudaMalloc(&u_d, (n+2) * (n+2)  * sizeof(double)));
    checkCuda(cudaMalloc(&u0_d, (n+2) * (n+2)  * sizeof(double)));
    checkCuda(cudaMalloc(&f_d, n*n * sizeof(double)));
    checkCuda(cudaMalloc(&norm_d, sizeof(double)));

    double tt1 = omp_get_wtime();
    checkCuda(cudaMemcpy(u_d, u, (n+2) * (n+2)*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(u0_d, u0, (n+2) * (n+2)*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(f_d, f, n*n*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(norm_d, norm, sizeof(double), cudaMemcpyHostToDevice));
    tt1 = omp_get_wtime()-tt1;
    

    double tt;
    tt = omp_get_wtime();
    norm_ref = jacobi2D(u, u0, f, n);
    tt = omp_get_wtime()-tt;
    printf("CPU Jacobi time: %8f\n", tt); 

    
    tt = omp_get_wtime();
    jacobi_kernel(u_d, u0_d, f_d, n, norm_d, norm);
    cudaDeviceSynchronize();
    tt = omp_get_wtime()-tt;

    double tt2 = omp_get_wtime();
    checkCuda(cudaMemcpy(u_h, u_d, (n+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost));
    tt2 = omp_get_wtime()-tt2+tt1+tt;
    printf("GPU Jacobi inner time: %8f\n", tt); 
    printf("GPU Jacobi outer time: %8f\n", tt2); 

    // printf("norm %f\n",norm[0]);
    double err = fabs(norm_ref - norm[0]);
    for (long i = 0; i < n+2; i++) {
        for (long j = 0; j < n+2; j++) {
            err += fabs(u[i*(n+2)+j]-u_h[i*(n+2)+j]);
            // printf("(i %d j %d: %f) ", i, j ,u_h[i*(n+2)+j]);
        }
    }
    
    printf("Error = %f\n", err);
    printf("N: %d. dimGrid: %d %d %d. dimBlock: %d %d %d\n", n,
        THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1, n/THREADS_PER_BLOCK, n/THREADS_PER_BLOCK, 1);

    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice : %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);



    aligned_free(u);
    aligned_free(u0);
    aligned_free(f);
    aligned_free(norm);
    checkCuda(cudaFree(u_d));
    checkCuda(cudaFree(u0_d));
    checkCuda(cudaFree(f_d));
    checkCuda(cudaFree(norm_d));
    return 0;
}


double jacobi2D(double *u, double *u0, double *f, long n) {
    double norm;
    long iter = 0;
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
        #pragma omp parallel for collapse(2) reduction(max:norm)
        for (long i = 1; i < n+1; i++) {
            for (long j = 1; j < n+1; j++) {
                double err = fabs(f[(i-1)*n+j-1] - (4*u[i*(n+2)+j]-u[(i-1)*(n+2)+j] - u[i*(n+2)+j-1] - u[(i+1)*(n+2)+j] - u[i*(n+2)+j+1])*(n+1)*(n+1));
                if (err > norm) norm = err;
            }
        }
    } while ((norm > 1e-5) && (iter < 5000));
    return norm;
}


__global__ void jacobi_update_kernel(double *u, double *u0, const double *f, long n, double *norm) {
    __shared__ double temp_norm[THREADS_PER_BLOCK*THREADS_PER_BLOCK];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x+1;
    int idy = blockIdx.y * blockDim.y + threadIdx.y+1;

    if (idx == 1 && idy == 1) norm[0]=0;

    if (idx < n+1 && idy < n+1) {
        u0[idx*(n+2)+idy] = (f[(idx-1)*n+idy-1]/(n+1)/(n+1) + u[(idx-1)*(n+2)+idy] + u[idx*(n+2)+idy-1] + u[(idx+1)*(n+2)+idy] + u[idx*(n+2)+idy+1])/4.0;
    }
    __syncthreads();

    if (idx < n+1 && idy < n+1) {
        double err = fabs(f[(idx-1)*n+idy-1] - (4*u0[idx*(n+2)+idy]-u0[(idx-1)*(n+2)+idy] - u0[idx*(n+2)+idy-1] - u0[(idx+1)*(n+2)+idy] - u0[idx*(n+2)+idy+1])*(n+1)*(n+1));
        temp_norm[threadIdx.x * THREADS_PER_BLOCK + threadIdx.y] = err;
    }
    __syncthreads();
    if ( threadIdx.x * THREADS_PER_BLOCK + threadIdx.y == 0) {
        double sum = 0;
        for (long i = 0; i < THREADS_PER_BLOCK*THREADS_PER_BLOCK; i++) {
            if (temp_norm[i] > sum) sum = temp_norm[i];
        }
        atomicMax(norm, sum);
    }
}

void jacobi_kernel(double *u, double *u0, double *f, long n, double *norm_d, double *norm) {
    dim3 block ( n/THREADS_PER_BLOCK , n/THREADS_PER_BLOCK) ;
    dim3 grid (THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    long iter = 0;
    do {
        iter += 1;
        jacobi_update_kernel<<<block, grid>>>(u, u0, f, n, norm_d);

        double* utemp = u;
        u = u0;
        u0 = utemp;
        
        checkCuda(cudaMemcpy(norm, norm_d, sizeof(double), cudaMemcpyDeviceToHost));
    } while ((norm[0] > 1e-5) && (iter < 5000));
}