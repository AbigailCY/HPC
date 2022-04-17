#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
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

#define THREADS_PER_BLOCK 1024

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


void vec_mul(double *z, const double* v, const double* w, long N) {
  double prod = 0;
  #pragma omp parallel for schedule(static) reduction(+:prod)
  for (long i = 0; i < N; i++) {
  	prod += v[i] * w[i];
  }
  z[0] = prod;
}

void matrix_vec_mul(double* z, double* A, double* x,long M, long N) {
  for (long i = 0; i < M; i++) {
    vec_mul(z+i, A+i*N,x,N);
  }
}


__global__ void vec_mul_kernel(double *z, const double* a, const double* b, long N){
  __shared__ double temp[THREADS_PER_BLOCK];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    temp[threadIdx.x] = a[idx] * b[idx];
  }
  __syncthreads();
  if ( threadIdx.x == 0) {
    double sum = 0;
    for (long i = 0; i < THREADS_PER_BLOCK; i++) {
        sum += temp[i];
    }
    atomicAdd(z, sum);
  }
}

__global__ void vec_mul_kernel1(double *z, const double* a, const double* b, long N){
  __shared__ double temp[THREADS_PER_BLOCK];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int x = threadIdx.x;
  if (idx < N) {
    temp[threadIdx.x] = a[idx] * b[idx];
  }
  __syncthreads();

  int i  = blockDim.x / 2 ;
  while ( i!=0 ) {
    if ( x < i )
      temp[x] += temp[x + i] ;
      __syncthreads();
      i/=2 ;
  }
  if ( x == 0 ) atomicAdd(z, temp[0]); ;
}

void matrix_vec_mul_gpu(double* z, double* A, double* x,long M, long N) {
  for (long i = 0; i < M; i++) {
    vec_mul_kernel1<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(z+i, A+i*N,x,N);
  }
}



int main() {
  long M = (1UL<<13);
  long N = (1UL<<13);

  double * A, *A_d, *x, *x_d, *z, *z_d, *z_ref;
  A = (double*) malloc(M*N * sizeof(double));
  x = (double*) malloc(N * sizeof(double));
  z = (double*) malloc(M * sizeof(double));
  z_ref = (double*) malloc(M * sizeof(double));

  checkCuda( cudaMalloc(&A_d, M*N * sizeof(double)));
  checkCuda( cudaMalloc(&x_d, N * sizeof(double)));
  checkCuda( cudaMalloc(&z_d, M * sizeof(double)));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < M; i++) {
      z[i] = 0;
      z_ref[i] = 0;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
      x[i] = i+2;
  }
  #pragma omp parallel for schedule(static) collapse(2)
  for (long i = 0; i < M; i++) {
      for (long j = 0; j < N; j++) { 
        A[i*N+j]=j+1+(M-i);
      }
  }

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0));
  printf("\nDevice : %s\n", prop.name);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("\nM %ld, N %ld\n", M, N);


  double tt = omp_get_wtime();
  matrix_vec_mul(z_ref, A, x, M,N);
  tt = omp_get_wtime()-tt;
  printf("CPU time %f, Bandwidth = %f GB/s\n",tt, 3*M*N*sizeof(double) / tt/1e9);



  tt = omp_get_wtime();
  checkCuda( cudaMemcpy(A_d, A, M*N*sizeof(double), cudaMemcpyHostToDevice));
  checkCuda( cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice));
  checkCuda( cudaMemcpy(z_d, z, M*sizeof(double), cudaMemcpyHostToDevice));

  double tt1 = omp_get_wtime();
  matrix_vec_mul_gpu(z_d, A_d, x_d, M,N);
  cudaDeviceSynchronize();
  tt1 = omp_get_wtime()-tt1;

  checkCuda( cudaMemcpy(z, z_d, M*sizeof(double), cudaMemcpyDeviceToHost));
  tt = omp_get_wtime()-tt;
  printf("GPU time %f, Bandwidth = %f GB/s\n", tt, 3*M*N*sizeof(double) / tt/1e9);
  printf("GPU inner time %f\n", tt1);

  double err = 0;
  for (long i = 0; i < M; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);
  


  free(A);
  free(x);
  free(z);
  free(z_ref);
  checkCuda( cudaFree(A_d));
  checkCuda( cudaFree(x_d));
  checkCuda( cudaFree(z_d));
}