#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  prefix_sum[0] = 0;
  long *sump;
  #pragma omp parallel 
  {
    const int pid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    #pragma omp single 
    {
      sump = (long*) malloc((nthreads+1) * sizeof(long));
      sump[0] = 0;
    }
    long sum = 0;
    #pragma omp for schedule(static) nowait
    for (long i = 1; i < n; i++) {
      sum += A[i-1];
      prefix_sum[i] = sum;
    }
    sump[pid+1] = sum;
    #pragma omp barrier
    #pragma omp single
    {
        long tmp = 0;
        for (long i = 1; i < (nthreads + 1); i++) {
            tmp += sump[i];
            sump[i] = tmp;
        }
    }
    #pragma omp for schedule(static)
      for (long i = 1; i < n; i++) {
        prefix_sum[i] += sump[pid];
      }
  }
  free(sump);
  
}

int main(int argc, char *argv[]) {
  int num_threads = argv[1] ? atoi(argv[1]):10;
  omp_set_num_threads(num_threads);
  printf("number of threads %d \n",num_threads);
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) {
    err = std::max(err, std::abs(B0[i] - B1[i]));
  }
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
