/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{

int nthreads, i, tid;
float total;
/*** Spawn parallel region ***/
// Bug: tid should be private for each thread. total should be shared
#pragma omp parallel shared(total, nthreads) private(tid,i)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  // Bug: need to add reduction(+:total) in omp instruction because total needs to be updated and added to i, which is the looping index.
  #pragma omp for schedule(dynamic,10) reduction(+:total)
  for (i=0; i<1000000; i++) {
     total = total + i*1.0;
  }
  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}

