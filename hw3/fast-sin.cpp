#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

// coefficients in the Taylor series expansion of cos(x)
static constexpr double c0  = (double) 1;
static constexpr double c2  = -1/((double)2);
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c12 = 1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
// cos(x) = c0 + c2*x^2 + c4*x^4 + c6*x^6 + x8*x^8 + c10*x^10 + c12*x^12

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}



void sin4_taylor_all(double* sinx, const double* x) {

  int false_flag = 1;
  for (int i = 0; i < 4; i++) {
    double x0  = x[i];
    if (x0 < 0) {
      false_flag = -1;
      x0 = -x0;
    } else {false_flag = 1;}
    int k = (int)(x0/(M_PI/2)+0.5);
    double x1 = x0 - (M_PI/2)*k;

    k=k%4;
    int sign = 1;
    if (k>=2) sign = -1;
    double sin = (k+1)%2*sign;
    double cos = k%2*sign;

    double x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    x2  = x1 * x1;
    x3  = x1 * x2;
    x5  = x3 * x2;
    x7  = x5 * x2;
    x9  = x7 * x2;
    x11 = x9 * x2;

    double s = sin *x1;
    s += sin*x3  * c3;
    s += sin*x5  * c5;
    s += sin*x7  * c7;
    s += sin*x9  * c9;
    s += sin*x11 * c11;

    x4  = x3 * x1;
    x6  = x5 * x1;
    x8  = x7 * x1;
    x10  = x9 * x1;
    x12  = x11 * x1;
    s += cos*c0;
    s += cos*x2  * c2;
    s += cos*x4  * c4;
    s += cos*x6  * c6;
    s += cos*x8  * c8;
    s += cos*x10  * c10;
    s += cos*x12  * c12;
    s *= false_flag;
    sinx[i] = s;

  }
}

void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)

// Adding more terms using AVX intrinsics
  __m256d x1, x2, x3, x5, x7, x9, x11;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x5  = _mm256_mul_pd(x3, x2);
  x7  = _mm256_mul_pd(x5, x2);
  x9  = _mm256_mul_pd(x7, x2);
  x11  = _mm256_mul_pd(x9, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

double *sign_K_sin = (double*) aligned_malloc(16*sizeof(double));
void sin4_vector_all(double* sinx, const double* x) {
  typedef Vec<double,4> Vec4;
  Vec4 x1;

  double x0;
  for (int j = 0; j < 4; j++){
      x0  = x[j];
      if (x0 < 0) {
          sign_K_sin[j] = -1.0;
          x0 = -x0;
      } else {sign_K_sin[j] = 1.0;}
      int K0=(int)(x0/(M_PI/2)+0.5);
      sign_K_sin[j+4] = (double) K0*M_PI/2;
      K0=K0%4;
      int sign = 1;
      if (K0>=2) sign = -1;
      sign_K_sin[j+8] = (K0+1)%2*sign;
      sign_K_sin[j+12] = K0%2*sign;
  }
  x1  = (Vec4::LoadAligned(x))*(Vec4::LoadAligned(sign_K_sin));
  x1 -= (Vec4::LoadAligned(sign_K_sin+4));

  Vec4 x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
  x2  = x1 * x1;
  x3  = x1 * x2;
  x5  = x3 * x2;
  x7  = x5 * x2;
  x9  = x7 * x2;
  x11 = x9 * x2;

  Vec4 s = x1;
  s += x3  * c3;
  s += x5  * c5;
  s += x7  * c7;
  s += x9  * c9;
  s += x11 * c11;
  s *= Vec4::LoadAligned(sign_K_sin+8);

  x4  = x3 * x1;
  x6  = x5 * x1;
  x8  = x7 * x1;
  x10  = x9 * x1;
  x12  = x11 * x1;
  Vec4 s1 = c0;
  s1 += x2  * c2;
  s1 += x4  * c4;
  s1 += x6  * c6;
  s1 += x8  * c8;
  s1 += x10  * c10;
  s1 += x12  * c12;
  s1 *=Vec4::LoadAligned(sign_K_sin+12);
  s += s1;
  s *= Vec4::LoadAligned(sign_K_sin);
  s.StoreAligned(sinx);
}


void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3, x5, x7, x9, x11;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;
  x5  = x3 * x2;
  x7  = x5 * x2;
  x9  = x7 * x2;
  x11 = x9 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s += x5  * c5;
  s += x7  * c7;
  s += x9  * c9;
  s += x11 * c11;

  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI*200; // [-pi*100,pi*100]
    // x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      // sin4_taylor(sinx_taylor+i, x+i);
      sin4_taylor_all(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      // sin4_vector(sinx_vector+i, x+i);
      sin4_vector_all(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
  aligned_free(sign_K_sin);
}

