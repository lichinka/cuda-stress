char program[] = "CUDALucas v2.05.1";
/* CUDALucas.c
   Shoichiro Yamada Oct. 2010

   This is an adaptation of Richard Crandall lucdwt.c, John Sweeney MacLucasUNIX.c,
   and Guillermo Ballester Valor MacLucasFFTW.c code.
   Improvement From Prime95.

   It also contains mfaktc code by Oliver Weihe and Eric Christenson
   adapted for CUDALucas use. Such code is under the GPL, and is noted as such.
*/

/* Include Files */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#ifndef _MSC_VER
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda_safecalls.h"
#include "parse.h"


/* In order to have the gettimeofday() function, you need these includes on Linux:
#include <sys/time.h>
#include <unistd.h>
On Windows, you need
#include <winsock2.h> and a definition for
int gettimeofday (struct timeval *tv, struct timezone *) {}
Both platforms are taken care of in parse.h and parse.c. */
/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */

/************************ definitions ************************************/
#ifdef _MSC_VER
#define strncasecmp strnicmp
#endif

/**************************************************************
***************************************************************
*                                                             *
*                       Global Variables                      *
*                                                             *
***************************************************************
**************************************************************/
double *g_x;                      //gpu data
double *g_ttp1;                   //weighting factors for splicing step
double *g_ttp2;                   //weighting factors for splicing step
double *g_ct;                     //factors used in multiplication kernel enabling real as complex
int    *g_data;                   //integer values of data for splicing step
int    *g_carry;                  //carry data for splicing step
long long int    *g_carryl;       //carry data for splicing step, 64 bit version
int    *g_xint;                   //integer copy of gpu data for transfer to cpu
float  *g_err;                    //current maximum error
double *g_ttp;                    //weighting factors
unsigned int *g_size1;            //information on the number of bits each digit of g_x uses, 0 or 1
unsigned int *g_s;                //same information on device
//unsigned int *g_s1;               //large product of ttp[grp]*ttp[col] information
__constant__ double g_ttpinc[3];  //factors for obtaining weights of adjacent digits
__constant__ int    g_qn[2];      //base size of bit values for each digit, adjusted by size data above

cufftHandle    g_plan;
cudaDeviceProp g_dev;             //structure holding device property information

int g_ffts[256];                  //array of ffts lengths
int g_fft_count = 0;              //number of ffts lengths in g_ffts
int g_fftlen;                     //index of the fft currently in use
int g_lbi = 0;                    //index of the smallest feasible fft length for the current exponent
int g_ubi = 256;                  //index of the largest feasible fft length for the current exponent
int g_thr[2] = {256, 128};        //current threads values
int g_cpi;                        //checkpoint interval
int g_ri;                         //report interval
int g_ei;                         //error check interval
int g_er = 85;                    //error reset value
int g_pf;                         //polite flag
int g_po;                         //polite value
int g_sl;                         //polite value
int g_sv;                         //polite flag
int g_bc;                         //big carry flag
int g_qu;                         //quitting flag
int g_sf;                         //safe all checkpoint files flag
int g_rt;                         //residue check flag
int g_ro;                         //roundoff test flag
int g_dn;                         //device number
int g_df;                         //show device info flag
int g_ki;                         //keyboard input flag
int g_th;                         //threads flag
int g_el;                         //error limit value

char g_folder[192];               //folder where savefiles will be kept
char g_input_file[192];           //input file name default worktodo.txt
char g_RESULTSFILE[192];          //output file name default results.txt
char g_INIFILE[192] = "CUDALucas.ini"; //initialization file name
char g_AID[192];                  // Assignment key
char g_output[50][32];
char g_output_string[192];
char g_output_header[192];
int  g_output_code[50];
int  g_output_interval;


typedef struct iteration_data
{
  int pr;
  int it;
  int os;
  unsigned long long tt;
  unsigned long long ta;
  unsigned ia;
} iteration_data;
/**************************************************************
***************************************************************
*                                                             *
*             Kernels and other device functions              *
*                                                             *
***************************************************************
**************************************************************/


/**************************************************************
*                                                             *
*             Functions for setting constant memory           *
*                                                             *
**************************************************************/

// Factors to convert from one weighting factor to an adjacent
// weighting factor.
void set_ttpinc(double *h_data)
{
  cudaMemcpyToSymbol(g_ttpinc, h_data, 2 * sizeof(double));
}

// Set to q/n, the number of bits each digit uses is given by
// numbits in the rcb and splicing kernels. Numbits is q/n + size.
// We can avoid many memory transfers by having this in constant
// memory.
void set_qn(int *h_qn)
{
  cudaMemcpyToSymbol(g_qn, h_qn, sizeof(int));
}

/**************************************************************
*                                                             *
*                       Device functions                      *
*                                                             *
**************************************************************/

// inline ptx rounding function
# define RINT(x)  __rintd(x)

__device__ static double __rintd (double z)
{
  double y;
  asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (z));
  return (y);
}

/**************************************************************
*                                                             *
*                           Kernels                           *
*                                                             *
**************************************************************/

// These two used in memtest only
__global__ void copy_kernel (double *in, int n, int pos, int s)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

    in[s * n + index] = in[pos * n + index];
}


__global__ void compare_kernel (double *in1,  double *in2, volatile int *compare)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  int temp;
  double d1, d2;

  d1 = in1[threadID];
  d2 = in2[threadID];
  temp = (d1 != d2);
  if(temp > 0) atomicAdd((int *) compare, 1);
}

// Applies the irrational base weights to balanced integer data.
// Used in initialization at the (re)commencement of a test.
__global__ void apply_weights (double *out, int *in, double *ittp, unsigned int *s)
{
  int	index, grp, col, i;
  __shared__ double ttp_grp[32];
  __shared__ double ttp_col[64];
  __shared__ unsigned int biglit1_flags[32];

  index = (blockIdx.x << 10) + threadIdx.x;
  grp = threadIdx.x >> 5;
  col = threadIdx.x & 31;
  if (grp == 0)
  {
    ttp_col[col] = ittp[col<<1];
    ttp_col[col + 32] = 0.5 * ttp_col[col];
  }
  if (grp == 1)
  {
    i = (blockIdx.x << 6) + (col << 1);
    ttp_grp[col] = ittp[i + 64];
    biglit1_flags[col] = s[i + 1];
  }
  __syncthreads();
  do
  {
    i = col;
    if (biglit1_flags[grp] & (1 << col)) i += 32;
    out[index] = (double) in[index] * ttp_grp[grp] * ttp_col[i];
    grp += blockDim.x >> 5;
    index += blockDim.x;
  }
  while(grp < 32);
}

// The pointwise multiplication of the fft'ed data.
// We are using real data interpreted as complex with odd
// indices being the imaginary part. The work with the number
// (wkr, wki) is needed for this.
__global__ void square (int n, double *a, double *ct)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int nc = n >> 2;
  double wkr, wki, xr, xi, yr, aj, aj1, ak, ak1;

  if (j)
  {
    wkr = 0.5 - ct[nc - j];
    wki = ct[j];
    j <<= 1;
    nc = n - j;
    aj = a[j];
    aj1 = a[1 + j];
    ak = a[nc];
    ak1 = a[1 + nc];

    xr = aj - ak;
    xi = aj1 + ak1;
    yr = wkr * xr - wki * xi;
    xi = wkr * xi + wki * xr;

    aj -= yr;
    aj1 -= xi;
    ak += yr;
    ak1 -= xi;

    xr = (aj - aj1) * (aj + aj1);
    xi = 2.0 * aj * aj1;
    yr = (ak - ak1) * (ak + ak1);
    ak = 2.0 * ak * ak1;

    aj1 = xr - yr;
    ak1 = xi + ak;
    aj = wkr * aj1 + wki * ak1;
    aj1 = wkr * ak1 - wki * aj1;

    a[j] = xr - aj;
    a[1 + j] = aj1 - xi;
    a[nc] = yr + aj;
    a[1 + nc] =  aj1 - ak;
  }
  else
  {
    j = n >> 1;
    aj = a[0];
    aj1 = a[1];
    xi = aj - aj1;
    aj += aj1;
    xi *= xi;
    aj *= aj;
    aj1 = 0.5 * (xi - aj);
    a[0] = aj + aj1;
    a[1] = aj1;
    xr = a[j];
    xi = a[1 + j];
    a[j] = (xr + xi) * (xr - xi);
    a[1 + j] = -2.0 * xr * xi;
  }
}
#define TF (1 << SC)
#define B1  10240  // 10K  7 --> 8
#define B2  49152  // 48K  8 --> 9
#define B3  98304  // 96K  9 --> 10
#define SC  10     // SC <= 2 * BA
#define TT  128
#define BA 5       // 5 for 32x32, 6 for 64x64
template < int sc , int error, int report>
__global__ void   rcb (double *in,
                       int *xint,
		                   int *data,
		                   double *ittp,
		                   unsigned int *s,
                       int *carry_out,
		                   volatile float *err,
		                   float maxerr,
		                   int digit,
		                   int bit)
{
    long long int bigint;
    int val, numbits, mask, shifted_carry;
    double ttp, ttmp, trint, tval;
    int	index, grp, col, i;
    float ferr, ferr_max;
    // Throw in that extra one at the end to unify notation and avoid another comparison.
    // We can probably reduce this to 129. 256 threads is much slower than 128 on all cards so far tested.
    __shared__ int carry[257];
    __shared__ double ttp_grp[32], ttmp_grp[32];
    __shared__ double ttp_col[64], ttmp_col[64];
    __shared__ unsigned int biglit_flags [32];
    __shared__ unsigned int biglit1_flags [32];
    __shared__ int carry_for_next_section;

    index = (blockIdx.x << sc) + threadIdx.x; // bI * 128, 256, 512, 1024
    grp = threadIdx.x >> 5;
    col = threadIdx.x & 31;
    ferr_max = 0.0;

    if (grp == 0)
    {
      // Reading column data into shared memory decreases iteration times from
      // 5.3000 to 5.2900, 0.19% increase
      i = col << 1;
      ttp_col[col] = ittp[i];
      ttmp_col[col] = ittp[i + 1];
      ttp_col[col + 32] = 0.5 * ttp_col[col];
      ttmp_col[col + 32] = 2.0 * ttmp_col[col];
    }
    if (grp == 1)
    {
      i = (blockIdx.x << (sc - 4)) + (col << 1);
      ttp_grp[col] = ittp[i + 64];
      ttmp_grp[col] = ittp[i + 65];
      biglit_flags[col] = s[i];
      biglit1_flags[col] = s[i + 1];
    }
    if(threadIdx.x == 0) carry_for_next_section = 0;
    __syncthreads ();

    //unrolling first iteration of the loop decreases iteration times from
    // 5.3400 to 5.2900, ~0.95%

    // There are occasional 2 way bank conflicts here, but because of the shared
    // memory access pattern, no amount of tweaking will get rid of it.
    // Also, using constant memory for column data is much slower.
    i = col;
    if (biglit1_flags[grp] & (1 << col)) i += 32;
    ttp = ttp_grp[grp] * ttp_col[i];
    ttmp = ttmp_grp[grp] * ttmp_col[i];

    // size information can also be gotten by comparing ttp < 2^((q % n) / n)
    // g_ttpinc[2] = 2^((q % n) / n) - k where k < 2^(1 / n) - 1 for any
    // conceivable value of n . 2^(1 / (65536 * 1024)) = 1.0000000103287
    // so k = 10^(-9) suffices. The k would not be needed, except that the last
    // digit has ttp = 2^((q % n) / n) so inexact multiplication gives inccorect results
    // in the comparison. Unfortunately, this method gives slower iteration times.
    numbits = g_qn[0];// + (ttp < g_ttpinc[2]);
    if (biglit_flags[grp] & (1 << col)) numbits++;
    mask = -1 << numbits;

    tval = in[index] * ttmp;
    trint = RINT (tval);
    bigint = (long long int) trint;
    if(error)
    {
      ferr = fabs ((float) (tval - trint));
      if (ferr > ferr_max) ferr_max = ferr;
    }

    if (index == digit) bigint -= bit;
    if (threadIdx.x == 0) bigint += carry_for_next_section;
    carry[threadIdx.x + 1] = (int) (bigint >> numbits);
    val = ((int) bigint) & ~mask;
    __syncthreads ();

    // Without the carries being offset, this section would need an extra sync
    // but with the offset, each thread is reading and writing the same location.
    if (threadIdx.x) val += carry[threadIdx.x];
    shifted_carry = val - (mask >> 1);
    carry[threadIdx.x] = shifted_carry >> numbits;
    val -= shifted_carry & mask;
    __syncthreads ();

    if (threadIdx.x) val += carry[threadIdx.x - 1];
    else carry_for_next_section = carry[blockDim.x] + carry[blockDim.x-1];
    in[index] = (double) val * ttp;
    if (report) xint[index] = val;

    // to be sent to splice kernel, writing to a separate array to coalesce memory
    // reads by the splice kernel. We do not need this except in the first iteration
    // of the loop which is why unrolling that first iteration is faster.
    if (threadIdx.x < 2 ) data[(blockIdx.x << 1) + threadIdx.x] = val;//if (grp == 0 && threadIdx.x < 2 )

    grp += blockDim.x >> 5;  //     4
    index += blockDim.x;     //bd 128
    __syncthreads ();

  while(grp < (1 << (sc - 5)))
  {
    i = col;
    if (biglit1_flags[grp] & (1 << col)) i += 32;//(ttp >= 2.0)
    ttp = ttp_grp[grp] * ttp_col[i]; //*g_ttpcol;
    ttmp = ttmp_grp[grp] * ttmp_col[i]; //*g_ttmpcol;

    numbits = g_qn[0];// + (ttp < g_ttpinc[2]);
    if (biglit_flags[grp] & (1 << col)) numbits++;
    mask = -1 << numbits;

    tval = in[index] * ttmp;
    trint = RINT (tval);
    bigint = (long long int) trint;
    if(error)
    {
      ferr = fabs ((float) (tval - trint));
      if (ferr > ferr_max) ferr_max = ferr;
    }

    if (index == digit) bigint -= bit;
    if (threadIdx.x == 0) bigint += carry_for_next_section;
    carry[threadIdx.x + 1] = (int) (bigint >> numbits);
    val = ((int) bigint) & ~mask;
    __syncthreads ();

    if (threadIdx.x) val += carry[threadIdx.x];
    shifted_carry = val - (mask >> 1);
    carry[threadIdx.x] = shifted_carry >> numbits;
    val -= shifted_carry & mask;
    __syncthreads ();

    if (threadIdx.x) val += carry[threadIdx.x - 1];
    else carry_for_next_section = carry[blockDim.x] + carry[blockDim.x-1];
    in[index] = (double) val * ttp;
    if (report) xint[index] = val;

    grp += blockDim.x >> 5;
    index += blockDim.x;
    __syncthreads ();
  }
  if (error && ferr_max > maxerr) atomicMax ((int*) err, __float_as_int (ferr_max));
  if (threadIdx.x == blockDim.x - 1)
  {
    // With the i, the threads diverge for only a possible assignment of 0;
    // old way without the i:
    //    if (blockIdx.x + 1 == gridDim.x) carry_out[0] = carry_for_next_section;
    //    else carry_out[blockIdx.x + 1] = carry_for_next_section;
    i = blockIdx.x + 1;
    if (i == gridDim.x) i = 0;
	  carry_out[i] = carry_for_next_section;
  }    /* Set maxerr after computing several error maximums locally */
}

// Splicing kernel, 32 bit version. Handles carries between blocks of rcb1
template <int report>
__global__ void splice (double *out,
                         int *xint,
                         int n,
                         int threads,
                         int *data,
                         int *carry_in,
                         double *ttp1)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads * threadID) << 1;
  int temp0, temp1;
  int mask, shifted_carry, numbits= g_qn[0];
  double temp;

  if (j < n) //make sure this index actually refers to some data
  {
    temp0 = data[threadID1] + carry_in[threadID];
    temp1 = data[threadID1 + 1];
    temp = ttp1[threadID];

    //extract and apply size information
    if(temp < 0.0)
    {
      numbits++;
      temp = -temp;
    }

    //do the last carry
    mask = -1 << numbits;
    shifted_carry = temp0 - (mask >> 1) ;
    temp0 = temp0 - (shifted_carry & mask);
    temp1 += (shifted_carry >> numbits);

    //apply weights before writing to memory
    out[j] = temp0 * temp;
    temp *= g_ttpinc[numbits == g_qn[0]];
    out[j + 1] = temp1 * temp;

    //write data to integer array when savefiles need writing
    if(report)
    {
      xint[j] = temp0;
      xint[j + 1] = temp1;
    }
  }
}

template < int sc , int error, int report>
__global__ void   rcbl (double *in,
                        int *xint,
		                    int *data,
		                    double *ittp,
		                    unsigned int *s,
                        long long int *carry_out,
		                    volatile float *err,
		                    float maxerr,
		                    int digit,
		                    int bit)
// Combine data and carry
// ittp_col, ittmp_col, ittp_grp, and ittmp_grp
// 0-31     32-63       64-95     96-128
//                      128-159   160-191, etc
// s, s1
{
    long long int bigint;
    int numbits;
    long long int mask, shifted_carry;
    double ttp, ttmp, trint, tval;
    int	index, grp, col, i;
    float ferr, ferr_max;
    __shared__ long long int carry[257];
    __shared__ double ttp_grp[32], ttmp_grp[32];
    __shared__ double ttp_col[64], ttmp_col[64];
    __shared__ unsigned int biglit_flags [32];
    __shared__ unsigned int biglit1_flags [32];
    __shared__ long long int carry_for_next_section;

    index = (blockIdx.x << sc) + threadIdx.x;
    grp = threadIdx.x >> 5;
    col = threadIdx.x & 31;
    ferr_max = 0.0;

    if (grp == 0)
    {
      i = col << 1;
      ttp_col[col] = ittp[i];
      ttmp_col[col] = ittp[i + 1];
      ttp_col[col + 32] = 0.5 * ttp_col[col];
      ttmp_col[col + 32] = 2.0 * ttmp_col[col];
    }
    if (grp == 1)
    {
      i = (blockIdx.x << (sc - 4)) + (col << 1);
      ttp_grp[col] = ittp[i + 64];
      ttmp_grp[col] = ittp[i + 65];
      biglit_flags[col] = s[i];
      biglit1_flags[col] = s[i + 1];
    }
    if(threadIdx.x == 0) carry_for_next_section = 0x0000000000000000;
    __syncthreads ();

    i = col;
    if (biglit1_flags[grp] & (1 << col)) i += 32;
    ttp = ttp_grp[grp] * ttp_col[i];
    ttmp = ttmp_grp[grp] * ttmp_col[i];

    numbits = g_qn[0];
    if (biglit_flags[grp] & (1 << col)) numbits++;
    mask = 0xffffffffffffffff << numbits;

    tval = in[index] * ttmp;
    trint = RINT (tval);
    bigint = (long long int) trint;
    if(error)
    {
      ferr = fabs ((float) (tval - trint));
      if (ferr > ferr_max) ferr_max = ferr;
    }

    if (index == digit) bigint -= bit;
    if (threadIdx.x == 0) bigint += carry_for_next_section;
    carry[threadIdx.x + 1] = bigint >> numbits;
    bigint &= ~mask;
    __syncthreads ();

    if (threadIdx.x) bigint += carry[threadIdx.x];
    shifted_carry = bigint - (mask >> 1);
    bigint -= (shifted_carry & mask);
    carry[threadIdx.x] = shifted_carry >> numbits;
    __syncthreads ();

    if (threadIdx.x) bigint += carry[threadIdx.x - 1];
    else carry_for_next_section = carry[blockDim.x] + carry[blockDim.x-1];
    in[index] = (double) bigint * ttp;
    if (report) xint[index] = (int) bigint;

    if (threadIdx.x < 2 ) data[(blockIdx.x << 1) + threadIdx.x] = (int) bigint;

    grp += blockDim.x >> 5;
    index += blockDim.x;
    __syncthreads ();

  while(grp < (1 << (sc - 5)))
  {
    i = col;
    if (biglit1_flags[grp] & (1 << col)) i += 32;
    ttp = ttp_grp[grp] * ttp_col[i];
    ttmp = ttmp_grp[grp] * ttmp_col[i];

    numbits = g_qn[0];
    if (biglit_flags[grp] & (1 << col)) numbits++;
    mask = 0xffffffffffffffff << numbits;

    tval = in[index] * ttmp;
    trint = RINT (tval);
    bigint = (long long int) trint;
    if(error)
    {
      ferr = fabs ((float) (tval - trint));
      if (ferr > ferr_max) ferr_max = ferr;
    }

    if (index == digit) bigint -= bit;
    if (threadIdx.x == 0) bigint += carry_for_next_section;
    carry[threadIdx.x + 1] = bigint >> numbits;
    bigint &= ~mask;
    __syncthreads ();

    if (threadIdx.x) bigint += carry[threadIdx.x];
    shifted_carry = bigint - (mask >> 1);
    bigint -= shifted_carry & mask;
    carry[threadIdx.x] = shifted_carry >> numbits;
    __syncthreads ();

    if (threadIdx.x) bigint += carry[threadIdx.x - 1];
    else carry_for_next_section = carry[blockDim.x] + carry[blockDim.x-1];
    in[index] = (double) bigint * ttp;
    if (report) xint[index] = (int) bigint;

    grp += blockDim.x >> 5;
    index += blockDim.x;
    __syncthreads ();
  }
  if (error && ferr_max > maxerr) atomicMax ((int*) err, __float_as_int (ferr_max));
  if (threadIdx.x == blockDim.x - 1)
  {
    i = blockIdx.x + 1;
    if (i == gridDim.x) i = 0;
	  carry_out[i] = carry_for_next_section;
  }
}

// Splicing kernel, 64 bit version.
template <int report>
__global__ void splicel (double *out,
                        int *xint,
                        int n,
                        int threads,
                        int *data,
                        long long int *carry_in,
                        double *ttp1)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads * threadID) << 1;
  long long int shifted_carry, temp0, temp1;
  int mask,  numbits = g_qn[0];
  double temp;

  if (j < n)
    {
      temp0 = data[threadID1] + carry_in[threadID];
      temp1 = data[threadID1 + 1];
      temp = ttp1[threadID];
      if(temp < 0.0)
      {
        numbits++;
        temp = -temp;
      }
      mask = -1 << numbits;
      shifted_carry = temp0 - (mask >> 1) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 = temp1 + (shifted_carry >> numbits);
      out[j] = temp0 * temp;
      temp *= g_ttpinc[numbits == g_qn[0]];
      out[j + 1] = temp1 * temp;
      if(report)
      {
        xint[j] = (int) temp0;
        xint[j + 1] = (int) temp1;
      }
    }
}

/**************************************************************
***************************************************************
*                                                             *
*                        Host functions                       *
*                                                             *
***************************************************************
**************************************************************/

/**************************************************************
*                                                             *
*                        Initialization                       *
*                                                             *
**************************************************************/

void init_device (int device_number, int show_prop)
{
  int device_count = 0;
  cudaGetDeviceCount (&device_count);
  if (device_number >= device_count)
  {
    printf ("device_number >=  device_count ... exiting\n");
    printf ("(This is probably a driver problem)\n\n");
    exit (2);
  }
  cudaSetDevice (device_number);
  cutilSafeCall(cudaSetDeviceFlags (cudaDeviceBlockingSync));
  cudaGetDeviceProperties (&g_dev, device_number);
  // From Iain
  if (g_dev.major == 1 && g_dev.minor < 3)
  {
    printf("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
	  printf("See http://www.mersenne.ca/cudalucas.php for a list of cards\n\n");
    exit (2);
  }
  if (show_prop)
  {
    printf ("------- DEVICE %d -------\n",    device_number);
    printf ("name                %s\n",       g_dev.name);
    printf ("Compatibility       %d.%d\n",    g_dev.major, g_dev.minor);
    printf ("clockRate (MHz)     %d\n",       g_dev.clockRate/1000);
    printf ("memClockRate (MHz)  %d\n",       g_dev.memoryClockRate/1000);
    printf ("totalGlobalMem      %llu\n",     (unsigned long long) g_dev.totalGlobalMem);
    printf ("totalConstMem       %llu\n",     (unsigned long long) g_dev.totalConstMem);
    printf ("l2CacheSize         %d\n",       g_dev.l2CacheSize);
    printf ("sharedMemPerBlock   %llu\n",     (unsigned long long) g_dev.sharedMemPerBlock);
    printf ("regsPerBlock        %d\n",       g_dev.regsPerBlock);
    printf ("warpSize            %d\n",       g_dev.warpSize);
    printf ("memPitch            %llu\n",     (unsigned long long) g_dev.memPitch);
    printf ("maxThreadsPerBlock  %d\n",       g_dev.maxThreadsPerBlock);
    printf ("maxThreadsPerMP     %d\n",       g_dev.maxThreadsPerMultiProcessor);
    printf ("multiProcessorCount %d\n",       g_dev.multiProcessorCount);
    printf ("maxThreadsDim[3]    %d,%d,%d\n", g_dev.maxThreadsDim[0], g_dev.maxThreadsDim[1], g_dev.maxThreadsDim[2]);
    printf ("maxGridSize[3]      %d,%d,%d\n", g_dev.maxGridSize[0], g_dev.maxGridSize[1], g_dev.maxGridSize[2]);
    printf ("textureAlignment    %llu\n",     (unsigned long long) g_dev.textureAlignment);
    printf ("deviceOverlap       %d\n\n",     g_dev.deviceOverlap);
  }
}

void init_threads(int n) //new requirements: 1024 | n, thr1 and thr2 the same, thr0 = 128
{
  FILE *threadf;
  char buf[192];
  char threadfile[32];
  int th0 = 0, th1 = 0;
  int temp;
  int i, j, k;

  // Priority:
  // 1. -threads th0 th1 on command line
  // 2. fft th0 th1 Values read from <gpu> threads.txt
  // 3. Threads=th0 th1 values given in ini file
  // 4. int g_thr[2] = {256,128}; default values

  // read thread values from <gpu> threads.txt only if -threads option on command line not used
  if(g_th < 0)
  {
    sprintf (threadfile, "%s threads.txt", g_dev.name);
    threadf = fopen_and_lock(threadfile, "r");
    if(threadf)
    {
      //get threads values from the last entry for the specified fft length
      while(fgets(buf, 192, threadf) != NULL)
      {
        sscanf(buf, "%d %d %d", &temp, &th0, &th1);
        if(n == temp * 1024)
        {
          g_thr[0] = th0;
          g_thr[1] = th1;
        }
      }
      unlock_and_fclose(threadf);
    }
  }

  // set k to the exponent of the largest power of 2 <= max threads, 2^k <= max threads < 2^(k+1)
  // I.e., 2^k is the largest possible thread size.
  temp = g_dev.maxThreadsPerBlock;
  k = -1;
  while(temp)
  {
    temp >>= 1;
    k++;
  }

  //set i to the exponent of the largest power of 2 which divides n, 2^i | n, but 2^(i+1) doesen't
  //set j to the exponent of the smallest power of 2 for which n <= 2^j * max grid
  temp = n; //full fft length
  i = 0; j = 0;
  while((temp & 1) == 0)
  {
    if(temp > g_dev.maxGridSize[0]) j++;
    temp >>= 1;
    i++;
  }
  if(j == i)
  {
    while(temp > g_dev.maxGridSize[0])
    {
      temp >>= 1;
      j++;
    }
  }

  // Restrictions on th[i] and n:
  // from square kernel:
  //   n <= 4 * th[0] * max grid
  //   4 * th[0] | n
  // from rcb kernel
  //   n <= 1024 * max grid
  //   1024 | n, gets checked in valid_assignment
  // 32 <= th[i] <= 2^k
  // th[i] = 2^m for some m
  // So:
  // 2^(j-2) <= th[0] <= 2^(i-2) so that n <= 4 * th[0] and 4 * th[0] | n
  // 32 <= th[i] <= 2^k
  // (j-2) <= k, fft vs device restriction
  // j <= 10 so that n <= 1024 * max grid, fft vs device restriction
  // j <= i, fft vs device restriction

  if(j < 7) j = 7; // since 32 is our smallest allowable thread size. Making j bigger doesnt affect the above property
  th0 = 1 << (i-2); // -2 because of the factor of 4 associated with th[0]
  th1 = 1 << (j-2);
  temp = 1 << k;

  if(k < j - 2 || j > 10) // square kernel needs n <= 4 * thr1 * max grid, rcb needs n <= 1024 * max grid
  {
    fprintf(stderr, "The fft %dK is too big for this device.\n", n / 1024);
    exit(2);
  }
  if (i < j) // square kernel needs n <= 4 * thr1 * max grid and 4 * thr1 | n
  {
    fprintf(stderr, "An fft of that size must be divisible by %d\n",(1 << j));
    exit(2);
  }
  if (g_thr[0] > th0)
  {
    fprintf (stderr, "fft length %d must be divisible by 4 * threads[0] = %d\n", n,  4 * g_thr[0]);
    fprintf(stderr, "Decreasing threads[0] to %d\n", th0);
    g_thr[0] = th0;
  }
  if (g_thr[0] < th1)
  {
    fprintf (stderr, "fft length / (4 * threads[0]) = %d must be less than the max block size = %d\n", n / (4 * g_thr[0]), g_dev.maxGridSize[0]);
    fprintf(stderr, "Increasing threads[0] to %d\n", th1);
    g_thr[0] = th1;
  }
  for(i = 0; i < 2; i++)
  {
    if(g_thr[i] < 32)
    {
      fprintf(stderr, "threads[%d] = %d must be at least 32, changing to %d.\n", i,  g_thr[i], 32);
      g_thr[i] = 32;
    }
    if(g_thr[i] > temp)
    {
      fprintf(stderr, "threads[%d] = %d must be no more than %d, changing to %d.\n", i,  g_thr[i], temp, temp);
      g_thr[i] = temp;
    }
    j = 1;
    while(j < g_thr[i]) j <<= 1;
    if(j != g_thr[i])
    {
      k = j - g_thr[i];
      if(k > (j >> 2)) j >>= 1;
      fprintf(stderr, "threads[%d] = %d must be a power of two, changing to %d.\n", i, g_thr[i], j);
      g_thr[i] = j;
    }
  }
  printf("Using threads: square %d, splice %d.\n", g_thr[0], g_thr[1]);
  fflush(stderr);
  return;
}

int init_ffts(int new_len)
{
  #define COUNT 162
  FILE *fft;
  char buf[192];
  char fftfile[32];
  int next_fft, j = 0, i = 0, k = 1;
  int temp_fft[256] = {0};
  int default_mult[COUNT] = {  //this batch from GTX570 timings, at least up to 16384
                                1, 2,  4,  8,  10,  14,    16,    18,    20,    32,    36,    42,
                               48,    50,    56,    60,    64,    70,    80,    84,    96,   112,
                              120,   126,   128,   144,   160,   162,   168,   180,   192,   224,
                              256,   288,   320,   324,   336,   360,   384,   392,   400,   448,
                              512,   576,   640,   648,   672,   720,   768,   784,   800,   864,
                              896,   900,  1024,  1152,  1176,  1280,  1296,  1344,  1440,  1568,
                             1600,  1728,  1792,  2048,  2160,  2304,  2352,  2592,  2688,  2880,
                             3024,  3136,  3200,  3584,  3600,  4096,  4320,  4608,  4704,  5120,
                             5184,  5600,  5760,  6048,  6144,  6272,  6400,  6480,  7168,  7200,
                             7776,  8064,  8192,  8640,  9216,  9408, 10240, 10368, 10584, 10800,
                            11200, 11520, 12096, 12288, 12544, 12960, 13824, 14336, 14400, 16384,
                            17496, 18144, 19208, 19600, 20000, 20250, 21952, 23328, 23814, 24300,
                            24500, 25088, 25600, 26244, 27000, 27216, 28000, 28672, 31104, 31250,
                            32000, 32400, 32768, 33614, 34992, 36000, 36288, 38416, 39200, 39366,
                            40500, 41472, 42336, 43200, 43904, 47628, 49000, 50000, 50176, 51200,
                            52488, 54432, 55296, 56000, 57344, 60750, 62500, 64000, 64800, 65536};


  //read fft.txt into temp array, inserting new_len appropriately
  sprintf (fftfile, "%s fft.txt", g_dev.name);
	fft = fopen_and_lock(fftfile, "r");
  if(fft)
  {
    while(fgets(buf, 192, fft) != NULL && i < 254)
    {
      if(next_fft = atoi(buf))
      {
        next_fft <<= 10;
        if((new_len > temp_fft[i]) && (new_len < next_fft))
        {
          i++;
          temp_fft[i] = new_len;
        }
        i++;
        temp_fft[i] = next_fft;
      }
    }
    unlock_and_fclose(fft);
  }
  else
  {
    i = 1;
    temp_fft[1] = new_len;
  }
  //put default values less than smallest entry in fft.txt into fft array
  while((j < COUNT) && (1024 * default_mult[j] < temp_fft[1]))
  {
    g_ffts[j] = default_mult[j] << 10;
    j++;
  }
  //put saved fft.txt values into fft array
  while(k <= i && j < 256)
  {
    g_ffts[j] = temp_fft[k];
    k++;
    j++;
  }
  k = 0;
  if (j) while(k < COUNT && default_mult[k] * 1024 <= g_ffts[j - 1]) k++;
  //put default values bigger than largest entry in fft.txt into fft array
  while(k < COUNT && j < 256)
  {
    g_ffts[j] = default_mult[k] << 10;
    j++;
    k++;
  }

  //for(i = 0; i < j; i++) printf("%d %d\n", i, g_ffts[i]/1024);
  return j;
}

int choose_fft_length (int q, int *index)
{
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */
  double ff1 = 0.999 * 1024.0 * 0.0000357822505975293;
  double ff2 = 0.99 * 1024.0 * 0.0002670641830112380;
  double e1 = 1.022179977969700;
  double e2 = 0.929905288591965;
  int lb;
  int ub;
  int i = 0;

  if(q > 0)
  {
    lb = (int) ceil (ff1 * exp (e1 * log ((double) q)));
    ub = (int) floor(ff2 * exp (e2 * log ((double) q)));
    g_lbi = 0;
    while( g_lbi < g_fft_count && g_ffts[g_lbi] < lb) g_lbi++;
    g_ubi = g_lbi;
    while( g_ubi < g_fft_count && g_ffts[g_ubi] <= ub) g_ubi++;
    g_ubi--;
  }
  //printf("Index: %d, Lower bound at %d: %dK, Upper bound at %d: %dK\n", *index, g_lbi, g_ffts[g_lbi]/1024, g_ubi, g_ffts[g_ubi]/1024);

  if(*index >= g_fft_count) while(i < g_fft_count && g_ffts[i] < *index) i++;
  else i = *index;
  if(i < g_lbi)
  {
    if(*index) printf("The fft length %dK is too small for exponent %d, increasing to %dK\n", g_ffts[i] / 1024, q, g_ffts[g_lbi] / 1024);
    i = g_lbi;
  }
  if(i > g_ubi)
  {
    printf("The fft length %dK is too large for exponent %d, decreasing to %dK\n", g_ffts[i] / 1024, q, g_ffts[g_ubi] / 1024);
    i = g_ubi;
  }
  *index = i;
  return g_ffts[i];
}

int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
  char* endptr;
  const char* ptr = str;
  int len, mult = 1;
  while( *ptr ) {
    if( *ptr == 'k' || *ptr == 'K' ) {
      mult = 1024;
      break;
    }
    if( *ptr == 'm' || *ptr == 'M' ) {
      mult = 1024*1024;
      break;
    }
    ptr++;
  }

  len = (int) strtoul(str, &endptr, 10)*mult;
  if( endptr != ptr ) { // The K or M must directly follow the num (or the num must extend to the end of the str)
    fprintf (stderr, "can't parse fft length \"%s\"\n\n", str);
    exit (2);
  }
  return len;
}

void alloc_gpu_mem(int n)
{
  //int size_d = n / 32 * 75 + 129;//73;
  //int size_i = n / 32 * 75;//35

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * n));//size_d));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * n / 4));//size_d));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp, sizeof (double) * (n / 16 + 64)));//size_d));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp2, sizeof (double) * n / 32));//size_d));
  cutilSafeCall (cudaMalloc ((void **) &g_xint, sizeof (int) * n));//size_i));
  cutilSafeCall (cudaMalloc ((void **) &g_s, sizeof (unsigned int) * n / 16));
  cutilSafeCall (cudaMalloc ((void **) &g_data, sizeof (int) * n / 64));
  cutilSafeCall (cudaMalloc ((void **) &g_carry, sizeof (int) * n / 128));
  cutilSafeCall (cudaMalloc ((void **) &g_carryl, sizeof (long long int) * n / 128));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  //g_data = &g_xint[2 * n]; // <= n/64
  //g_carry = &g_xint[n / 16 * 34]; // <= n/128
  //g_s1 =  &g_s[n / 32]; // n/32
  //g_ttmp = &g_x[n]; // n, 2 * n
  //g_ct = &g_x[2 * n]; // n/4 , 9 / 4 * n
  //g_ttp_grp = &g_x[n / 32 * 72]; // 1/32 * n, 73 / 32 * n
  //g_ttmp_grp = &g_x[n / 32 * 73]; // 1/ 32, 74 / 32 * n
  //g_ttp1 = &g_x[n / 32 * 74]; // 1/32, 75 / 32 * n
  //g_ttp_col = &g_x[n / 32 * 75]; // 64, 75 / 32 * n + 32
  //g_ttmp_col = &g_x[n / 32 * 75 + 64];
  //g_err = (float *) &g_x[n / 32 * 75 + 128];
}

void write_gpu_data(int q, int n)
{
  double *s_ttp1 = NULL, *s_ct = NULL, *s_ttmp_col = NULL, *s_ttp_col = NULL, *s_ttmp_grp = NULL, *s_ttp_grp = NULL, *s_grp;
  unsigned int *s_size1 = NULL, *s_size2;
  double d = 2.0 / (double) n;
  double *h_ttpinc = NULL;
  int *h_qn = NULL;
  int j, i;
  int b = q % n;

  s_ct = (double *) malloc (sizeof (double) * n / 4);
  s_ttmp_col = (double *) malloc (sizeof (double) * 32);
  s_ttp_col = (double *) malloc (sizeof (double) * 32);
  s_ttmp_grp = (double *) malloc (sizeof (double) * n / 32);
  s_ttp_grp = (double *) malloc (sizeof (double) * n / 32);
  s_grp = (double *) malloc (sizeof (double) * (n / 16 + 64));
  g_size1 = (unsigned int *) malloc (sizeof (unsigned int) * n / 32);
  s_size1 = (unsigned int *) malloc (sizeof (unsigned int) * n / 32);
  s_size2 = (unsigned int *) malloc (sizeof (unsigned int) * n / 16);
  s_ttp1 = (double *) malloc (sizeof (double) * n / 32);
	h_ttpinc = (double*) malloc(2 * sizeof(double));
	h_qn = (int*) malloc(1 * sizeof(int));

  // Square kernel data
  for (j = (n >> 2) - 1; j > 0; j--) s_ct[j] = 0.5 * cospi (j * d);
  cudaMemcpy (g_ct, s_ct, sizeof (double) * (n / 4), cudaMemcpyHostToDevice);

  d *= 0.5;

  //ttp, ttmp column data
  i = 0;
  for(j = 0; j < 32; j++)
  {
    s_ttp_col[j] = exp2(i * d);
    s_ttmp_col[j] = exp2(-i * d) * 2.0 * d;
    if(j & 1) s_ttmp_col[j] = -s_ttmp_col[j];
    i -= b;
    if(i < 0) i += n;
  }

  //ttp, ttmp group data
  i = 0;
  for(j = 0; j < n / 32; j++)
  {
    s_ttp_grp[j] = exp2(i * d);
    s_ttmp_grp[j] = exp2(-i * d);
    i -= b * 32;
    while(i < 0) i += n;
  }

  for(j = 0; j < 32; j++)
  {
    s_grp[j * 2] = s_ttp_col[j];
    s_grp[j * 2 + 1] = s_ttmp_col[j];
  }
  for(j = 0; j < n / 32; j++)
  {
    s_grp[j * 2 + 64] = s_ttp_grp[j];
    s_grp[j * 2 + 65] = s_ttmp_grp[j];
  }
  cudaMemcpy (g_ttp, s_grp, sizeof (double) * (n / 16 + 64), cudaMemcpyHostToDevice);

  // Constant memory data
  h_ttpinc[0] = exp2((n - b) * d);
  h_ttpinc[1] = exp2(-b * d);
  h_qn[0] = q / n;
  set_ttpinc(h_ttpinc);
  set_qn(h_qn);

  // biglit data
  double comp1 = exp2(b * d) - 0.000000001;
  double comp2 = 2.0 * comp1;
  for(j = 0; j < n / 32; j++)
  {
    g_size1[j] = 0;
    s_size1[j] = 0;
    for(i = 31; i >= 0; i--)
    {
      d = s_ttp_col[i] * s_ttp_grp[j];
      g_size1[j] <<= 1;
      s_size1[j] <<= 1;
      if(d >= 2.0)
      {
        s_size1[j]++;
        if(d < comp2) g_size1[j]++;
      }
      else if(d < comp1) g_size1[j]++;
    }
  }

  for(j = 0; j < n / 32; j++)
  {
    s_size2[j * 2] = g_size1[j];
    s_size2[j * 2 + 1] = s_size1[j];
  }
  cudaMemcpy (g_s, s_size2, sizeof (unsigned int) * n / 16, cudaMemcpyHostToDevice);

  // ttp and size data for splice kernel
  for(i = 0, j = 0; i < n / 32; i += 2, j++)
  {
    s_ttp1[j] = s_ttp_grp[i];
    if(g_size1[i] & 1) s_ttp1[j] = -s_ttp1[j];
  }
  for(i = 0; i < j; i += 2, j++) s_ttp1[j] = s_ttp1[i];
  cudaMemcpy (g_ttp2, s_ttp1, sizeof (double) * n / 32, cudaMemcpyHostToDevice);

  if(n >= B3) i = 0;
  else if(n >= B2) i = 1;
  else if(n >= B1) i = 2;
  else i = 3;
  j = 512 >> i;
  i = n * ((j >> 5) - 1) / j;
  g_ttp1 = &g_ttp2[i];

  cufftSafeCall (cufftPlan1d (&g_plan, n / 2, CUFFT_Z2Z, 1));

  free ((char *) s_ct);
  free ((char *) s_size1);
  free ((char *) s_size2);
  free ((char *) s_ttmp_col);
  free ((char *) s_ttp_col);
  free ((char *) s_ttmp_grp);
  free ((char *) s_ttp_grp);
  free ((char *) s_grp);
  free ((char *) s_ttp1);
  free ((char *) h_ttpinc);
  free ((char *) h_qn);
}

void init_x(int *x_int, unsigned *x_packed, int q, int n, int *offset)
{
  int j;
  int digit, bit;
  int end = (q + 31) / 32;
  if(*offset < 0)
  {
    srand(time(0));
    *offset = rand() % q;
    bit = (*offset + 2) % q;
    digit = floor(bit * (n / (double) q));
    bit = bit - ceil(digit * (q / (double) n));
    for(j = 0; j <  n; j++) x_int[j] = 0;
    x_int[digit] = (1 << bit);
    if(x_packed)
    {
      for(j = 0; j < end; j++) x_packed[j] = 0;
      x_packed[*offset / 32] = (1 << (*offset % 32));
      x_packed[end] = q;
      x_packed[end + 1] = n;
    }
  }
  cudaMemcpy (g_xint, x_int, sizeof (int) * n , cudaMemcpyHostToDevice);
  apply_weights <<<n /1024,128 >>> (g_x, g_xint, g_ttp, g_s);
  }

int size(int pos)
{
  return (int) (g_size1[pos >> 5] & (1 << (pos & 31))) / (1 << (pos & 31));
}

void unpack_bits(int *x_int, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;
  int mask1 = -1 << (qn + 1);
  int mask2;
  int mask;

  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  for(i = 0; i < n; i++)
  {
    if(k < qn + size(i))
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(size(i)) mask = mask1;
    else mask = mask2;
    x_int[i] = ((int) temp2) & mask;
    temp2 >>= (qn + size(i));
    k -= (qn + size(i));
  }
}

void balance_digits(int* x, int q, int n)
{
  int half_low = (1 << (q / n - 1));
  int low = half_low << 1;
  int high = low << 1;
  int upper, adj, carry = 0;
  int j;

  for(j = 0; j < n; j++)
  {
    if(size(j))
    {
      upper = low;
      adj = high;
    }
    else
    {
      upper = half_low;
      adj = low;
    }
    x[j] += carry;
    carry = 0;
    if(x[j] >= upper)
    {
      x[j] -= adj;
      carry = 1;
    }
  }
  x[0] += carry; // Good enough for our purposes.
}

int *init_lucas(unsigned *x_packed,
                           int       q,
                           int       *n,
                           int       *j,
                           int       *offset,
                           unsigned long long *total_time,
                           unsigned long long  *time_adj,
                           unsigned  *iter_adj)
{
  int *x_int;
  int end = (q + 31) / 32;
  int new_test = 0;

  *n = x_packed[end + 1];
  if(*n == 0) new_test = 1;
  *j = x_packed[end + 2];
  *offset = x_packed[end + 3];
  if(total_time) *total_time = (unsigned long long) x_packed[end + 4] << 15;
  if(time_adj) *time_adj = (unsigned long long) x_packed[end + 5] << 15;
  if(iter_adj) *iter_adj = x_packed[end + 6];
  if(g_fftlen == 0) g_fftlen = *n;
  if(g_fft_count == 0) g_fft_count = init_ffts(g_fftlen);
  *n = choose_fft_length(q, &g_fftlen);
  if(*n != (int) x_packed[end + 1])
  {
    x_packed[end + 1] = *n;
    if(time_adj) *time_adj = *total_time;
    if(*j > 1 && iter_adj) *iter_adj = *j;
  }
  //printf("time_adj: %llu, iter_adj: %u\n", *time_adj, *iter_adj);
  init_threads(*n);

  x_int = (int *) malloc (sizeof (int) * 2 * *n);
  alloc_gpu_mem(*n);
  write_gpu_data(q, *n);
  if(!new_test)
  {
    unpack_bits(x_int, x_packed, q, *n);
    balance_digits(x_int, q, *n);
  }
  init_x(x_int, x_packed, q, *n, offset);
  return x_int;
}

/**************************************************************************
 *                                                                        *
 *                               Cleanup                                  *
 *                                                                        *
 **************************************************************************/

void free_host (int *x)
{
  free ((char *) g_size1);
  if(x) free ((char *) x);
}

void free_gpu(int dp)
{
  if(dp) cufftSafeCall (cufftDestroy (g_plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_ttp2));
  cutilSafeCall (cudaFree ((char *) g_ttp));
  cutilSafeCall (cudaFree ((char *) g_s));
  cutilSafeCall (cudaFree ((char *) g_xint));
  cutilSafeCall (cudaFree ((char *) g_data));
  cutilSafeCall (cudaFree ((char *) g_carry));
  cutilSafeCall (cudaFree ((char *) g_carryl));
  cutilSafeCall (cudaFree ((char *) g_err));
}

void close_lucas (int *x)
{
  free_host(x);
  free_gpu(1);
}

/**************************************************************************
 *                                                                        *
 *                        Checkpoint Processing                           *
 *                                                                        *
 **************************************************************************/


//From apsen
void print_time_from_seconds (unsigned in_sec, char *res, int mode)
{
  char s_time[32] = {0};
  unsigned day = 0;
  unsigned hour;
  unsigned min;
  unsigned sec;

  if(mode >= 2)
  {
    day = in_sec / 86400;
    in_sec %= 86400;
  }
  hour = in_sec / 3600;
  in_sec %= 3600;
  min = in_sec / 60;
  sec = in_sec % 60;
  if (day)
  {
    if(mode & 1) sprintf (s_time, "%ud%02uh%02um%02us", day, hour, min, sec);
    else sprintf (s_time, "%u:%02u:%02u:%02u", day, hour, min, sec);
  }
  else if (hour)
  {
     if(mode & 1) sprintf (s_time, "%uh%02um%02us", hour, min, sec);
     else sprintf (s_time, "%u:%02u:%02u", hour, min, sec);
  }
  else
  {
     if(mode & 1) sprintf (s_time, "%um%02us", min, sec);
     else sprintf (s_time, "%u:%02u", min, sec);
  }
  if(res) sprintf(res, "%12s", s_time);
  else printf ("%s", s_time);
}

int standardize_digits(int *x_int, int q, int n)
{
  int j, qn = q / n, carry = 0;
  int temp;
  int lo = 1 << qn;
  int hi = lo << 1;

  j = n - 1;
  while(x_int[j] == 0 && j) j--;
  if(j == 0 && x_int[0] == 0) return(1);
  else if (x_int[j] < 0) carry = -1;
  for(j = 0; j < n; j++)
  {
      x_int[j] += carry;
      if (size(j)) temp = hi;
      else temp = lo;
      if(x_int[j] < 0)
      {
        x_int[j] += temp;
        carry = -1;
      }
      else carry = 0;
  }
  return(0);
}

unsigned long long find_residue(int *x_int, int q, int n, int offset)
{
  int j, k = 0;
  int digit, bit;
  unsigned long long residue = 0;
  int qn = q / n, carry = 0;
  int lo = 1 << qn;
  int hi = lo << 1;
  int tx, temp;

  digit = floor(offset * (n / (double) q));
  bit = offset - ceil(digit * (q / (double) n));
  j = (n + digit - 1) % n;
  while(x_int[j] == 0 && j != digit)
  {
     j--;
     if(j < 0) j += n;
  }
  if(j == digit && x_int[digit] == 0) return(0);
  else if (x_int[j] < 0) carry = -1;
  for(j = 0; j < 10; j++)
  {
    tx = x_int[digit] + carry;
    if (size(digit)) temp = hi;
    else temp = lo;
    if(tx < 0)
    {
      tx += temp;
      carry = -1;
    }
    else carry = 0;
    residue += (unsigned long long) tx << k;
    k += q / n + size(digit);
    if(j == 0)
    {
       k -= bit;
       residue >>= bit;
    }
    if(k >= 64) break;
    digit++;
    if(digit == n) digit = 0;
  }
  return residue;
}

void printbits(unsigned long long residue, int q, int n, int offset, FILE* fp, int o_f)
{
  if (fp)
  {
    printf ("M( %d )C, 0x%016llx,", q, residue);
    fprintf (fp, "M( %d )C, 0x%016llx,", q, residue);
    if(o_f) fprintf(fp, " offset = %d,", offset);
    fprintf (fp, " n = %dK, %s", n/1024, program);
  }
  else printf ("/ %d, 0x%016llx,", q, residue);
  printf (" %dK, %s", n/1024, program);
}

void pack_bits(int *x_int, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;

  for(i = 0; i < n; i++)
  {
    temp1 = x_int[i];
    temp2 += (temp1 << k);
    k += qn + size(i);
    if(k >= 32)
    {
      packed_x[j] = (unsigned) temp2;
      temp2 >>= 32;
      k -= 32;
      j++;
    }
  }
  packed_x[j] = (unsigned) temp2;
}

void set_checkpoint_data(unsigned *x_packed, int q, int j, int offset, unsigned long long tt, unsigned long long ta, unsigned ia)
{
  int end = (q + 31) / 32;

  x_packed[end + 2] = j;
  x_packed[end + 3] = offset;
  x_packed[end + 4] = tt >> 15;
  x_packed[end + 5] = ta >> 15;
  x_packed[end + 6] = ia;
}

void reset_err(float* maxerr, float value)
{
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  *maxerr *= value;
}

void process_output(int q,
                    int n,
                    int j,
                    int offset,
                    int last_chk,
                    float maxerr,
                    unsigned long long residue,
                    unsigned long long diff,
                    unsigned long long diff1,
                    unsigned long long diff2,
                    unsigned long long total_time)
{
  time_t now;
  struct tm *tm_now = NULL;
  int index = 0;
  char buffer[192];
  char temp [32];
  int i;
  static int header_iter = 0;

  now = time(NULL);
  tm_now = localtime(&now);
  strftime(g_output[0], 32, "%a", tm_now);                             //day of the week Sat
  strftime(g_output[1], 32, "%b", tm_now);                             //month
  strftime(g_output[2], 32, "%d", tm_now);                             //date
  strftime(g_output[3], 32, "%H", tm_now);                             //hour
  strftime(g_output[4], 32, ":%M", tm_now);                            //min
  strftime(g_output[5], 32, ":%S", tm_now);                            //sec
  strftime(g_output[6], 32, "%y", tm_now);                             //year
  strftime(g_output[7], 32, "%Z", tm_now);                             //time zone
  sprintf(g_output[8],  "%9d", q);                                      //exponent
  sprintf(temp,  "M%d", q);
  sprintf(g_output[9],  "%10s", temp);                                 //Mersenne number
  sprintf(g_output[10], "%5dK", n / 1024);                             //fft, multiple of 1k
  sprintf(g_output[11], "%8d", n);                                     //fft
  sprintf(g_output[12], "%9d", j);                                     //iteration
  sprintf(g_output[13], "%9d", offset);                                //offset
  sprintf(g_output[14], "0x%016llx", residue);                         //residue, lower case hex
  sprintf(g_output[15], "0x%016llX", residue);                         //residue, upper case hex
  sprintf(g_output[18], "%s", program);                                //program name
  sprintf(g_output[19], "%%");                                         //percent done
  sprintf(g_output[25], "%7.5f", maxerr);                              //round off error
  sprintf(g_output[26], "%9.5f", diff1 / 1000.0 / (j - last_chk));     //ms/iter
  sprintf(g_output[27], "%10.5f", diff1 / 1000000.0 );                 //time since last checkpoint xxx.xxxxx seconds format
  sprintf(g_output[28], "%10.5f", (j - last_chk) * 1000000.0 / diff1); //iter/sec
  sprintf(g_output[29], "%10.5f", j / (float) (q - 2) * 100 );         //percent done
  print_time_from_seconds((unsigned) diff, g_output[16], 0);                      //time since last checkpoint hh:mm:ss format
  print_time_from_seconds((unsigned) diff2, g_output[17], 2);                     //ETA in d:hh:mm:ss format

  i = 0;
  while( i < 50 && g_output_code[i] >= 0)
  {
    index += sprintf(buffer + index,"%s",g_output[g_output_code[i]]);
    if(g_output_code[i] >= 25 && g_output_code[i] < 30)
    {
      i++;
      index -= g_output_code[i];
    }
    i++;
  }
  if(header_iter == 0 && g_output_interval > 0)
  {
    header_iter = g_output_interval - 1;
    printf("%s\n", g_output_header);
  }
  header_iter--;
  printf("%s\n", buffer);
 }

/**************************************************************************
 *                                                                        *
 *                        File Related Functions                          *
 *                                                                        *
 **************************************************************************/
unsigned magic_number(unsigned *x_packed, int q)
{
  return 0;
}

unsigned int checkpoint_checksum(char *string, int chars)
/* generates a CRC-32 like checksum of the string */
{
  unsigned int chksum=0;
  int i,j;

  for(i=0;i<chars;i++)
  {
    for(j=7;j>=0;j--)
    {
      if((chksum>>31) == (((unsigned int)(string[i]>>j))&1))
      {
        chksum<<=1;
      }
      else
      {
        chksum = (chksum<<1)^0x04C11DB7;
      }
    }
  }
  return chksum;
}

unsigned ch_sum(unsigned *x_packed, int q)
{
  int end = (q + 31) / 32;
  int j;
  unsigned sum = 0;

  for(j = 0; j < end + 9; j++) sum += x_packed[j];
  return sum;
}


unsigned *read_checkpoint(int q)
{
  FILE *fPtr;
  unsigned *x_packed;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;
  int i;

  x_packed = (unsigned *) malloc (sizeof (unsigned) * (end + 10));
  for(i = 0; i < 10; i++) x_packed[end + i] = 0;
  x_packed[end + 2] = 1;
  x_packed[end + 3] = (unsigned) -1;
  if(0 <= g_rt) return(x_packed);

  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (fPtr)
  {
    if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
    {
      fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
      fclose (fPtr);
    }
    else
    {
      fclose(fPtr);
      if(x_packed[end + 9] != checkpoint_checksum((char*) x_packed, 4 * (end + 9)))
      //ch_sum(x_packed, q))
      fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
      else return x_packed;
    }
  }
  fPtr = fopen(chkpnt_tfn, "rb");
  if (fPtr)
  {
    if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
    {
      fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
      fclose (fPtr);
    }
    else
    {
      fclose(fPtr);
      if(x_packed[end + 9] != ch_sum(x_packed, q)) fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
      else return x_packed;
    }
  }
  return x_packed;
}

void write_checkpoint(unsigned *x_packed, int q, unsigned long long residue)
{
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;

  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr)
  {
    fprintf(stderr, "Couldn't write checkpoint.\n");
    return;
  }
  x_packed[end + 8] = magic_number(x_packed, q);
  x_packed[end + 9] = checkpoint_checksum((char*) x_packed, 4 * (end + 9));
  fwrite (x_packed, 1, sizeof (unsigned) * (end + 10), fPtr);
  fclose (fPtr);
  if (g_sf > 0)			// save all checkpoint files
  {
    char chkpnt_sfn[64];
    char test[64];
#ifndef _MSC_VER
    sprintf (chkpnt_sfn, "%s/s" "%d.%d.%016llx", g_folder, q, x_packed[end + 2] - 1, residue);
    sprintf (test, "%s/%s", g_folder, ".empty.txt");
#else
    sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%016llx.cls", g_folder, q, x_packed[end + 2] - 1, residue);
    sprintf (test, "%s\\%s", g_folder, ".empty.txt");
#endif
    fPtr = NULL;
    fPtr = fopen (test, "r");
    if(!fPtr)
    {
#ifndef _MSC_VER
      mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
      if (mkdir (g_folder, mode) != 0) fprintf (stderr, "mkdir: cannot create directory `%s': File exists\n", g_folder);
#else
      if (_mkdir (g_folder) != 0) fprintf (stderr, "mkdir: cannot create directory `%s': File exists\n", g_folder);
#endif
      fPtr = fopen(test, "w");
      if(fPtr) fclose(fPtr);
    }
    else fclose(fPtr);

    fPtr = fopen (chkpnt_sfn, "wb");
    if (!fPtr) return;
    fwrite (x_packed, 1, sizeof (unsigned) * (((q + 31) / 32) + 10), fPtr);
    fclose (fPtr);
  }
}

void
rm_checkpoint (int q)
{
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  (void) unlink (chkpnt_cfn);
  (void) unlink (chkpnt_tfn);
}

/**************************************************************************
 *                                                                        *
 *                        LL Test Iteration Loop                          *
 *                                                                        *
 **************************************************************************/

float lucas_square (int q, int n, int iter, int last, int* offset, float* maxerr, int error_flag)
{
  int digit, bit, sc = 10, tf = 1024, i;
  float terr = 0.0f;
  cudaError err = cudaSuccess;

  if(n >= B3) i = 0;
  else if(n >= B2) i = 1;
  else if(n >= B1) i = 2;
  else i = 3;
  sc -= i;
  tf >>= i;

  *offset = (2 * *offset) % q;
  bit = (*offset + 1) % q;
  digit = floor(bit * (n / (double) q));
  bit = bit - ceil(digit * (q / (double) n));
  bit = 1 << bit;

  cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
  square <<< n / (4 * g_thr[0]), g_thr[0] >>> (n, g_x, g_ct);
  cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
  if(g_bc == 0)
  {
    if(sc == 10)
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcb <10, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
        else rcb <10, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
      }
      else rcb <10, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
    }
    else if(sc == 9)
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcb <9, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
        else rcb <9, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
      }
      else rcb <9, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
    }
    else if(sc == 8)
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcb <8, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
        else rcb <8, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
      }
      else rcb <8, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
    }
    else
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcb <7, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
        else rcb <7, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
      }
      else rcb <7, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carry, g_err, *maxerr, digit, bit);
    }
    if(error_flag & 1) splice <1> <<< (n / tf + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, g_xint, n, tf / 2, g_data, g_carry, g_ttp1);
    else splice <0> <<< (n / tf + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, g_xint, n, tf / 2, g_data, g_carry, g_ttp1);
  }
  else
  {
    if(sc == 10)
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcbl <10, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
        else rcbl <10, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
      }
      else rcbl <10, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
    }
    else if(sc == 9)
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcbl <9, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
        else rcbl <9, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
      }
      else rcbl <9, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
    }
    else if(sc == 8)
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcbl <8, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
        else rcbl <8, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
      }
      else rcbl <8, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
    }
    else
    {
      if(iter % g_ei == 0 || error_flag & 1)
      {
        if(error_flag & 1) rcbl <7, 1, 1> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
        else rcbl <7, 1, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
      }
      else rcbl <7, 0, 0> <<< n / tf, TT >>> (g_x, g_xint, g_data, g_ttp, g_s, g_carryl, g_err, *maxerr, digit, bit);
    }
    if(error_flag & 1) splicel <1> <<< (n / tf + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, g_xint, n, tf / 2, g_data, g_carryl, g_ttp1);
    else splicel <0> <<< (n / tf + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, g_xint, n, tf / 2, g_data, g_carryl, g_ttp1);
  }
  if (error_flag & 3)
  {
    err = cutilSafeCall1 (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
    //if( g_pf && g_sl) usleep(g_sv);//, nanosleep sleep(1);
  }
  else if (g_pf && (iter % g_po) == 0)
  {
    err = cutilSafeThreadSync();
    //if(g_sl) usleep(g_sv);//, nanosleep sleep(1);
  }
  if(err != cudaSuccess) terr = -1.0f;
  return (terr);
}

/**************************************************************************
 *                                                                        *
 *                        Benchmarking and Testing                        *
 *                                                                        *
 **************************************************************************/
int isReasonable(int fft)
{ //From an idea of AXN's mentioned on the forums
  int i;

  while(!(fft & 1)) fft >>= 1;
  for(i = 3; i <= 7; i += 2) while((fft % i) == 0) fft /= i;
  return (fft);
}

void kernel_percentages (int fft, int passes, int mode)
{
  int pass, sync, i, j;
  int n = fft * 1024;
  float total0[4] = {0.0f};
  float total1[4] = {0.0f};
  float maxerr = 0.0f, outerTime, t0, t1;
  cudaEvent_t start, stop;

  alloc_gpu_mem(n);
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
  cufftSafeCall (cufftPlan1d (&g_plan, n / 2, CUFFT_Z2Z, 1));

  if(passes == 0)
  {
    sync = mode & 1;
    if(sync) mode--;
    i = 1;
    time_t now;
    struct tm *tm_now = NULL;
    char ost[32];
    if(sync) cutilSafeCall (cudaEventRecord (start, 0));
    while(1)
    {
      cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
      if(i % mode == 0)
      {
        now = time(NULL);
        tm_now = localtime(&now);
        strftime(ost, 32, "%D ", tm_now);                             //day of the week Sat
        printf("%s",ost);
        strftime(ost, 32, "%T ", tm_now);                             //day of the week Sat
        printf("%s",ost);
        if(sync)
        {
          cutilSafeCall (cudaEventRecord (stop, 0));
          cutilSafeCall (cudaEventSynchronize (stop));
          cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
          outerTime /= (float) i;
        }
        printf("Iteration %d", i);
        if(sync) printf(", Average iteration time =  %0.5f, Total time = %0.1f\n", outerTime, outerTime * i);
        else printf("\n");
      }
      else if(i % 100 && sync) cutilSafeThreadSync();
      i++;
    }
  }

  init_threads(n);
  for(j = 0; j < 4; j++)
  {
    for(pass = 0; pass < passes; pass++)
    {
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 50; i++)
      {
        if(j == 0)
        {
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
        }
        if(j == 1)
        {
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / TF + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, NULL, n, TF / 2, g_data, g_carry, g_ttp1);
        }
        if(j == 2)
        {
          square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / TF + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, NULL, n, TF / 2, g_data, g_carry, g_ttp1);
        }
        if(j == 3)
        {
          cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          square <<< n / (4 * g_thr[0]), g_thr[0] >>> (n, g_x, g_ct);
          cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / TF + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, NULL, n, TF / 2, g_data, g_carry, g_ttp1);
        }
      }
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      outerTime /= 50.0f;
      total0[j] += outerTime;
    }
  }
  for(j = 0; j < 4; j++)
  {
    for(pass = 0; pass < passes; pass++)
    {
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 50; i++)
      {
        if(j == 0)
        {
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
        }
        if(j == 1)
        {
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / TF + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, NULL, n, TF / 2, g_data, g_carry, g_ttp1);
        }
        if(j == 2)
        {
          square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / TF + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, NULL, n, TF / 2, g_data, g_carry, g_ttp1);
        }
        if(j == 3)
        {
          cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          square <<< n / (4 * g_thr[0]), g_thr[0] >>> (n, g_x, g_ct);
          cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          rcb <10, 0, 0> <<< n / TF, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / TF + g_thr[1] - 1) / g_thr[1], g_thr[1] >>> (g_x, NULL, n, TF / 2, g_data, g_carry, g_ttp1);
        }
      }
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      outerTime /= 50.0f;
      total1[j] += outerTime;
    }
  }
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));
  free_gpu(1);

  t0 = total0[3];
  t1 = total1[3];
  printf("Total time: %11.5f, %8.5f, %6.2f\n", t0/passes, t1/passes, 100.0);
  printf("RCB Kernel: %11.5f, %8.5f, %6.2f\n", total0[0]/passes, total1[0]/passes, total0[0] / t0 * 100.0);
  printf("Splice Kernel: %8.5f, %8.5f, %6.2f\n", (total0[1] - total0[0])/passes, (total1[1]-total1[0])/passes, (total0[1] - total0[0]) / t0 * 100.0);
  printf("Mult Kernel: %10.5f, %8.5f, %6.2f\n", (total0[2] - total0[1])/passes, (total1[2] - total1[1])/passes, (total0[2] - total0[1]) / t0 * 100.0);
  printf("FFTs: %17.5f, %8.5f, %6.2f\n", (total0[3] - total0[2])/passes, (total1[3] - total1[2])/passes, (total0[3] - total0[2]) / t0 * 100.0);
}



void threadbench (int st_fft, int end_fft, int passes, int mode, int device_number)
{
  float outerTime, maxerr = 0.5f;
  float total[10] = {0.0f};  // if maxThreadsPerBlock ever exceeds 8192,
  int th[10] = {0};          // these limits will need to be increased from 10.
  int maxi = 0;
  float best_time = 0.0f;
  int results[512][3];
  float time[512] = {0.0f};
  int pass;
  int n = end_fft << 10;
  int j = st_fft << 10;
  int fft;
  int b_t[2] = {0,0};
  int s[2];
  int e[2];
  int i, k, t, t0 = 0, t1 = 2;
  cudaEvent_t start, stop;

  if(st_fft == 0)
  {
    kernel_percentages (end_fft, passes, mode);
    return;
  }

  if(st_fft == end_fft) printf("Thread bench, testing various thread sizes for fft %dK, doing %d passes.\n", end_fft, passes);
  else printf("Thread bench, testing various thread sizes for ffts %dK to %dK, doing %d passes.\n", st_fft, end_fft, passes);
  fflush(NULL);

  alloc_gpu_mem(n);
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));

  th[0] = 32;
  for(i = 1; i < 10; i++)
  {
    th[i] = th[i-1] << 1;
    if(th[i] > g_dev.maxThreadsPerBlock)
    {
      maxi = i;
      break;
    }
  }
  s[0] = s[1] = 0;
  e[0] = e[1] = maxi;
  if(mode & 2) s[0]++;
  if(mode & 4) e[0]--;

  for(i = 0; i < 512; i++) results[i][0] = 0;
  g_fft_count = init_ffts(j);
  fft = choose_fft_length(0, &j);

  while(fft <= n)
  {
    if(isReasonable(fft) <= 1 && fft / 1024 <= g_dev.maxGridSize[0] && fft % 1024 == 0)
    {
      cufftSafeCall (cufftPlan1d (&g_plan, fft / 2, CUFFT_Z2Z, 1));
      for(k = 0; k < 2; k++)
      {
        for (t = s[k]; t < e[k]; t++)
        {
          if(k == 1 || (fft / (4 * th[t]) <= g_dev.maxGridSize[0] && fft % (4 * th[t]) == 0))
          {
            if(k == 0)
            {
              t0 = t;
              t1 = 2;
            }
            else t1 = t;
            for(pass = 1; pass <= passes; pass++)
            {
              cutilSafeCall (cudaEventRecord (start, 0));
              for (i = 0; i < 50; i++)
              {
                cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
                square <<< fft / (4 * th[t0]), th[t0] >>> (fft, g_x, g_ct);
                cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
                rcb <10, 0, 0><<< fft / 1024, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
                splice <0> <<< (fft / 1024 + th[t1] - 1) / th[t1], th[t1] >>> (g_x, NULL, fft, 512, g_data, g_carry, g_ttp2);
              }
              cutilSafeCall (cudaEventRecord (stop, 0));
              cutilSafeCall (cudaEventSynchronize (stop));
              cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
              outerTime /= 50.0f;
              total[t] += outerTime;
            }
            if(!(mode & 8)) printf ("fft = %dK, ave time = %7.4f ms, square: %4d, splice: %4d\n", fft / 1024 , total[t] / passes, th[t0], th[t1]);
            fflush(NULL);
          }
        }
        best_time = 1000000.0f;
        for (i = 0; i < maxi; i++)
        {
          if(total[i] < best_time && total[i] > 0.0f)
          {
            best_time = total[i];
            b_t[k] = i;
          }
        }
        t0 = b_t[0];
        for(i = 0; i < maxi; i++) total[i] = 0.0f;
      }
      cufftSafeCall (cufftDestroy (g_plan));
      printf("fft = %dK, min time = %7.4f ms, square: %4d, splice: %4d\n", fft / 1024, best_time / passes, th[b_t[0]], th[b_t[1]]);
      if(!(mode & 8)) printf("\n");
      results[j][0] = fft / 1024;
      results[j][1] = th[b_t[0]];
      results[j][2] = th[b_t[1]];
      time[j] = best_time / passes;
      j++;
    }
    else if (mode & 1) j++;
    if((mode & 1) && j < g_fft_count) fft = g_ffts[j];
    else if (!(mode & 1)) fft += 1024;
    else break;
  }
  free_gpu(0);
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));


  char threadfile[32];
  FILE *fptr;

  sprintf (threadfile, "%s threads.txt", g_dev.name);
	fptr = fopen_and_lock(threadfile, "a+");
  if(fptr)
  {
    for(j = 0; j < 512; j++)
      if(results[j][0]) fprintf(fptr, "%5d %4d %4d %10.5f\n", results[j][0], results[j][1], results[j][2], time[j]);
    unlock_and_fclose(fptr);
  }
}


int isprime(unsigned int n)
{
  unsigned int i;

  if(n<=1) return 0;
  if(n>2 && n%2==0)return 0;

  i=3;
  while(i*i <= n && i < 0x10000)
  {
    if(n%i==0)return 0;
    i+=2;
  }
  return 1;
}

void memtest(int s, int iter, int device)
{
  int i, j, k, m;
  int q = 60000091;
  int n = 3200 * 1024;
  int rand_int;
  int *i_data;
  double *d_data;
  double *dev_data1;
  int *d_compare;
  int h_compare;
  int total = 0;
  int total_iterations;
  int iterations_done = 0;
  float percent_done = 0.0f;
  timeval time0, time1;
  unsigned long long diff;
  unsigned long long diff1;
  unsigned long long diff2;
  unsigned long long ttime = 0;
  double total_bytes;
  size_t global_mem, free_mem;

  cudaMemGetInfo(&free_mem, &global_mem);
  printf("CUDA reports %lluM of %lluM GPU memory free.\n", (unsigned long long)free_mem/1024/1024, (unsigned long long)global_mem/1024/1024);
  if((size_t) s *1024 * 1024 * 25  > free_mem )
  {
    s = free_mem / 1024 / 1024 / 25;
     printf("Reducing size to %d\n", s);
  }
  printf("\nInitializing memory test using %0.0fMB of memory on device %d...\n", n / 1024.0 * s / 1024.0 * 8.0, device);

  i_data = (int *) malloc (sizeof (int) * n);
  d_data = (double *) malloc (sizeof (double) * n * 5);

  alloc_gpu_mem(n);
  write_gpu_data(q, n);

  srand(time(0));
  for (j = 0; j < n; j++)    i_data[j] = 1;
  cudaMemcpy (g_xint, i_data, sizeof (int) * n, cudaMemcpyHostToDevice);
  apply_weights <<<n /1024,128 >>> (g_x, g_xint, g_ttp, g_s);
  cudaMemcpy (&d_data[0 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);

  for (j = 0; j < n; j++)
  {
    rand_int = rand() % (1 << 18);
    rand_int -= (1 << 17);
    i_data[j] = rand_int;
  }
  cudaMemcpy (g_xint, i_data, sizeof (int) * n, cudaMemcpyHostToDevice);
  apply_weights <<<n /1024,128 >>> (g_x, g_xint, g_ttp, g_s);
  cudaMemcpy (&d_data[1 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);

  cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
  cudaMemcpy (&d_data[2 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);

  square <<< n / (4 * g_thr[0]), g_thr[0] >>> (n, g_x, g_ct);
  cudaMemcpy (&d_data[3 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);

  cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
  cudaMemcpy (&d_data[4 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);

  free(i_data);
  free(g_size1);
  free_gpu(1);

  cutilSafeCall (cudaMalloc ((void **) &d_compare, sizeof (int)));
  cutilSafeCall (cudaMemset (d_compare, 0, sizeof (int)));
  cutilSafeCall (cudaMalloc ((void **) &dev_data1, sizeof (double) * n * s));

  total_iterations = s * 5 * iter;
  iter *= 10000;
  printf("Beginning test.\n\n");
  fflush(NULL);
  gettimeofday (&time0, NULL);
  for(j = 0; j < s; j++)
  {
    m = (j + 1) % s;
    for(i = 0; i < 5; i++)
    {
      cutilSafeCall (cudaMemcpy (&dev_data1[j * n], &d_data[i * n], sizeof (double) * n, cudaMemcpyHostToDevice));
      for(k = 1; k <= iter; k++)
      {
        copy_kernel <<<n / 512, 512 >>> (dev_data1, n, j, m);
        compare_kernel<<<n / 512, 512>>> (&dev_data1[m * n], &dev_data1[j * n], d_compare);
        if(k%100 == 0) cutilSafeThreadSync();
        if(k%10000 == 0)
        {
          cutilSafeCall (cudaMemcpy (&h_compare, d_compare, sizeof (int), cudaMemcpyDeviceToHost));
          cutilSafeCall (cudaMemset (d_compare, 0, sizeof (int)));
          total += h_compare;
          iterations_done++;
          percent_done = iterations_done * 100 / (float) total_iterations;
          gettimeofday (&time1, NULL);
          diff = time1.tv_sec - time0.tv_sec;
          diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
          time0.tv_sec = time1.tv_sec;
          time0.tv_usec = time1.tv_usec;
          ttime += diff1;
          diff2 = ttime  * (total_iterations - iterations_done) / iterations_done / 1000000;
          total_bytes = 244140625 / (double) diff1;
          printf("Position %d, Data Type %d, Iteration %d, Errors: %d, completed %2.2f%%, Read %0.2fGB/s, Write %0.2fGB/s, ETA ",
                  j, i, iterations_done * 100000, total, percent_done, 3.0 * total_bytes, total_bytes);
          print_time_from_seconds ((unsigned) diff2, NULL, 0);
          printf (")\n");
          fflush(NULL);
        }
      }
    }
  }
  printf("Test complete. Total errors: %d.\n", total);
  fflush(NULL);
  cutilSafeCall (cudaFree ((char *) dev_data1));
  cutilSafeCall (cudaFree ((char *) d_compare));
  free((char*) d_data);
}

void cufftbench (int cufftbench_s, int cufftbench_e, int passes, int device_number)
{

  cudaEvent_t start, stop;
  float outerTime;
  int i, j, k;
  int end = cufftbench_e - cufftbench_s + 1;
  float best_time;
  float *total, *max_diff, maxerr = 0.5f;
  int th[] = {32, 64, 128, 256, 512, 1024};
  int n = cufftbench_e << 10;
  int warning = 0;
  cudaError err = cudaSuccess;

  printf ("CUDA bench, testing reasonable fft sizes %dK to %dK, doing %d passes.\n", cufftbench_s, cufftbench_e, passes);
  j = 1;
  while(j < cufftbench_s) j <<= 1;
  k = j;
  while(j < cufftbench_e) j <<= 1;
  if(k > cufftbench_s || j > cufftbench_e) warning = 1;

  total = (float *) malloc (sizeof (float) * end);
  max_diff = (float *) malloc (sizeof (float) * end);

  for(i = 0; i < end; i++)
  {
    total[i] = max_diff[i] = 0.0f;
  }

  alloc_gpu_mem(n);
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
  for (j = cufftbench_s; j <= cufftbench_e; j++)
  {
    if(isReasonable(j) <= 1)
    {
      n = j * 1024;
      cufftSafeCall (cufftPlan1d (&g_plan, n / 2, CUFFT_Z2Z, 1));
      for(k = 0; k < passes; k++)
      {
        cutilSafeCall (cudaEventRecord (start, 0));
        for (i = 0; i < 50; i++)
  	    {
          cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
          square <<< n / (4 * th[3]), th[3] >>> (n, g_x, g_ct);
          cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE);
          rcb <10, 0, 0><<< n / 1024, 128 >>>(g_x, NULL, g_data, g_ttp, g_s, g_carry, g_err, maxerr, 0,0);
          splice <0> <<< (n / 1024 + th[2] - 1) / th[2], th[2] >>> (g_x, NULL, n, 512, g_data, g_carry, g_ttp2);
        }
        cutilSafeCall (cudaEventRecord (stop, 0));
        err = cutilSafeCall1(cudaEventSynchronize (stop));
      	if( cudaSuccess != err)
        {
          total[j] = 0.0;
          max_diff[j] = 0.0;
          k = passes;
          j--;
          cudaDeviceReset();
          init_device(g_dn,0);
          n = cufftbench_e * 1024;
          alloc_gpu_mem(n);
          cutilSafeCall (cudaEventCreate (&start));
          cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
        }
        else
        {
          cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
          i = j - cufftbench_s;
          outerTime /= 50.0f;
          total[i] += outerTime;
          if(outerTime > max_diff[i]) max_diff[i] = outerTime;
        }
      }
     	if(cudaSuccess == err)
      {
        cufftSafeCall (cufftDestroy (g_plan));
        printf ("fft size = %dK, ave time = %6.4f msec, max-ave = %0.5f\n", j, total[i] / passes, max_diff[i] - total[i] / passes);
        fflush(NULL);
      }
    }
  }
  free_gpu(0);
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));

  i = end - 1;
  j = 1;
  while(j < end) j <<= 1;
  j >>= 1;
  k = j - cufftbench_s;
  best_time = total[i] + 1000000.0f;
  while(i >= 0)
  {
    if(total[i] > 0.0f && total[i] < best_time) best_time = total[i];
    else if(i != k) total[i] = 0.0f;
    if(i == k)
    {
      j >>= 1;
      k = j - cufftbench_s;
    }
    i--;
  }

  char fftfile[32];
  FILE *fptr;


  char fftfile_bak[64];

  time_t now;
  struct tm *tm_now = NULL;
  char buffer[192];
  char output[4][32];
  int index = 0;

  now = time(NULL);
  tm_now = localtime(&now);
  strftime(output[0], 32, "%d", tm_now);                             //date
  strftime(output[1], 32, "%H", tm_now);                             //hour
  strftime(output[2], 32, "%M", tm_now);                            //min
  strftime(output[3], 32, "%S", tm_now);                            //sec
  for(i = 0; i < 4; i++) index += sprintf(buffer + index,"%s",output[i]);
  sprintf (fftfile, "%s fft.txt", g_dev.name);
  sprintf (fftfile_bak, "%s fft%s.txt", g_dev.name, buffer);
  (void) unlink (fftfile_bak);
  (void) rename (fftfile, fftfile_bak);
	fptr = fopen_and_lock(fftfile, "w");

  if(!fptr)
  {
    printf ("Cannot open %s.\n",fftfile);
    printf ("Device              %s\n", g_dev.name);
    printf ("Compatibility       %d.%d\n", g_dev.major, g_dev.minor);
    printf ("clockRate (MHz)     %d\n", g_dev.clockRate/1000);
    printf ("memClockRate (MHz)  %d\n", g_dev.memoryClockRate/1000);
    printf("\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
         int tl = (int) (exp(0.9782989529 * log ((double) cufftbench_s + i)) * 22379.9259682859 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        printf("%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fflush(NULL);
  }
  else
  {
    fprintf (fptr, "Device              %s\n", g_dev.name);
    fprintf (fptr, "Compatibility       %d.%d\n", g_dev.major, g_dev.minor);
    fprintf (fptr, "clockRate (MHz)     %d\n", g_dev.clockRate/1000);
    fprintf (fptr, "memClockRate (MHz)  %d\n", g_dev.memoryClockRate/1000);
    fprintf(fptr, "\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
        int tl = (int) (exp(0.9784876919 * log ((double) cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        fprintf(fptr, "%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    unlock_and_fclose(fptr);
    if(warning) printf("\nWARNING, the bounds were not both powers of two, results at either end may not be accurate.\n\n");
    printf("Old %s moved to %s.\n", fftfile, fftfile_bak);
    printf("Optimal fft lengths saved in %s.\nPlease email a copy to james@mersenne.ca.\n", fftfile);
    fflush(NULL);
  }

  free ((char *) total);
  free ((char *) max_diff);
}

int round_off_test(int q, int n, int *j, int *offset)
{
  int k;
  float totalerr = 0.0;
  float terr, avgerr, maxerr = 0.0;
  float max_err = 0.0, max_err1 = 0.0;
  int l_offset = *offset;
  int last = q - 2;
  int error_flag = 0;
  int t_ei = g_ei;


  printf("Running careful round off test for 1000 iterations.\n");
  printf("If average error > 0.25, or maximum error > 0.35,\n");
  printf("the test will restart with a longer FFT.\n");
  fflush(NULL);
  g_ei = 1;
  for (k = 0; k < 1000 && (k + *j <= last); k++)
  {
    error_flag = 2;
    if (k == 999) error_flag = 1;;
    terr = lucas_square (q, n, *j + k, q - 1, &l_offset, &maxerr, error_flag);
    if(terr > maxerr) maxerr = terr;
    if(terr > max_err) max_err = terr;
    if(terr > max_err1) max_err1 = terr;
    totalerr += terr;
    reset_err(&maxerr, 0.0);
    if(terr > 0.35)
    {
      printf ("Iteration = %d < 1000 && err = %5.5f > 0.35, increasing n from %dK\n", k, terr, n/1024);
      g_fftlen++;
      return 1;
    }
    if( k && (k % 100 == 0) )
    {
      printf( "Iteration  %d, average error = %5.5f, max error = %5.5f\n", k, totalerr / k, max_err);
      max_err = 0.0;
    }
  }
  avgerr = totalerr/1000.0;
  if( avgerr > 0.25 )
  {
    printf("Iteration 1000, average error = %5.5f > 0.25 (max error = %5.5f), increasing FFT length and restarting\n", avgerr, max_err);
    g_fftlen++;
    return 1;
  }
  else if( avgerr < 0 )
  {
    fprintf(stderr, "Something's gone terribly wrong! Avgerr = %5.5f < 0 !\n", avgerr);
    exit (2);
  }
  else
  {
    printf("Iteration 1000, average error = %5.5f <= 0.25 (max error = %5.5f), continuing test.\n", avgerr, max_err1);
  }
  *offset = l_offset;
  *j += 1000;
  g_ei = t_ei;
  return 0;
}

void check_residue (int ls)
{
  int q, n, j, last = 10000, offset;
  unsigned *x_packed = NULL;
  int  *x_int = NULL;
  float terr, maxerr = 0.25;
  int restarting;
  int error_flag = 0;
  int bad_selftest = 0;
  int i;
  unsigned long long diff;
  unsigned long long diff1;
  unsigned long long expected_residue;
  unsigned long long residue;
  timeval time0, time1;
  int lim = 20;

  typedef struct res_test
  {
    int exponent;
    unsigned long long residue;
  } res_test;

  res_test tests[20] = {{    86243, 0x23992ccd735a03d9}, {    132049, 0x4c52a92b54635f9e},
                       {    216091, 0x30247786758b8792}, {    756839, 0x5d2cbe7cb24a109a},
                       {    859433, 0x3c4ad525c2d0aed0}, {   1257787, 0x3f45bf9bea7213ea},
                       {   1398269, 0xa4a6d2f0e34629db}, {   2976221, 0x2a7111b7f70fea2f},
                       {   3021377, 0x6387a70a85d46baf}, {   6972593, 0x88f1d2640adb89e1},
                       {  13466917, 0x9fdc1f4092b15d69}, {  20996011, 0x5fc58920a821da11},
                       {  24036583, 0xcbdef38a0bdc4f00}, {  25964951, 0x62eb3ff0a5f6237c},
                       {  30402457, 0x0b8600ef47e69d27}, {  32582657, 0x02751b7fcec76bb1},
                       {  37156667, 0x67ad7646a1fad514}, {  42643801, 0x8f90d78d5007bba7},
                       {  43112609, 0xe86891ebf6cd70c4}, {  57885161, 0x76c27556683cd84d}};

  res_test testl[92] = {{    22133, 0xb3ed67a5d4be2a9f}, {     43633, 0xa3f3e669e061bfbe},
                       {     85933, 0x474e01b06d9698f8}, {    169409, 0x3d32414583ac7d9a},
                       {    190097, 0xf681a2a2ad0397fb}, {    210739, 0x7e3b5a8b4e4c43a0},
                       {    292921, 0xade86fd9a4ad99f1}, {    333803, 0xe7311b6b953a609f},
                       {    374587, 0xdb4ca1fad1c3e276}, {    435571, 0xb6711834a5dd0ad1},
                       {    657719, 0xc56715170d6819ef}, {    738083, 0x2d6f637091a3ce6d},
                       {    818239, 0xe63f1cc63af264cb}, {    978041, 0xc049101d2a0f945b},
                       {   1137271, 0x8e006ecd6719158f}, {   1216693, 0x847bbf5d0a077cad},
                       {   1296011, 0xc5002ae321cc3055}, {   1414741, 0x7c6237b4f067d8ca},
                       {   1612249, 0xfd4c832d1dbf746f}, {   1927129, 0x155eb42707e90cd9},
                       {   2240863, 0xf5e629fcc1247e4a}, {   2397383, 0xf58df7be5d73b369},
                       {   2553659, 0x0a9357c955a13ca5}, {   2865601, 0xdac4555e41809026},
                       {   3176779, 0x1a8ed18fa285f151}, {   3215629, 0x9e0c8eb95b15ae66},
                       {   3332107, 0x7a539a9b8060b0d8}, {   3564823, 0x820e428ad972cd86},
                       {   3797219, 0xa7223e4ecad04860}, {   3874583, 0xf60c0a837ec80a50},
                       {   3951977, 0xb5da72b662eb3652}, {   4415431,0x31d58de3e1d79a85 },
                       {   4723783, 0x89fac94192cc3bec}, {   5031737, 0x6371a0f4aef2af32},
                       {   5646379, 0xf03d3bbaa51fbe63}, {   6259537, 0x026805e6b4f1b0fe},
                       {   6336103, 0x0f0718d2da4aaad3}, {   6565633, 0x98d84929de4e6f63},
                       {   7024163, 0xc030210791a14e9c}, {   7482053, 0xf54842c06ba7e3c4},
                       {   7634537, 0x24872a305faae4f8}, {   7786967, 0x060dfca872385f9b},
                       {   8700169, 0x4f95b9afd65b3c7e}, {   9914521, 0xc6ff39fd699d624c},
                       {  11125619, 0x1484b22a17312da0}, {  12333809, 0xbd8435b83f062d6a},
                       {  12484649, 0xd56aa903718b2353}, {  12936919, 0x1527f970ad3020bf},
                       {  13840423, 0x002a2aa0b97306b7}, {  14742617, 0xf63addc70cde0add},
                       {  15043099, 0xe52f5e2045a2db30}, {  15343429, 0x5eda3ab1816505b9},
                       {  16543493, 0x85ea99f94dc20a8b}, {  17142793, 0x94edb737270e4ca7},
                       {  17217653, 0x3ec0aa4728b97ef3}, {  19535569, 0x708d03d1fe4c06f2},
                       {  21921901, 0x38651890258c09bb}, {  22368691, 0xcc264d5bc61bc957},
                       {  24599717, 0x76e7ab92aca768ab}, {  25490893, 0x135f7e63e7357efb},
                       {  27271147, 0xb1a47be0366530ef}, {  29640913, 0x4d51f44ffae1c913},
                       {  30232693, 0xcab02c7178561218}, {  32597297, 0xec15163c817fcfdd},
                       {  33778141, 0x9145db2f2080172e}, {  38492887, 0x36efe22824adb35c},
                       {  43194913, 0xf13bdac634316fa2}, {  47885689, 0x51a71b010cacfd92},
                       {  48471289, 0x5f504f82abefdd1c}, {  50227213, 0x643739d94fe6789c},
                       {  53735041, 0x653267115ab57688}, {  58404433, 0x6e4b3224349080c8},
                       {  59570449, 0x8d7c6550107a2d0a}, {  64229677, 0x796701d5562dce22},
                       {  66556463, 0xe418b5d4fb5c7e45}, {  66847171, 0xc01ffc681dbb8966},
                       {  75846319, 0x626ab6b09b607ba9}, {  85111207, 0x14bb68062d100ebd},
                       {  86845813, 0x88220ac98093b65c}, {  95507747, 0xa92c24ad9b6655f1},
                       {  97454309, 0xd6d4b3b14c1b73d8}, { 105879517, 0x3310a4b03a18f4a8},
                       { 111056879, 0x59e95ddb73c59624}, { 115080019, 0x293d0b202c9e3c63},
                       { 117377567, 0xdf57cd5d866f1065}, { 118813021, 0x1cd6a8caef092277},
                       { 126558077, 0x40b1cf42af7f99a7}, { 131142761, 0x046ab21b10a27189},
                       { 131715607, 0x8fcfb36b868738ec}, { 142017539, 0x801794367f35ccb6},
                       { 147162241, 0x72c8773764a260b0}, { 149447533, 0x24702427a0f5d673}};

  g_ei = 1;
  if(ls) lim = 92;
  for(i = 0; i < lim; i++)
  {
    if(ls)
    {
      q = testl[i].exponent;
      expected_residue = testl[i].residue;
    }
    else
    {
      q = tests[i].exponent;
      expected_residue = tests[i].residue;
    }
    g_fftlen = 0;
    do
    {
      restarting = 0;
      if(!x_packed) x_packed = read_checkpoint(q);
      x_int = init_lucas(x_packed, q, &n, &j, &offset, NULL, NULL, NULL);
      if(!x_int) exit (2);
      if(!restarting) printf ("Starting self test M%d fft length = %dK\n", q, n/1024);
      gettimeofday (&time0, NULL);
      if(g_ro) restarting = round_off_test(q, n, &j, &offset);
      if(restarting) close_lucas (x_int);
      fflush (stdout);

      for (; !restarting && j <= last; j++)
      {
        if(j == last) error_flag = 1;
        else if(j % 100 == 0) error_flag = 2;
        else error_flag = 0;
        terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
        if(error_flag == 2 && j <= 1000) reset_err(&maxerr, 0.75);
        if(terr > 0.4)
        {
          g_fftlen++;
          restarting = 1;
        }
      }
    }
    while (restarting);
    cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
    residue = find_residue(x_int, q, n, offset);
    gettimeofday (&time1, NULL);
    diff = time1.tv_sec - time0.tv_sec;
    diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
    printf ("Iteration %d ", j - 1);
    printbits(residue, q, n, offset, 0, 0);
    printf (", error = %5.5f, real: ", maxerr);
    print_time_from_seconds ((unsigned) diff, NULL, 0);
    printf (", %4.4f ms/iter\n", diff1 / 1000.0 / last);
    fflush (stdout);

    close_lucas (x_int);
    free ((char *) x_packed);
    x_packed = NULL;
    if(residue != expected_residue)
    {
      printf("Expected residue [%016llx] does not match actual residue [%016llx]\n\n", expected_residue, residue);
      fflush (stdout);
      bad_selftest++;
    }
    else
    {
      printf("This residue is correct.\n\n");
      fflush (stdout);
    }
  }
  if (bad_selftest)
  {
    fprintf(stderr, "Error: There ");
    bad_selftest > 1 ? fprintf(stderr, "were %d bad selftests!\n",bad_selftest) : fprintf(stderr, "was a bad selftest!\n");
  }
}

/**************************************************************************
 *                                                                        *
 *                          Keyboard Interaction                          *
 *                                                                        *
 **************************************************************************/
void
SetQuitting (int sig)
{
  g_qu = 1;
  sig==SIGINT ? printf( "\tSIGINT") : (sig==SIGTERM ? printf( "\tSIGTERM") : printf( "\tUnknown signal")) ;
  printf( " caught, writing checkpoint.");
}

#ifndef _MSC_VER
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
int
_kbhit (void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr (STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl (STDIN_FILENO, F_GETFL, 0);
  fcntl (STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar ();

  tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
  fcntl (STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
  {
    ungetc (ch, stdin);
    return 1;
  }

  return 0;
}
#else
#include <conio.h>
#endif

int interact(int );
/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/

int
check (int q)
{
  int  *x_int = NULL;
  unsigned *x_packed = NULL;
  int n, j, j_b, last = q - 2;
  int offset, o_b;
  float maxerr, temperr, terr = 0.0f;
  int interact_result = 0;
  timeval time0, time1;
  unsigned long long total_time = 0, tt_b = 0, diff, diff1, diff2;
  unsigned long long time_adj = 0, ta_b= 0;
  unsigned long long residue;
  unsigned iter_adj = 0, ia_b = 0;
  int last_chk = 0;
  int restarting;
  int retry = 1;
  int fft_reset = 0;
  int error_flag;
  int error_reset = 1;
  int end = (q + 31) / 32;
  float error_limit = (g_el - g_el / 8.0 * log ( (float) g_ei)/ log(100.0)) / 100.0;
  signal (SIGTERM, SetQuitting);
  signal (SIGINT, SetQuitting);
  do
  {				/* while (restarting) */
    maxerr = temperr = 0.25f;

    if(!x_packed) x_packed = read_checkpoint(q);
    x_int = init_lucas(x_packed, q, &n, &j, &offset, &total_time, &time_adj, &iter_adj);
    if(!x_int) exit (2);
    j_b = j - 1;
    o_b = offset;
    tt_b = total_time;
    ta_b = time_adj;
    ia_b = iter_adj;

    restarting = 0;
    if(j == 1)
    {
      printf ("Starting M%d fft length = %dK\n", q, n/1024);
      if(g_ro)
      {
        restarting = round_off_test(q, n, &j, &offset);
        iter_adj = 1000;
        if(!restarting)
        {
          cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
          j_b = j - 1;
          o_b = offset;
          tt_b = total_time;
          ia_b = iter_adj;
          set_checkpoint_data(x_packed, q, j, offset, total_time, time_adj, iter_adj);
          standardize_digits(x_int, q, n);
          pack_bits(x_int, x_packed, q, n);
          last_chk = j;
        }
      }
    }
    else
    {
      printf ("\nContinuing M%d @ iteration %d with fft length %dK, %5.2f%% done\n\n", q, j, n/1024, (float) j/q*100);
      last_chk = j;
    }
    fflush (stdout);
    if(!restarting)
    {
      gettimeofday (&time0, NULL);
    }
    for (; !restarting && j <= last; j++) // Main LL loop
    {
	    error_flag = 0;
	    if (j % g_ri == 0 || j == last) error_flag = 1;
      else if ((j % 100) == 0) error_flag = 2;
      terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
      if(terr < 0.0f)  //Nvidia, cufft error
      {
        if(j_b + 1 != (int) x_packed[end + 2]);
        {
          set_checkpoint_data(x_packed, q, j_b + 1, o_b, tt_b, ta_b, ia_b);
          standardize_digits(x_int, q, n);
          pack_bits(x_int, x_packed, q, n);
        }
        printf("Resetting device and restarting from last checkpoint.\n\n");
        cudaDeviceReset();
        init_device(g_dn, 0);
        restarting = 1;
      }
      else
      {
        if ((error_flag & 1) || g_qu) //checkpoint iteration or quitting or cufft error
        {
          if (!(error_flag & 1)) //quitting, but g_int not up to date, do 1 more iteration
          {
            j++;
            error_flag = 1;
            terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
          }
          if(terr <= error_limit)
          {
            cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
            residue = find_residue(x_int, q, n, offset);
            gettimeofday (&time1, NULL);
            diff = time1.tv_sec - time0.tv_sec;
            diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
            total_time += diff1;
            diff2 = (unsigned long long) (total_time - time_adj) / 1000000.0 * (last - j) / (j - iter_adj);
            time0.tv_sec = time1.tv_sec;
            time0.tv_usec = time1.tv_usec;
            j_b = j;
            o_b = offset;
            tt_b = total_time;
            ta_b = time_adj;
            ia_b = iter_adj;
            if(j % g_cpi == 0 || g_qu)
            {
              set_checkpoint_data(x_packed, q, j + 1, offset, total_time, time_adj, iter_adj);
              standardize_digits(x_int, q, n);
              pack_bits(x_int, x_packed, q, n);
              write_checkpoint(x_packed, q, residue);
            }
            if(g_qu)
            {
              printf(" Estimated time spent so far: ");
              print_time_from_seconds((unsigned) (total_time / 1000000), NULL,0);
              printf("\n\n");
              j = last + 1;
            }
            else if(j < last) //screen output
            {
              if(g_output_interval) process_output(q, n, j, offset, last_chk, maxerr, residue, diff, diff1, diff2, total_time);
              else
              {
                printf ("Iteration %d ", j);
                printbits(residue, q, n, offset, 0, 0);
                printf (", error = %5.5f, real: ", maxerr);
                print_time_from_seconds ((unsigned) diff, NULL, 0);
                printf (", %4.4f ms/iter, ETA: ", diff1 / 1000.0 / (j - last_chk));
                print_time_from_seconds ((unsigned) diff2, NULL, 0);
                printf (", %5.2f%%\n", (float) j/q*100);
              }
              fflush (stdout);
              last_chk = j;
              if(!error_reset) reset_err(&maxerr, g_er / 100.0f); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
              if(fft_reset) // Larger fft fixed the error, reset fft and continue
              {
                if(j % g_cpi)
                {
                  set_checkpoint_data(x_packed, q, j + 1, offset, total_time, time_adj, iter_adj);
                  standardize_digits(x_int, q, n);
                  pack_bits(x_int, x_packed, q, n);
                }
                g_fftlen--;
                restarting = 1;
                printf("Resettng fft.\n\n");
                fft_reset = 0;
                retry = 1;
              }
              else if(retry == 0) // Retrying fixed the error
              {
                printf("Looks like the error went away, continuing.\n\n");
                retry = 1;
              }
            }
          }
        }
      }
      if (terr > error_limit)
      {
        printf ("Round off error at iteration = %d, err = %0.5g > %0.5g, fft = %dK.\n", j,  terr, error_limit, n / 1024);
        if(g_fftlen >= g_ubi) // fft is at upper bound. overflow errors possible
        {
          printf("Something is wrong! Quitting.\n\n");
          exit (2);
        }
        //else if(g_fftlen <= g_lbi) // fft is at lower bound, round off errors possible
        //{
        //  printf("Increasing fft and restarting from last checkpoint.\n\n");
        //  g_fftlen++;
        //}
        else // fft is not a boundary case
        {
          if(retry)  // Redo from last checkpoint, the more complete rounding/balancing
          {          // on cpu sometimes takes care of the error. Hardware errors go away too.
            printf("Restarting from last checkpoint to see if the error is repeatable.\n\n");
            retry = 0;
          }
          else // Retrying didn't fix the error
          {
            if(fft_reset) // Larger fft didn't fix the error, give up
            {
              printf("The error won't go away. I give up.\n\n");
              exit(2);
            }
            else // Larger fft will usually fix a round off error
            {
              printf("The error persists.\n");
              printf("Trying a larger fft until the next checkpoint.\n\n");
              g_fftlen++;
              fft_reset = 1;
            }
          }
        }
        if(j_b + 1 != (int) x_packed[end + 2]);
        {
          set_checkpoint_data(x_packed, q, j_b + 1, o_b, tt_b, ta_b, ia_b);
          standardize_digits(x_int, q, n);
          pack_bits(x_int, x_packed, q, n);
        }
        restarting = 1;
        reset_err(&maxerr, 0.25);
        error_reset = 1;
      }
      else if(error_reset && (error_flag & 3))
      {
          if(terr < temperr + 0.0001f)
          {
            maxerr *= 0.5f;
            temperr = maxerr;
          }
          else error_reset = 0;
      }

      if ( g_ki && !restarting && !g_qu && (!(j & 15)) && _kbhit()) interact_result = interact(n);
      if(interact_result & 3)
      {
        if(fft_reset) fft_reset = 0;
        else
        {
          if(!(error_flag & 1))
          {
            j++;
            error_flag |= 1;
            terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
            if(terr <= error_limit)
            {
              cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
              gettimeofday (&time1, NULL);
              diff = time1.tv_sec - time0.tv_sec;
              diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
              total_time += diff1;
              time_adj = total_time;
              iter_adj = j + 1;
              time0.tv_sec = time1.tv_sec;
              time0.tv_usec = time1.tv_usec;
              j_b = j;
              o_b = offset;
              tt_b = total_time;
              ta_b = time_adj;
              ia_b = iter_adj;
              last_chk = j;
              set_checkpoint_data(x_packed, q, j + 1, offset, total_time, time_adj, iter_adj);
              standardize_digits(x_int, q, n);
              pack_bits(x_int, x_packed, q, n);
              reset_err(&maxerr, g_er / 100.0f);
              error_reset = 1;
            }
          }
          if(interact_result & 1) restarting = 1;
        }
      }
	    else if (interact_result & 4)
      {
        error_limit = (g_el - g_el / 8.0 * log ( (float) g_ei)/ log(100.0)) / 100.0;
        //printf("%0.5f\n", error_limit);
      }
      interact_result = 0;
	    fflush (stdout);
	  } /* end main LL for-loop */

    if (!restarting && !g_qu)
	  { // done with test
	    gettimeofday (&time1, NULL);
	    FILE* fp = fopen_and_lock(g_RESULTSFILE, "a");
	    if(!fp)
	    {
	      fprintf (stderr, "Cannot write results to %s\n\n", g_RESULTSFILE);
	      exit (1);
	    }
	    cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
      residue = find_residue(x_int, q, n, offset);
      if (residue == 0)
      {
        printf ("M( %d )P, n = %dK, %s", q, n / 1024, program);
        if (fp) fprintf (fp, "M( %d )P, offset = %d, n = %dK, %s", q, offset, n / 1024, program);
      }
	    else printbits(residue, q, n, offset, fp, 1);
      diff = time1.tv_sec - time0.tv_sec;
      total_time +=  diff * 1000000 + time1.tv_usec - time0.tv_usec;
      printf (", estimated total time = ");
      print_time_from_seconds((unsigned) (total_time / 1000000), NULL, 0);

	    if( g_AID[0] && strncasecmp(g_AID, "N/A", 3) )
      { // If (AID is not empty), AND (AID is NOT "N/A") (case insensitive)
        fprintf(fp, ", AID: %s\n", g_AID);
	    }
      else fprintf(fp, "\n");
	    unlock_and_fclose(fp);
	    fflush (stdout);
	    if(residue) rm_checkpoint (q);
      g_fft_count = 0;
	    printf("\n\n");
	  }
    if(terr < 0.0) free_host(x_int);
    else close_lucas(x_int);
  }
  while (restarting);
  free ((char *) x_packed);
  x_packed = NULL;
  return (0);
}


void parse_args(int argc, char *argv[], int* q, int* cufftbench_s, int* cufftbench_e, int* cufftbench_d, int* cufftbench_m);
		/* The rest of the opts are global */

void encode_output_options(void)
{
  int i = 0, j = 0, temp;
  unsigned k;
  char token[196];
  char c;

  c = g_output_string[0];
  while(c)
  {
    if(c == '%')
    {
      i++;
      c = g_output_string[i];
      switch (c)
      {
        case 'a' : //day of week
                   g_output_code[j] = 0;
                   j++;
                   break;
        case 'M' : //month
                   g_output_code[j] = 1;
                   j++;
                   break;
        case 'D' : //date
                   g_output_code[j] = 2;
                   j++;
                   break;
        case 'h' : //hour
                   g_output_code[j] = 3;
                   j++;
                   break;
        case 'm' : //minutes
                   g_output_code[j] = 4;
                   j++;
                   break;
        case 's' : //seconds
                   g_output_code[j] = 5;
                   j++;
                   break;
        case 'y' : //year
                   g_output_code[j] = 6;
                   j++;
                   break;
        case 'z' : //time zone
                   g_output_code[j] = 7;
                   j++;
                   break;
        case 'p' : //testing exponent
                   g_output_code[j] = 8;
                   j++;
                   break;
        case 'P' : //testing Mersenne
                   g_output_code[j] = 9;
                   j++;
                   break;
        case 'f' : //fft as a multiple of K
                   g_output_code[j] = 10;
                   j++;
                   break;
        case 'F' : //fft
                   g_output_code[j] = 11;
                   j++;
                   break;
        case 'i' : //iteration number
                   g_output_code[j] = 12;
                   j++;
                   break;
        case 'o' : //offset
                   g_output_code[j] = 13;
                   j++;
                   break;
        case 'r' : //residue
                   g_output_code[j] = 14;
                   j++;
                   break;
        case 'R' : //residue with upper case hex
                   g_output_code[j] = 15;
                   j++;
                   break;
        case 'C' : //time since last checkpoint hh:mm:ss
                   g_output_code[j] = 16;
                   j++;
                   break;
        case 'T' : //eta d:hh:mm:ss
                   g_output_code[j] = 17;
                   j++;
                   break;
        case 'N' : //residue with upper case hex
                   g_output_code[j] = 18;
                   j++;
                   break;
        case '%' : //residue with upper case hex
                   g_output_code[j] = 19;
                   j++;
                   break;
        case 'x' : //round off error
                   g_output_code[j] = 25;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
        case 'q' : //iteration timing
                   g_output_code[j] = 26;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                  j++;
                   break;
        case 'c' : //checkpoint interval timing ms/iter
                   g_output_code[j] = 27;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
        case 'Q' : //checkpoint interval timing iter/sec
                   g_output_code[j] = 28;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
        case 'd' : //percent completed
                   g_output_code[j] = 29;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
       default  :
                   break;
      }//end switch
      i++;
      c = g_output_string[i];
    }
    else // c != '%', write the string up to the next % to array
    {
      k = 0;
      while(c && c != '%')
      {
        token[k] = c;
        k++;
        i++;
        c = g_output_string[i];
      }
      token[k] = '\0';
      int m = 30, found = 0;
      while(m < 50 && !found)
      {
        if (strlen(g_output[m]) == 0)
        {
          k = (k < 32 ? k : 32);
          strncpy(g_output[m], token, k);
          g_output_code[j] = m;
          j++;
          found = 1;
        }
        else if(!strncmp(token, g_output[m], k) && strlen(g_output[m]) == k)
        {
          g_output_code[j] = m;
          j++;
          found = 1;
       }
        m++;
      }
    }
  }//end while(c)
  g_output_code[j] = -1;
}

int check_interval(int interval)
{
  int k = interval, j = 1;

  while(j <= k) j *= 10;
  j /= 10;
  k =  (int) floor( k / (double) j + 0.5);
  if(k == 3) k = 2;
  else if(k > 3 && k < 8) k = 5;
  else if ( k > 7) k = 10;
  k *= j;
  return k;
}

void process_options(void)
{
#define THREADS_DFLT 256
#define ERROR_ITER_DFLT 100
#define REPORT_ITER_DFLT 10000
#define CHECKPOINT_ITER_DFLT 100000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define SLEEP_FLAG_DFLT 0
#define SLEEP_VALUE_DFLT 100
#define ROT_F_DFLT 0
#define POLITE_FLAG_DFLT 0
#define POLITE_VALUE_DFLT 1
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"
#define ER_DFLT 85
#define OUTPUT_DFLT "0"
#define HEADER_DFLT "0"
#define HEADERINT_DFLT 15
#define BIG_CARRY_DFLT 0
#define ERROR_LIMIT_DFLT 40
  char fft_str[192] = "\0";


  if (file_exists(g_INIFILE))
  {
    if( g_sf < 0 &&           !IniGetInt(g_INIFILE, "SaveAllCheckpoints", &g_sf, S_F_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
    if( g_ki < 0 &&           !IniGetInt(g_INIFILE, "Interactive", &g_ki, 0) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
    if( g_ro < 0 &&           !IniGetInt(g_INIFILE, "RoundoffTest", &g_ro, ROT_F_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option RoundoffTest; using default: off\n")*/;
    if( g_df < 0 &&           !IniGetInt(g_INIFILE, "PrintDeviceInfo", &g_df, D_F_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
    if( g_pf < 0 &&	          !IniGetInt(g_INIFILE, "Polite", &g_pf, POLITE_FLAG_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT)*/;
    if( g_po < 0 &&	          !IniGetInt(g_INIFILE, "PoliteValue", &g_po, POLITE_VALUE_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT)*/;
    if( g_sl < 0 &&	          !IniGetInt(g_INIFILE, "Sleep", &g_sl, SLEEP_FLAG_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT)*/;
    if( g_sv < 0 &&	          !IniGetInt(g_INIFILE, "SleepValue", &g_sv, SLEEP_VALUE_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT)*/;
    if( g_dn < 0 &&           !IniGetInt(g_INIFILE, "DeviceNumber", &g_dn, 0) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n")*/;
    if( g_cpi < 0 &&          !IniGetInt(g_INIFILE, "CheckpointIterations", &g_cpi, CHECKPOINT_ITER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
    if( g_ri < 0 &&          !IniGetInt(g_INIFILE, "ReportIterations", &g_ri, REPORT_ITER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
    if( g_ei < 0 &&          !IniGetInt(g_INIFILE, "ErrorIterations", &g_ei, ERROR_ITER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
    if( g_el < 0 &&          !IniGetInt(g_INIFILE, "ErrorLimit", &g_el, ERROR_LIMIT_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
    if( g_er < 0 &&           !IniGetInt(g_INIFILE, "ErrorReset", &g_er, ER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option ErrorReset using default: 85\n")*/;
    if( g_th < 0 &&           !IniGetInt2(g_INIFILE, "Threads", &g_thr[0], &g_thr[1], THREADS_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: 256 128 128%d\n", THREADS_DFLT)*/;
    if( g_th < 0 &&           !IniGetInt(g_INIFILE, "BigCarry", &g_bc, BIG_CARRY_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: 256 128 128%d\n", THREADS_DFLT)*/;
    if( g_fftlen < 0 &&       !IniGetStr(g_INIFILE, "FFTLength", fft_str, "\0") )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
    if( !g_input_file[0] &&	  !IniGetStr(g_INIFILE, "WorkFile", g_input_file, WORKFILE_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
      /* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
    if( !g_RESULTSFILE[0] &&  !IniGetStr(g_INIFILE, "ResultsFile", g_RESULTSFILE, RESULTSFILE_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
    if( g_output_string[0] < 0 &&  !IniGetStr(g_INIFILE, "OutputString", g_output_string, OUTPUT_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option OutputString; using default \"%s\"\n", OUTPUT_DFLT);
    if( g_output_header[0] < 0 &&  !IniGetStr(g_INIFILE, "OutputHeader", g_output_header, HEADER_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option OutputHeader; using default \"%s\"\n", HEADER_DFLT);
    if( g_output_interval < 0 &&  !IniGetInt(g_INIFILE, "OutputHInterval", &g_output_interval, 0) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option OutputHInterval; using default \"%s\"\n", HEADERINT_DFLT)*/;
    if( g_folder[0] < 0 &&           !IniGetStr(g_INIFILE, "SaveFolder", g_folder, SAVE_FOLDER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
  }
  else fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");

  //set default values for anything not found in ini file
  if( g_cpi < 0 ) g_cpi = CHECKPOINT_ITER_DFLT;
  if( g_ri < 0 ) g_ri = REPORT_ITER_DFLT;
  if( g_ei < 0 ) g_ei = ERROR_ITER_DFLT;
  if( g_el < 0 ) g_el = ERROR_LIMIT_DFLT;
  if( g_thr[0] < 0 ) g_thr[0] = THREADS_DFLT;
  if( g_fftlen < 0 ) g_fftlen = 0;
  if( g_sf < 0 ) g_sf = S_F_DFLT;
  if( g_ki < 0 ) g_ki = K_F_DFLT;
  if( g_dn < 0 ) g_dn = 0;
  if( g_df < 0 ) g_df = D_F_DFLT;
  if( g_bc < 0 ) g_bc = BIG_CARRY_DFLT;
  if( g_pf < 0 ) g_po = POLITE_FLAG_DFLT;
  if( g_po < 0 ) g_po = POLITE_VALUE_DFLT;
  if( g_sl < 0 ) g_po = SLEEP_FLAG_DFLT;
  if( g_sv < 0 ) g_po = SLEEP_VALUE_DFLT;
  if( g_output_string[0] < 0) sprintf(g_output_string, OUTPUT_DFLT);
  if( g_folder[0] < 0) sprintf(g_folder, SAVE_FOLDER_DFLT);
  if( !g_input_file[0] ) sprintf(g_input_file, WORKFILE_DFLT);
  if( !g_RESULTSFILE[0] ) sprintf(g_RESULTSFILE, RESULTSFILE_DFLT);
  if( !g_output_header[0] ) sprintf(g_output_header, HEADER_DFLT);
  if( !g_output_string[0] ) sprintf(g_output_string, OUTPUT_DFLT);
  if( g_output_interval < 0 ) g_output_interval = HEADERINT_DFLT;

  if( g_fftlen < 0 ) g_fftlen = fft_from_str(fft_str); // possible if -f not on command line
  int k = check_interval(g_cpi);
  if(k != g_cpi)
  {
    printf("CheckpointIterations = %d from CUDALucas.ini must have the form k*10^m for k = 1, 2, or 5.\n",g_cpi);
    printf("Changing to %d.\n", k);
    g_cpi = k;
  }
  k = check_interval(g_ri);
  if(k != g_ri)
  {
    printf("ReportIterations = %d from CUDALucas.ini must have the form k*10^m for k = 1, 2, or 5.\n",g_ri);
    printf("Changing to %d.\n", k);
    g_ri = k;
  }
  k = check_interval(g_ei);
  if(k != g_ei)
  {
    printf("ErrorIterations = %d from CUDALucas.ini must have the form k*10^m for k = 1, 2, or 5.\n",g_ei);
    printf("Changing to %d.\n", k);
    g_ei = k;
  }
  k = check_interval(g_sv);
  if(k != g_sv)
  {
    printf("sleep value = %d from CUDALucas.ini must have the form k*10^m for k = 1, 2, or 5.\n",g_sv);
    printf("Changing to %d.\n", k);
    g_sv = k;
  }
  if(g_el > 47) g_el = 47;
  if(g_el < 1) g_el = 1;
  encode_output_options();
}

int main (int argc, char *argv[])
{
  /* "Production" opts to be read in from command line or ini file */
  int q = -1;
  int f_f;
  int cufftbench_s, cufftbench_e, cufftbench_d, cufftbench_m;

  printf("\n");

  cufftbench_s = cufftbench_e = cufftbench_d = 0;
  cufftbench_m = g_cpi = g_ri = g_er = g_thr[0] = g_fftlen = g_pf = g_sl = g_sv = -1;
  g_po = g_sf = g_df = g_ki = g_ro = g_th = g_ei = g_bc = g_rt = g_el = g_dn = -1;
  g_output_string[0] = g_output_header[0] = g_output_interval = g_folder[0] = -1;
  g_AID[0] = g_input_file[0] = g_RESULTSFILE[0] = 0; /* First character is null terminator */

  parse_args(argc, argv, &q, &cufftbench_s, &cufftbench_e, &cufftbench_d, &cufftbench_m); // The rest of the args are globals
  process_options();
  f_f = g_fftlen; // if the user has given an override... then note this length must be kept between tests
  init_device (g_dn, g_df);
  if (g_rt >= 0) check_residue (g_rt);
  else if (cufftbench_m > -1) threadbench (cufftbench_s, cufftbench_e, cufftbench_d, cufftbench_m, g_dn);
  else if (cufftbench_d > 0) cufftbench (cufftbench_s, cufftbench_e, cufftbench_d, g_dn);
  else if (cufftbench_e > 0) memtest (cufftbench_s, cufftbench_e, g_dn);
  else
  {
    if (q <= 0)
    {
      do
      {
        g_fftlen = f_f; // fftlen and AID change between tests, so be sure to reset them
        g_AID[0] = 0;
        if(get_next_assignment(g_input_file, &q, &g_fftlen, &g_AID) || q < 0) exit (2);
        check (q);
        if(!g_qu && clear_assignment(g_input_file, q)) exit (2);
      } while(!g_qu);
    }
    else
    {
      int res = valid_assignment(q, g_fftlen); //! v_a prints warning
      if (!res) exit (2);
      if (res == -1)
      {
        q |= 1;
        while(q > 0 && !isprime(q)) q += 2;
        if(q < 0)
        {
          fprintf(stderr, "Warning: exponent is too big besides!\n");
          exit (2);
        }
        else fprintf(stderr, "Using %d instead.\n", q);
      }
      check (q);
    }
  }
}


void parse_args(int argc,
                char *argv[],
                int* q,
                int* cufftbench_s,
                int* cufftbench_e,
                int* cufftbench_d,
                int* cufftbench_m)
{

  while (argc > 1)
  {
    if (strcmp (argv[1], "-h") == 0) //Help
    {
  	  fprintf (stderr, "$ CUDALucas -h|-v\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads t1 t2] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-polite iteration] [-k] exponent|input_filename\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads t1 t2] -r [0|1]\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] -cufftbench start end passes (see cudalucas.ini)\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] -threadbench start end passes mode (see cudalucas.ini)\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] -memtest size passes (see cudalucas.ini)\n\n");
      fprintf (stderr, "                       -h          print this help message\n");
      fprintf (stderr, "                       -v          print version number\n");
      fprintf (stderr, "                       -info       print device information\n");
      fprintf (stderr, "                       -i          set .ini file name (default = \"CUDALucas.ini\")\n");
  	  fprintf (stderr, "                       -threads    set threads numbers (eg -threads 256 128)\n");
  	  fprintf (stderr, "                       -f          set fft length (if round off error then exit)\n");
  	  fprintf (stderr, "                       -s          save all checkpoint files\n");
  	  fprintf (stderr, "                       -polite     GPU is polite every n iterations (default -polite 0) (-polite 0 = GPU aggressive)\n");
 	  fprintf (stderr, "                       -r          exec residue test.\n");
  	  fprintf (stderr, "                       -k          enable keys (see CUDALucas.ini for details.)\n\n");
  	  exit (2);
    }
    else if (strcmp (argv[1], "-v") == 0) //Version
    {
      printf("%s\n\n", program);
      exit (2);
    }
    else if (strcmp (argv[1], "-polite") == 0) // Polite option
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -polite option\n\n");
	      exit (2);
	    }
	    g_po = atoi (argv[2]);
	    if (g_po == 0)
	    {
	      g_pf = 0;
	      g_po = 100;
	    }
	    else g_pf = 1;
      argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-r") == 0) // Residue check
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
        g_rt = 0;
	      argv++;
	      argc--;
	    }
      else
      {
	      g_rt = atoi (argv[2]);
	      argv += 2;
	      argc -= 2;
      }
	  }
    else if (strcmp (argv[1], "-k") == 0) // Interactive option
	  {
	    g_ki = 1;
	    argv++;
	    argc--;
	  }
    else if (strcmp (argv[1], "-d") == 0) // Device number
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d option\n\n");
	      exit (2);
	    }
	    g_dn = atoi (argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-i") == 0) //ini file
	  {
	    if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -i option\n\n");
	      exit (2);
	    }
	    sprintf (g_INIFILE, "%s", argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-e") == 0) //ini file
	  {
	    if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -e option\n\n");
	      exit (2);
	    }
	    g_el = atoi (argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-info") == 0) // Print device info
    {
      g_df = 1;
      argv++;
      argc--;
    }
    else if (strcmp (argv[1], "-cufftbench") == 0) //cufftbench parameters
	  {
	    if (argc < 5 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-')
	    {
	      fprintf (stderr, "can't parse -cufftbench option\n\n");
	      exit (2);
	    }
	    *cufftbench_s = atoi (argv[2]);
	    *cufftbench_e = atoi (argv[3]);
	    *cufftbench_d = atoi (argv[4]);
	    argv += 4;
	    argc -= 4;
	  }
    else if (strcmp (argv[1], "-threadbench") == 0) //cufftbench parameters
	  {
	    if (argc < 6 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-' || argv[5][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threadbench option\n\n");
	      exit (2);
	    }
	    *cufftbench_s = atoi (argv[2]);
	    *cufftbench_e = atoi (argv[3]);
	    *cufftbench_d = atoi (argv[4]);
	    *cufftbench_m = atoi (argv[5]);
      argv += 5;
	    argc -= 5;
	  }
    else if (strcmp (argv[1], "-memtest") == 0)// memtest parameters
	  {
	    if (argc < 4 || argv[2][0] == '-' || argv[3][0] == '-' )
	    {
	      fprintf (stderr, "can't parse -memtest option\n\n");
	      exit (2);
	    }
	    *cufftbench_s = atoi (argv[2]);
	    *cufftbench_e = atoi (argv[3]);
	    argv += 3;
	    argc -= 3;
	  }
    else if (strcmp (argv[1], "-threads") == 0) // Threads
	  {
	    if (argc < 4 || argv[2][0] == '-' || argv[3][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threads option\n\n");
	      exit (2);
	    }
	    g_th = 0;
      g_thr[0] = atoi (argv[2]);
	    g_thr[1] = atoi (argv[3]);
	    argv += 3;
	    argc -= 3;
	  }
    else if (strcmp (argv[1], "-c") == 0) // checkpoint iteration
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	    g_cpi = atoi (argv[2]);
	    if (g_cpi == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-x") == 0) // report iteration
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -x option\n\n");
	      exit (2);
	    }
	    g_ri = atoi (argv[2]);
	    if (g_ri == 0)
	    {
	      fprintf (stderr, "can't parse -x option\n\n");
	      exit (2);
	    }
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-f") == 0) //fft length
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -f option\n\n");
	      exit (2);
	    }
	    g_fftlen = fft_from_str(argv[2]);
      argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-s") == 0)
	  {
	    g_sf = 1;
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -s option\n\n");
	      exit (2);
	    }
	    sprintf (g_folder, "%s", argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else // the prime exponent
    {
      if (*q != -1 || strcmp (g_input_file, "") != 0 )
      {
        fprintf (stderr, "can't parse options\n\n");
        exit (2);
      }
      int derp = atoi (argv[1]);
      if (derp == 0)
      {
        sprintf (g_input_file, "%s", argv[1]);
      }
      else *q = derp;
      argv++;
      argc--;
    }
  }
}

int interact(int n)
{
  int c = getchar ();
  int k;
  int max_threads = (int) g_dev.maxThreadsPerBlock;

  switch( c )
  {
    case 'p' :
                g_pf ^= 1;
                printf (" -- polite = %d\n", g_pf * g_po);
                break;
    case 'x' :
                g_sl ^= 1;
                printf (" -- sleep = %d\n", g_sl * g_sv);
                break;
    case 'b' :
                g_bc ^= 1;
                if(g_bc) printf (" -- Using 64 bit rcb and splice kernels.\n");
                else printf (" -- Using 32 bit rcb and splice kernels.\n");
                break;
    case 's' :
                g_sf ^= 1;
                if(g_sf) printf (" -- enabling -s.\n");
                else printf (" -- disabling -s.\n");
                break;
    case 'F' :
                printf(" -- Increasing fft length.\n");
                g_fftlen++;
                return 1;
    case 'f' :
                printf(" -- Decreasing fft length.\n");
                g_fftlen--;
                return 1;
    case 'W' :
                if(g_thr[0] < max_threads && (n % (8 * g_thr[0]) == 0)) g_thr[0] *= 2;
                printf(" -- square threads increased to %d\n", g_thr[0]);
                break;
    case 'w' :
                if(g_thr[0] > 32 && (n / ( 2 * g_thr[0]) <= g_dev.maxGridSize[0])) g_thr[0] /= 2;
                printf(" -- square threads decreased to %d\n", g_thr[0]);
                break;
    case 'E' :
                if(g_thr[1] < max_threads) g_thr[1] *= 2;
                printf(" -- splice threads increased to %d\n", g_thr[1]);
                break;
    case 'e' :
                if(g_thr[1] > 32) g_thr[1] /= 2;
                printf(" -- splice threads decreased to %d\n", g_thr[1]);
                break;
    case 'R' :
                if(g_er < 100) g_er += 5;
                printf(" -- error_reset increased to %d\n", g_er);
                break;
    case 'r' :
                if(g_er > 0) g_er -= 5;
                printf(" -- error_reset decreased to %d\n", g_er);
                break;
    case 'T' :
                k = g_cpi;
                while(k % 10 == 0) k /= 10;
                if(k == 2) g_cpi *= 2.5;
                else g_cpi *= 2;
                printf(" -- checkpoint_iter increased to %d\n", g_cpi);
                break;
    case 't' :
                k = g_cpi;
                while (k % 10 == 0) k /= 10;
                if (k == 5) g_cpi /= 2.5;
                else if (g_cpi > 1) g_cpi /= 2;
                printf(" -- checkpoint_iter decreased to %d\n", g_cpi);
                break;
    case 'Y' :
                k = g_ri;
                while(k % 10 == 0) k /= 10;
                if(k == 2) g_ri *= 2.5;
                else g_ri *= 2;
                printf(" -- report_iter increased to %d\n", g_ri);
                break;
    case 'y' :
                k = g_ri;
                while (k % 10 == 0) k /= 10;
                if (k == 5) g_ri /= 2.5;
                else if (g_ri > 1) g_ri /= 2;
                printf(" -- report_iter decreased to %d\n", g_ri);
                break;
    case 'U' :
                k = g_ei;
                while(k % 10 == 0) k /= 10;
                if(k == 2) g_ei *= 2.5;
                else g_ei *= 2;
                printf(" -- error_iter increased to %d\n", g_ei);
                return 4;
    case 'u' :
                k = g_ei;
                while (k % 10 == 0) k /= 10;
                if (k == 5) g_ei /= 2.5;
                else if (g_ei > 1) g_ei /= 2;
                printf(" -- error_iter decreased to %d\n", g_ei);
                return 4;
    case 'Q' :
                k = g_sv;
                while(k % 10 == 0) k /= 10;
                if(k == 2) g_sv *= 2.5;
                else g_sv *= 2;
                printf(" -- sleep value increased to %d\n", g_sv);
                break;
    case 'q' :
                k = g_sv;
                while (k % 10 == 0) k /= 10;
                if (k == 5) g_sv /= 2.5;
                else if (g_sv > 1) g_sv /= 2;
                printf(" -- sleep value decreased to %d\n", g_sv);
                break;
    case 'I' :
                if(g_el < 47)
                {
                  g_el++;
                  printf(" -- error_limit increased to %d\n", g_el);
                }
                else printf(" -- error_limit is already 47. Larger values are not allowed.\n");
                return 4;
    case 'i' :
                if(g_el > 20)
                {
                  g_el--;
                  printf(" -- error_limit decreased to %d.\n", g_el);
                }
                else printf(" -- error_limit is already 20. Smaller values are not allowed.\n");
                return 4;
    case 'O' :
                if(g_po == 100)
                {
                  g_po = 1;
                  printf(" -- polite interval cycled back to %d\n", g_po);
                }
                else
                {
                  k = g_po;
                  while(k % 10 == 0) k /= 10;
                  if(k == 2) g_po *= 2.5;
                  else g_po *= 2;
                  printf(" -- polite interval increased to %d\n", g_po);
                }
                break;
    case 'o' :
                if(g_po == 1)
                {
                  g_po = 100;
                  printf(" -- polite interval cycled back to %d\n", g_po);
                }
                else
                {
                  k = g_po;
                  while(k % 10 == 0) k /= 10;
                  if(k == 5) g_po /= 2.5;
                  else g_po /= 2;
                  printf(" -- polite interval decreased to %d\n", g_po);
                }
                break;
    case 'm' :
                IniGetStr(g_INIFILE, "OutputString", g_output_string, "0");
                IniGetStr(g_INIFILE, "OutputHeader", g_output_header, "0");
                IniGetInt(g_INIFILE, "OutputHInterval", &g_output_interval, 0);
                encode_output_options();
                printf(" -- refreshing output format options.\n");
                break;
    case 'n' :
                printf(" -- resetting timer.\n");
                return 2;
    case 'z' :
                printf("  -- fft count                      %d\n", g_fft_count);
                printf("  -- current fft                    %dK\n", g_ffts[g_fftlen] / 1024);
                printf("  -- smallest fft for this exponent %dK\n", g_ffts[g_lbi] / 1024);
                printf("  -- largest fft for this exponent  %dK\n", g_ffts[g_ubi] / 1024);
                printf("  -- square threads                 %d\n", g_thr[0]);
                printf("  -- splice threads                 %d\n", g_thr[1]);
                printf("  -- checkpoint interval            %d\n", g_cpi);
                printf("  -- report interval                %d\n", g_ri);
                printf("  -- error check interval           %d\n", g_ei);
                printf("  -- error reset percent            %d\n", g_er);
                printf("  -- error limit                    %d\n", g_el);
                printf("  -- polite flag                    %d\n", g_pf);
                printf("  -- polite value                   %d\n", g_po);
                printf("  -- sleep flag                     %d\n", g_sl);
                printf("  -- sleep value                    %d\n", g_sv);
                printf("  -- 64 bit carry flag              %d\n", g_bc);
                printf("  -- save all checkpoints flag      %d\n", g_sf);
                printf("  -- device number                  %d\n", g_dn);
                printf("  -- savefile folder                %s\n", g_folder);
                printf("  -- ini file                       %s\n", g_INIFILE);
                printf("  -- input file                     %s\n", g_input_file);
                printf("  -- results file                   %s\n", g_RESULTSFILE);
                break;
    default  :
                break;
  }
  fflush (stdin);
  return 0;
}
