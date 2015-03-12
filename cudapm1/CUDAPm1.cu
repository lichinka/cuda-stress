char program[] = "CUDAPm1 v0.20";
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
#include <gmp.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _MSC_VER
//#define stat _stat
#define strncasecmp strnicmp // _strnicmp
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

/************************ definitions ************************************/
/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */
/* global variables needed */
double *g_ttmp, *g_ttp1;
double *g_x, *g_ct;
double *e_data;
double *rp_data;
int *g_xint;

char *size;
int threads1, threads2 = 128, threads3= 128;
float *g_err, g_max_err = 0.0f;
int *g_datai, *g_carryi;
long long int *g_datal, *g_carryl;
cufftHandle plan;
cudaDeviceProp dev;

int fft_count;
int multipliers[250];
int quitting, checkpoint_iter, fftlen, tfdepth=74, llsaved=2, s_f, t_f, r_f, d_f, k_f;
int unused_mem = 100;
int polite, polite_f;
int b1 = 0, g_b1_commandline = 0;
int g_b2 = 0, g_b2_commandline = 0;
int g_d = 0, g_d_commandline = 0;
int g_e = 0;
int g_nrp = 0;
int g_eb1 = 0;
int keep_s1 = 0;

char folder[132];
char input_filename[132], RESULTSFILE[132];
char INIFILE[132] = "CUDAPm1.ini";
char AID[132]; // Assignment key
char s_residue[32];

__constant__ double g_ttp_inc[2];

__constant__ int g_qn[2];

/************************ kernels ************************************/
# define RINT_x86(x) (floor(x+0.5))
# define RINT(x)  __rintd(x)

void set_ttp_inc(double *h_ttp_inc){
    cudaMemcpyToSymbol(g_ttp_inc, h_ttp_inc, 2 * sizeof(double));
}

void set_qn(int *h_qn){
    cudaMemcpyToSymbol(g_qn, h_qn, 2 * sizeof(int));
}

__global__ void square (int n,
                         double *a,
                         double *ct)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  //double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;
      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];
      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;

      xi = 2.0 * ajr * aji;
      xr = (ajr - aji) * (ajr + aji);
      yi = 2.0 * akr * aki;
      yr = (akr - aki) * (akr + aki);

      ajr = xr - yr;
      aji = xi + yi;
      akr = wkr * ajr + wki * aji;
      aki = wkr * aji - wki * ajr;

      a[j] = xr - akr;
      a[1 + j] = aki - xi;
      a[nminusj] = yr + akr;
      a[1 + nminusj] =  aki - yi;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      a[0] = xr * xr + xi * xi;
      a[1] = -xr * xi - xi * xr;
      xr = a[0 + m];
      xi = a[1 + m];
      a[1 + m] = -xr * xi - xi * xr;
      a[0 + m] = xr * xr - xi * xi;
    }
}

__global__ void square1 (int n,
                         double *b,
                         double *a,
                         double *ct)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;
	    wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];
      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];

      new_aji = 2.0 * ajr * aji;
      new_ajr = (ajr - aji) * (ajr + aji);
      new_aki = 2.0 * akr * aki;
      new_akr = (akr - aki) * (akr + aki);

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;

      b[j] = new_ajr - yr;
      b[1 + j] = yi - new_aji;
      b[nminusj] = new_akr + yr;
      b[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      b[0] = xr * xr + xi * xi;
      b[1] = -xr * xi - xi * xr;
      xr = a[0 + m];
      xi = a[1 + m];
      b[1 + m] = -xr * xi - xi * xr;
      b[0 + m] = xr * xr - xi * xi;
    }
}

__global__ void mult2 (double *g_out,
                         double *a,
                         double *b,
                         double *ct,
                         int n)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];
      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;
      xr = b[j];
      xi = b[1 + j];
      yr = b[nminusj];
      yi = b[1 + nminusj];

      new_aji = ajr * xi + xr * aji;
      new_ajr = ajr * xr - aji * xi;

      new_aki = akr * yi + yr * aki;
      new_akr = akr * yr - aki * yi;

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;

      g_out[j] = new_ajr - yr;
      g_out[1 + j] = yi - new_aji;
      g_out[nminusj] = new_akr + yr;
      g_out[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      yr = b[0];
      yi = b[1];
      g_out[0] = xr * yr + xi * yi;
      g_out[1] = -xr * yi - xi * yr;
      xr = a[0 + m];
      xi = a[1 + m];
      yr = b[0 + m];
      yi = b[1 + m];
      g_out[1 + m] = -xr * yi - xi * yr;
      g_out[0 + m] = xr * yr - xi * yi;
    }
}

__global__ void mult3 (double *g_out,
                         double *a,
                         double *b,
                         double *ct,
                         int n)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki, bjr, bji, bkr, bki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];

      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;

      bjr = b[j];
      bji = b[1 + j];
      bkr = b[nminusj];
      bki = b[1 + nminusj];
      xr = bjr - bkr;
      xi = bji + bki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      bjr -= yr;
      bji -= yi;
      bkr += yr;
      bki -= yi;

      new_aji = ajr * bji + bjr * aji;
      new_ajr = ajr * bjr - aji * bji;
      new_aki = akr * bki + bkr * aki;
      new_akr = akr * bkr - aki * bki;

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;
      g_out[j] = new_ajr - yr;
      g_out[1 + j] = yi - new_aji;
      g_out[nminusj] = new_akr + yr;
      g_out[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      yr = b[0];
      yi = b[1];
      g_out[0] = xr * yr + xi * yi;
      g_out[1] = -xr * yi - xi * yr;
      xr = a[0 + m];
      xi = a[1 + m];
      yr = b[0 + m];
      yi = b[1 + m];
      g_out[1 + m] = -xr * yi - xi * yr;
      g_out[0 + m] = xr * yr - xi * yi;
    }
}

__global__ void sub_mul (double *g_out,
                         double *a,
                         double *b1,
                         double *b2,
                         double *ct,
                         int n)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki, bjr, bji, bkr, bki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];

      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;

      bjr = b1[j] - b2[j];
      bji = b1[1 + j] - b2[1 + j];
      bkr = b1[nminusj] - b2[nminusj];
      bki = b1[1 + nminusj] - b2[1 + nminusj];

      new_aji = ajr * bji + bjr * aji;
      new_ajr = ajr * bjr - aji * bji;
      new_aki = akr * bki + bkr * aki;
      new_akr = akr * bkr - aki * bki;

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;
      g_out[j] = new_ajr - yr;
      g_out[1 + j] = yi - new_aji;
      g_out[nminusj] = new_akr + yr;
      g_out[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      yr = b1[0] - b2[0];
      yi = b1[1] - b2[1];
      g_out[0] = xr * yr + xi * yi;
      g_out[1] = -xr * yi - xi * yr;
      xr = a[0 + m];
      xi = a[1 + m];
      yr = b1[0 + m] - b2[0 + m];
      yi = b1[1 + m] - b2[1 + m];
      g_out[1 + m] = -xr * yi - xi * yr;
      g_out[0 + m] = xr * yr - xi * yi;
    }
}

__global__ void pre_mul (int n,
                           double *a,
                           double *ct)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
  {
    int nminusj = n - j;

    wkr = 0.5 - ct[nc - j2];
    wki = ct[j2];
    ajr = a[j];
    aji = a[1 + j];
    akr = a[nminusj];
    aki = a[1 + nminusj];
    xr = ajr - akr;
    xi = aji + aki;
    yr = wkr * xr - wki * xi;
    yi = wkr * xi + wki * xr;
    ajr -= yr;
    aji -= yi;
    akr += yr;
    aki -= yi;
    a[j] = ajr;
    a[1 + j] = aji;
    a[nminusj] = akr;
    a[1 + nminusj] =  aki;
  }
}

__device__ static double __rintd (double z)
{
  double y;

  asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (z));
  return (y);
}

__global__ void apply_weights (double *g_out,
                                 int *g_in,
                                 double *g_ttmp)
{
  int val[2], test = 1;
  double ttp_temp[2];
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;

  val[0] = g_in[index];
  val[1] = g_in[index + 1];
  ttp_temp[0] = g_ttmp[index];
  ttp_temp[1] = g_ttmp[index + 1];
  if(ttp_temp[0] < 0.0) test = 0;
  if(ttp_temp[1] < 0.0) ttp_temp[1] = -ttp_temp[1];
  g_out[index + 1] = (double) val[1] * ttp_temp[1];
  ttp_temp[1] *= -g_ttp_inc[test];
  g_out[index] = (double) val[0] * ttp_temp[1];
}

__global__ void norm1a (double *g_in,
                         int *g_data,
                         int *g_xint,
                         double *g_ttmp,
                         int *g_carry,
		                     volatile float *g_err,
		                     float maxerr,
		                     int g_err_flag)
{
  long long int bigint[2];
  int val[2], numbits[2] = {g_qn[0],g_qn[0]}, mask[2], shifted_carry;
  double ttp_temp;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  const int index1 = blockIdx.x << 1;
  __shared__ int carry[1024 + 1];

  {
    double tval[2], trint[2];
    float ferr[2];

    tval[0] = g_ttmp[index];
    ttp_temp = g_ttmp[index + 1];
    trint[0] = g_in[index];
    trint[1] = g_in[index + 1];
    if(tval[0] < 0.0)
    {
      numbits[0]++;
      tval[0] = -tval[0];
    }
    if(ttp_temp < 0.0)
    {
      numbits[1]++;
      ttp_temp = -ttp_temp;
    }
    tval[1] = tval[0] * g_ttp_inc[numbits[0] == g_qn[0]];
    tval[0] = trint[0] * tval[0];
    tval[1] = trint[1] * tval[1];
    trint[0] = RINT (tval[0]);
    ferr[0] = tval[0] - trint[0];
    ferr[0] = fabs (ferr[0]);
    bigint[0] = (long long int) trint[0];
    trint[1] = RINT (tval[1]);
    ferr[1] = tval[1] - trint[1];
    ferr[1] = fabs (ferr[1]);
    bigint[1] = (long long int) trint[1];
    mask[0] = -1 << numbits[0];
    mask[1] = -1 << numbits[1];
    if(ferr[0] < ferr[1]) ferr[0] = ferr[1];
    if (ferr[0] > maxerr) atomicMax((int*) g_err, __float_as_int(ferr[0]));
  }
  val[1] = ((int) bigint[1]) & ~mask[1];
  carry[threadIdx.x + 1] = (int) (bigint[1] >> numbits[1]);
  val[0] = ((int) bigint[0]) & ~mask[0];
  val[1] += (int) (bigint[0] >> numbits[0]);
  __syncthreads ();

  if (threadIdx.x) val[0] += carry[threadIdx.x];
  shifted_carry = val[1] - (mask[1] >> 1);
  val[1] = val[1] - (shifted_carry & mask[1]);
  carry[threadIdx.x] = shifted_carry >> numbits[1];
  shifted_carry = val[0] - (mask[0] >> 1);
  val[0] = val[0] - (shifted_carry & mask[0]);
  val[1] += shifted_carry >> numbits[0];
  __syncthreads ();

  if (threadIdx.x == (blockDim.x - 1))
  {
    if (blockIdx.x == gridDim.x - 1) g_carry[0] = carry[threadIdx.x + 1] + carry[threadIdx.x];
    else   g_carry[blockIdx.x + 1] =  carry[threadIdx.x + 1] + carry[threadIdx.x];
  }

  if (threadIdx.x)
  {
    val[0] += carry[threadIdx.x - 1];
    {
        g_in[index + 1] = (double) val[1] * ttp_temp;
        ttp_temp *= -g_ttp_inc[numbits[0] == g_qn[0]];
        g_in[index] = (double) val[0] * ttp_temp;
    }
    if(g_err_flag)
    {
      g_xint[index + 1] = val[1];
      g_xint[index] = val[0];
    }
  }
  else
  {
    g_data[index1] = val[0];
    g_data[index1 + 1] = val[1];
  }
}

__global__ void norm1b (double *g_in,
                         long long int *g_data,
                         int *g_xint,
                         double *g_ttmp,
                         long long int *g_carry,
		                     volatile float *g_err,
		                     float maxerr,
		                     int g_err_flag)
{
  long long int bigint[2], shifted_carry;
  int  numbits[2] = {g_qn[0],g_qn[0]}, mask[2];
  double ttp_temp;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  const int index1 = blockIdx.x << 1;
  __shared__ long long int carry[1024 + 1];

  {
    double tval[2], trint[2];
    float ferr[2];

    tval[0] = g_ttmp[index];
    ttp_temp = g_ttmp[index + 1];
    trint[0] = g_in[index];
    trint[1] = g_in[index + 1];
    if(tval[0] < 0.0)
    {
      numbits[0]++;
      tval[0] = -tval[0];
    }
    if(ttp_temp < 0.0)
    {
      numbits[1]++;
      ttp_temp = -ttp_temp;
    }
    tval[1] = tval[0] * g_ttp_inc[numbits[0] == g_qn[0]];
    tval[0] = trint[0] * tval[0];
    tval[1] = trint[1] * tval[1];
    trint[0] = RINT (tval[0]);
    ferr[0] = tval[0] - trint[0];
    ferr[0] = fabs (ferr[0]);
    bigint[0] = (long long int) trint[0];
    trint[1] = RINT (tval[1]);
    ferr[1] = tval[1] - trint[1];
    ferr[1] = fabs (ferr[1]);
    bigint[1] = (long long int) trint[1];
    mask[0] = -1 << numbits[0];
    mask[1] = -1 << numbits[1];
    if(ferr[0] < ferr[1]) ferr[0] = ferr[1];
    if (ferr[0] > maxerr) atomicMax((int*) g_err, __float_as_int(ferr[0]));
  }
  bigint[0] *= 3;
  bigint[1] *= 3;
  carry[threadIdx.x + 1] = (bigint[1] >> numbits[1]);
  bigint[1] = bigint[1] & ~mask[1];
  bigint[1] += bigint[0] >> numbits[0];
  bigint[0] =  bigint[0] & ~mask[0];
  __syncthreads ();

  if (threadIdx.x) bigint[0] += carry[threadIdx.x];
  shifted_carry = bigint[1] - (mask[1] >> 1);
  bigint[1] = bigint[1] - (shifted_carry & mask[1]);
  carry[threadIdx.x] = shifted_carry >> numbits[1];
  shifted_carry = bigint[0] - (mask[0] >> 1);
  bigint[0] = bigint[0] - (shifted_carry & mask[0]);
  bigint[1] += shifted_carry >> numbits[0];
  __syncthreads ();

  if (threadIdx.x == (blockDim.x - 1))
  {
    if (blockIdx.x == gridDim.x - 1) g_carry[0] = carry[threadIdx.x + 1] + carry[threadIdx.x];
    else   g_carry[blockIdx.x + 1] =  carry[threadIdx.x + 1] + carry[threadIdx.x];
  }

  if (threadIdx.x)
  {
    bigint[0] += carry[threadIdx.x - 1];
    {
        g_in[index + 1] = (double) bigint[1] * ttp_temp;
        ttp_temp *= -g_ttp_inc[numbits[0] == g_qn[0]];
        g_in[index] = (double) bigint[0] * ttp_temp;
    }
    if(g_err_flag)
    {
      g_xint[index + 1] = bigint[1];
      g_xint[index] = bigint[0];
    }
  }
  else
  {
    g_data[index1] = bigint[0];
    g_data[index1 + 1] = bigint[1];
  }
}


__global__ void
norm2a (double *g_x, int *g_xint, int g_N, int threads1, int *g_data, int *g_carry, double *g_ttp1, int g_err_flag)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads1 * threadID) << 1;
  int temp0, temp1;
  int mask, shifted_carry, numbits= g_qn[0];
  double temp;

  if (j < g_N)
    {
      temp0 = g_data[threadID1] + g_carry[threadID];
      temp1 = g_data[threadID1 + 1];
      temp = g_ttp1[threadID];
      if(temp < 0.0)
      {
        numbits++;
        temp = -temp;
      }
      mask = -1 << numbits;
      shifted_carry = temp0 - (mask >> 1) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 += (shifted_carry >> numbits);
      {
        g_x[j + 1] = temp1 * temp;
        temp *= -g_ttp_inc[numbits == g_qn[0]];
        g_x[j] = temp0 * temp;
      }
      if(g_err_flag)
      {
        g_xint[j + 1] = temp1;
        g_xint[j] = temp0;
      }
    }
}

__global__ void
norm2b (double *g_x, int *g_xint, int g_N, int threads1, long long int *g_data, long long int *g_carry, double *g_ttp1, int g_err_flag)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads1 * threadID) << 1;
  long long int shifted_carry, temp0, temp1;
  int mask,  numbits = g_qn[0];
  double temp;

  if (j < g_N)
    {
      temp0 = g_data[threadID1] + g_carry[threadID];
      temp1 = g_data[threadID1 + 1];
      temp = g_ttp1[threadID];
      if(temp < 0.0)
      {
        numbits++;
        temp = -temp;
      }
      mask = -1 << numbits;
      shifted_carry = temp0 - (mask >> 1) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 = temp1 + (shifted_carry >> numbits);
      g_x[j + 1] = temp1 * temp;
      temp *= -g_ttp_inc[numbits == g_qn[0]];
      g_x[j] = temp0 * temp;
      if(g_err_flag)
      {
        g_xint[j + 1] = temp1;
        g_xint[j] = temp0;
      }
    }
}

__global__ void
copy_kernel (double *save, double *y)
{
  const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  save[threadID] = y[threadID];
  save[threadID + 1] = y[threadID + 1];
}

/****************************************************************************
 *                                Erato                                     *
 ***************************************************************************/
//Many thanks to Ben Buhrow.

typedef unsigned char u8;
typedef unsigned int uint32;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef long long unsigned int uint64;

const int threadsPerBlock = 256;
const uint32 block_size = 8192;
const int startprime = 8;


__constant__ uint32 _step5[5] = 	{2418280706,604570176,151142544,37785636,1083188233};
__constant__ uint32 _step7[7] = 	{1107363844,69210240,2151809288,134488080,
									276840961,17302560,537952322};
__constant__ uint32 _step11[11] = 	{33816584,1073774848,135266336,132096,541065345,
									528384,2164261380,2113536,67110928,8454146,268443712};
__constant__ uint32 _step13[13] = 	{1075838992,16809984,262656,536875016,8388672,
									67239937,1050624,2147500064,33554688,268959748,4202496,
									65664,134218754};
__constant__ uint32 _step17[17] = 	{268435488,1073741952,512,2049,8196,32784,131136,
									524544,2098176,8392704,33570816,134283264,537133056,
									2148532224,4194304,16777218,67108872};
__constant__ uint32 _step19[19] = 	{2147483712,4096,262176,16779264,1073872896,8388608,
									536870928,1024,65544,4194816,268468224,2097152,134217732,
									256,16386,1048704,67117056,524288,33554433};

__global__ static void SegSieve(uint32 *primes, int maxp, int nump, uint32 N, uint8 *results)
{
	/*
	expect as input a set of primes to sieve with, how many of those primes there are (maxp)
	how many primes each thread will be responsible for (nump), and the maximum index
	that we need to worry about for the requested sieve interval.  Also, an array into
	which we can put this block's count of primes.

	This routine implements a segmented sieve using a wheel mod 6.  Each thread block on the gpu
	sieves a different segment of the number line.  Each thread within each block simultaneously
	sieves a small set of primes, marking composites within shared memory.  There is no memory
	contention between threads because the marking process is write only.  Because each thread
	block starts at a different part of the number line, a small amount of computation must
	be done for each prime prior to sieving to figure out where to start.  After sieving
	is done, each thread counts primes in part of the shared memory space; the final count
	is returned in the provided array for each block.  The host cpu will do the final sum
	over blocks.  Note, it would not be much more difficult to compute and return the primes
	in the block instead of just the count, but it would be slower due to the extra
	memory transfer required.
	*/

	uint32 i,j,k;
	uint32 maxID = (N + 1) / 3;
	uint32 bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint32 range = block_size / threadsPerBlock;
	__shared__ uint8 locsieve[block_size];
	__shared__ uint32 bitsieve[block_size / 32];


	// everyone init the array.
	if ((bid+1)*block_size > maxID)
	{
		for (j=threadIdx.x * range, k=0; k<range; k++)
		{
			// we're counting hits in the kernel as well, so clear the bytes representing primes > N
			if ((bid * block_size + j + k) < maxID)
				locsieve[j+k] = 1;
			else
				locsieve[j+k] = 0;
		}
	}
	else
	{
		for (j=threadIdx.x * range/4, k=0; k<range/4; k++)
		{
			((uint32 *) locsieve)[j+k] = 0x01010101;
		}
	}

	// the smallest primes are dealt with a bit differently.  They are sieved in a separate
	// shared memory space in a packed bit array.  constant memory holds pre-computed
	// information about where each prime lands within a given 32 bit region.  each thread
	// in the block will use this info to simultaneously sieve a small portion of the
	// packed bit array (that way we make use of the broadcast capabilities of constant memory).
	// When counting or computing primes, we then have to check both the packed bit array as
	// well as the regular byte array, but overall it is a win to greatly speed up the
	// sieving of the smallest primes.

	// compute starting offset for prime 5:
	i = (bid * 256 + threadIdx.x) % 5;
	// then sieve prime 5 in the bit array
	bitsieve[threadIdx.x] = _step5[i];

	// compute starting offset for prime 7:
	i = (bid * 256 + threadIdx.x) % 7;
	// then sieve prime 7 in the bit array
	bitsieve[threadIdx.x] |= _step7[i];

	// compute starting offset for prime 11:
	i = (bid * 256 + threadIdx.x) % 11;
	// then sieve prime 11 in the bit array
	bitsieve[threadIdx.x] |= _step11[i];

	// compute starting offset for prime 13:
	i = (bid * 256 + threadIdx.x) % 13;
	// then sieve prime 13 in the bit array
	bitsieve[threadIdx.x] |= _step13[i];

	// compute starting offset for prime 17:
	i = (bid * 256 + threadIdx.x) % 17;
	// then sieve prime 17 in the bit array
	bitsieve[threadIdx.x] |= _step17[i];

	// compute starting offset for prime 19:
	i = (bid * 256 + threadIdx.x) % 19;
	// then sieve prime 19 in the bit array
	bitsieve[threadIdx.x] |= _step19[i];


	// regroup before sieving
	__syncthreads();

	// now sieve the array
	for (j=0; j<nump; j++)
	{
		int pid = (j * threadsPerBlock) + threadIdx.x + startprime;

		if (pid < maxp)
		{
			uint32 p = primes[pid];
			uint32 pstart = p/3;
			uint32 p2 = 2*p;
			uint32 block_start = bid * block_size;
			uint32 start_offset;
			uint32 s[2];

			// the wheel sieve with all multiples of 2 and 3 removed from the array is equivalent to
			// alternately stepping through the number line by (p+2)*mult, (p-2)*mult,
			// where mult = (p+1)/6
			s[0] = p+(2*((p+1)/6));
			s[1] = p-(2*((p+1)/6));

			// compute the starting location of this prime in this block
			if ((bid == 0) || (pstart >= block_start))
			{
				// start one increment past the starting value of p/3, since
				// we want to count the prime itself as a prime.
				start_offset = pstart + s[0] - block_start;
				k = 1;
			}
			else
			{
				// measure how far the start of this block is from where the prime first landed,
				// as well as how many complete (+2/-2) steps it would need to take
				// to cover that distance
				uint32 dist = (block_start - pstart);
				uint32 steps = dist / p2;

				if ((dist % p2) == 0)
				{
					// if the number of steps is exact, then we hit the start
					// of this block exactly, and we start below with the +2 step.
					start_offset = 0;
					k = 0;
				}
				else
				{
					uint32 inc = pstart + steps * p2 + s[0];
					if (inc >= block_start)
					{
						// if the prime reaches into this block on the first stride,
						// then start below with the -2 step
						start_offset = inc - block_start;
						k = 1;
					}
					else
					{
						// we need both +2 and -2 strides to get into the block,
						// so start below with the +2 stride.
						start_offset = inc + s[1] - block_start;
						k = 0;
					}
				}
			}

			// unroll the loop for the smallest primes.
			if (p < 1024)
			{
				uint32 stop = block_size - (2 * p * 4);

				if (k == 0)
				{
					for(i=start_offset ;i < stop; i+=8*p)
					{
						locsieve[i] = 0;
						locsieve[i+s[0]] = 0;
						locsieve[i+p2] = 0;
						locsieve[i+p2+s[0]] = 0;
						locsieve[i+4*p] = 0;
						locsieve[i+4*p+s[0]] = 0;
						locsieve[i+6*p] = 0;
						locsieve[i+6*p+s[0]] = 0;
					}
				}
				else
				{
					for(i=start_offset ;i < stop; i+=8*p)
					{
						locsieve[i] = 0;
						locsieve[i+s[1]] = 0;
						locsieve[i+p2] = 0;
						locsieve[i+p2+s[1]] = 0;
						locsieve[i+4*p] = 0;
						locsieve[i+4*p+s[1]] = 0;
						locsieve[i+6*p] = 0;
						locsieve[i+6*p+s[1]] = 0;
					}
				}
			}
			else
				i=start_offset;

			// alternate stepping between the large and small strides this prime takes.
			for( ;i < block_size; k = !k)
			{
				locsieve[i] = 0;
				i += s[k];
			}
		}
	}

	// regroup before counting
	__syncthreads();

	for (j=threadIdx.x * range, k=0; k<range; k++)
		locsieve[j + k] = (locsieve[j+k] & ((bitsieve[(j+k) >> 5] & (1 << ((j+k) & 31))) == 0));

	__syncthreads();

  if(threadIdx.x == 0)
    for (k=0; k < block_size; k++)
    {
      j = ((bid * block_size + k) * 3 + 1) >> 1;
      if(j < N >> 1) results[j] = locsieve[k];
    }
}

uint32 tiny_soe(uint32 limit, uint32 *primes)
{
	//simple sieve of erathosthenes for small limits - not efficient
	//for large limits.
	uint8 *flags;
	uint16 prime;
	uint32 i,j;
	int it;

	//allocate flags
	flags = (uint8 *)malloc(limit/2 * sizeof(uint8));
	if (flags == NULL)
		printf("error allocating flags\n");
	memset(flags,1,limit/2);

	//find the sieving primes, don't bother with offsets, we'll need to find those
	//separately for each line in the main sieve.
	primes[0] = 2;
	it=1;

	//sieve using primes less than the sqrt of the desired limit
	//flags are created only for odd numbers (mod2)
	for (i=1;i<(uint32)(sqrt((double)limit)/2+1);i++)
	{
		if (flags[i] > 0)
		{
			prime = (uint32)(2*i + 1);
			for (j=i+prime;j<limit/2;j+=prime)
				flags[j]=0;

			primes[it]=prime;
			it++;
		}
	}

	//now find the rest of the prime flags and compute the sieving primes
	for (;i<limit/2;i++)
	{
		if (flags[i] == 1)
		{
			primes[it] = (uint32)(2*i + 1);
			it++;
		}
	}

	free(flags);
	return it;
}

int gtpr(int n, uint8* bprimes)
{
	uint32 Nsmall = (uint32) sqrt((double) n);
	int numblocks;
	int primes_per_thread;
	uint32* primes;
	uint32* device_primes;
	uint32 np;
	uint8* results;

	// find seed primes
	primes = (uint32*)malloc(Nsmall*sizeof(uint32));
	np = tiny_soe(Nsmall, primes);

	// put the primes on the device
	cudaMalloc((void**) &device_primes, sizeof(uint32) * np);
	cudaMemcpy(device_primes, primes, sizeof(uint32)*np, cudaMemcpyHostToDevice);

	// compute how many whole blocks we have to sieve and how many primes each
	// thread will be responsible for.
	numblocks = (n / 3 / block_size + 1);
	primes_per_thread = ((np - startprime) + threadsPerBlock - 1) / threadsPerBlock;
	dim3 grid((uint32)sqrt((double)numblocks)+1,(uint32)sqrt((double)numblocks)+1);

	cudaMalloc((void**) &results, sizeof(uint8) * (n >> 1));
	cudaMemset(results, 0, sizeof(uint8) * (n >> 1));

	SegSieve<<<grid, threadsPerBlock, 0>>>(device_primes, np, primes_per_thread, n, results);

	cudaThreadSynchronize();
	cudaMemcpy (bprimes, results, sizeof (uint8) * (n >> 1), cudaMemcpyDeviceToHost);
	cudaFree(device_primes);
	cudaFree(results);
	free(primes);

	return 0;
}

/**************************************************************
 *
 *      FFT and other related Functions
 *
 **************************************************************/
/* rint is not ANSI compatible, so we need a definition for
 * WIN32 and other platforms with rint.
 * Also we use that to write the trick to rint()
 */

/****************************************************************************
 *           Lucas Test - specific routines                                 *
 ***************************************************************************/
void reset_err(float* maxerr, float value)
{
  *maxerr *= value;
  cutilSafeCall (cudaMemcpy (g_err, maxerr, sizeof (float), cudaMemcpyHostToDevice));
}



float
lucas_square (/*double *x,*/ int q, int n, int iter, int last, float* maxerr, int error_flag, int bit, int stage, int chkpt)
{
  float terr = 0.0;

  if (iter < 100 && iter % 10 == 0)
  {
    cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
  }
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  square <<< n / (4 * threads2), threads2 >>> (n, g_x, g_ct);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));

  if(!bit)
  {
    norm1a <<<n / (2 * threads1), threads1 >>> (g_x, g_datai, g_xint, g_ttmp, g_carryi, g_err, *maxerr, chkpt);
    norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_x, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, chkpt);
  }
  else
  {
    norm1b <<<n / (2 * threads1), threads1 >>> (g_x, g_datal, g_xint, g_ttmp, g_carryl, g_err, *maxerr, chkpt);
    norm2b <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_x, g_xint, n, threads1, g_datal, g_carryl, g_ttp1, chkpt);
  }

  if (error_flag)
  {
    cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
  }
  else if (polite_f && (iter % polite) == 0) cutilSafeThreadSync();
  return (terr);
}

void init_x_int(int *x_int, unsigned *x_packed, int q, int n, int *stage)
{
  int j;

  if(*stage == 0)
  {
    *stage = 1;
    for(j = 0; j < n; j++) x_int[j] = 0;
    x_int[0] = 1;
    if(x_packed)
    {
      for(j = 0; j < (q + 31) /32; j++) x_packed[j] = 0;
      x_packed[0] = 1;
    }
  }
  cudaMemcpy (g_xint, x_int, sizeof (int) * n , cudaMemcpyHostToDevice);
}

void E_init_d(double *g, double value, int n)
{
  double x[1] = {value};

  cutilSafeCall (cudaMemset (g, 0.0, sizeof (double) * n));
  cudaMemcpy (g, x, sizeof (double) , cudaMemcpyHostToDevice);
}

void E_pre_mul(double *g_out, double *g_in, int n, int fft_f)
{
  if(fft_f) cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_in, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  pre_mul <<<n / (4 * threads2), threads2>>> (n, g_out, g_ct);
}

void E_mul(double *g_out, double *g_in1, double *g_in2, int n, float err, int fft_f)
{

  if(fft_f) cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_in1, (cufftDoubleComplex *) g_in1, CUFFT_INVERSE));
  mult3 <<<n / (4 * threads2), threads2>>> (g_out, g_in1, g_in2, g_ct, n);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, err, 0);
  norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
}

void E_sub_mul(double *g_out, double *g_in1, double *g_in2, double *g_in3, int n, float err, int chkpt)
{

  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_in1, (cufftDoubleComplex *) g_in1, CUFFT_INVERSE));
  sub_mul <<<n / (4 * threads2), threads2>>> (g_out, g_in1, g_in2, g_in3, g_ct, n);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, &g_xint[n], g_ttmp, g_carryi, g_err, err, chkpt);
  norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, &g_xint[n], n, threads1, g_datai, g_carryi, g_ttp1, chkpt);
}

void E_half_mul(double *g_out, double *g_in1, double *g_in2, int n, float err)
{

  mult2 <<<n / (4 * threads2), threads2>>> (g_out, g_in1, g_in2, g_ct, n);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, err, 0);
  norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
}

int E_to_the_p(double *g_out, double *g_in, mpz_t p, int n, int trans, float *err)
{
// Assume g_in is premultiplied

  int last, j;
  int checksync = trans / (2 * 50) * 2 * 50;
  int checkerror = trans / (200) * 200;
  int checksave = trans / (2 * checkpoint_iter) * 2 * checkpoint_iter;
  int sync = 1;

  last = mpz_sizeinbase (p, 2);
  if (last == 1)
  {
    E_init_d(g_out, 1.0, n);
    if(mpz_tstbit (p, last - 1))
    {
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
      mult2 <<< n / (4 * threads2), threads2 >>> (g_out, g_out, g_in, g_ct, n);
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
      norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err, 0);
      norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
      trans += 2;
    }
    return trans;
  }
  square1 <<< n / (4 * threads2), threads2 >>> (n, g_out, g_in, g_ct);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err, 0);
  norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
  trans += 2;
  cutilSafeCall (cudaMemcpy (err, g_err, sizeof (float), cudaMemcpyDeviceToHost));
  if(mpz_tstbit (p, last - 2))
  {
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
    mult2 <<< n / (4 * threads2), threads2 >>> (g_out, g_out, g_in, g_ct, n);
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
    norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err, 0);
    norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
    trans += 2;
  }

  for(j = 3; j <= last && !quitting; j++)
  {
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
    square <<< n / (4 * threads2), threads2 >>> (n, g_out, g_ct);
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
    norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err, 0);
    norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
    trans += 2;
    if(mpz_tstbit (p, last - j))
    {
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
      mult2 <<< n / (4 * threads2), threads2 >>> (g_out, g_out, g_in, g_ct, n);
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
      norm1a <<<n / (2 * threads1), threads1 >>> (g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err, 0);
      norm2a <<< (n / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_out, g_xint, n, threads1, g_datai, g_carryi, g_ttp1, 0);
      trans += 2;
    }
    if(trans - checkerror > 200)
    {
      sync = 0;
      checkerror += 200;
      cutilSafeCall (cudaMemcpy (err, g_err, sizeof (float), cudaMemcpyDeviceToHost));
      if(*err > 0.4) quitting = 2;
    }
    if(trans - checksave > 2 * checkpoint_iter)
    {
      checksave += 2 * checkpoint_iter;
      reset_err(err, 0.85f);
    }
    if(sync && polite_f && trans - checksync > 2 * polite)
    {
      checksync += 2 * polite;
      cutilSafeThreadSync();
    }
    sync = 1;
    fflush(NULL);
  }
  return trans;
}

/* -------- initializing routines -------- */
void
makect (int nc, double *c)
{
  int j;
  double d = (double) (nc << 1);

  for (j = 1; j <= nc; j++) c[j] = 0.5 * cospi (j / d);
}

void alloc_gpu_mem(int n)
{
  cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));
  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * n ));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * n / 4));
  cutilSafeCall (cudaMalloc ((void **) &g_xint, sizeof (int) * 2 * n ));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * 2 * n / threads1));
  cutilSafeCall (cudaMalloc ((void **) &g_datai, sizeof (int) * 2 * n / threads1));
  cutilSafeCall (cudaMalloc ((void **) &g_datal, sizeof (long long int) * 2 * n / threads1));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  cutilSafeCall (cudaMalloc ((void **) &g_carryl, sizeof (long long int) * n / threads1));
  cutilSafeCall (cudaMalloc ((void **) &g_carryi, sizeof (int) * n / threads1));
}

void write_gpu_data(int q, int n)
{
  double *s_ttmp, *s_ttp1, *s_ct;
  int i, j, qn = q / n, b = q % n;
  int a, c, bj;
  double *h_ttp_inc;
  int *h_qn;

  s_ct = (double *) malloc (sizeof (double) * (n / 4));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_ttp1 = (double *) malloc (sizeof (double) * 2 * (n / threads1));
  size = (char *) malloc (sizeof (char) * n);
  h_ttp_inc = (double *) malloc (sizeof (double) * 2);
  h_qn = (int *) malloc (sizeof (int) * 2);


  c = n - b;
  bj = 0;
  for (j = 1; j < n; j++)
  {
    bj += b;
    bj %= n;
    a = bj - n;
    if(j % 2 == 0) s_ttmp[j] = exp2 (a / (double) n) * 2.0 / n;
    else s_ttmp[j] = exp2 (-a / (double) n);
    size[j] = (bj >= c);
    if(size[j]) s_ttmp[j] = -s_ttmp[j];
  }
  size[0] = 1;
  s_ttmp[0] = -2.0 / n;
  size[n-1] = 0;
  s_ttmp[n-1] = -s_ttmp[n-1];

  for (i = 0, j = 0; i < n ; i += 2 * threads1)
  {
      s_ttp1[j] = abs(s_ttmp[i + 1]);
      if(size[i]) s_ttp1[j] = -s_ttp1[j];
      j++;
  }

  makect (n / 4, s_ct);

  h_ttp_inc[0] = -exp2((b-n) / (double) n);
  h_ttp_inc[1] = -exp2(b / (double) n);
  set_ttp_inc(h_ttp_inc);
  h_qn[0] = qn;
  h_qn[1] = n;
  set_qn(h_qn);


  cutilSafeCall(cudaMemcpy (g_ttmp, s_ttmp, sizeof (double) * n, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy (g_ttp1, s_ttp1, sizeof (double) * 2 * n / threads1, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy (g_ct, s_ct, sizeof (double) * (n / 4), cudaMemcpyHostToDevice));

  free ((char *) s_ct);
  free ((char *) s_ttmp);
  free ((char *) s_ttp1);
  free ((char *) h_ttp_inc);
  free ((char *) h_qn);
}

void free_host (int *x_int)
{
  free ((char *) size);
  free ((char *) x_int);
}

void free_gpu(void)
{
  cufftSafeCall (cufftDestroy (plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_xint));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_datai));
  cutilSafeCall (cudaFree ((char *) g_datal));
  cutilSafeCall (cudaFree ((char *) g_carryl));
  cutilSafeCall (cudaFree ((char *) g_carryi));
}

void close_lucas (int *x_int)
{
  free_host(x_int);
  free_gpu();
}



/**************************************************************************
 *                                                                        *
 *       End LL/GPU Functions, Begin Utility/CPU Functions                *
 *                                                                        *
 **************************************************************************/
void init_threads(int n)
{
  FILE *threads;
  char buf[132];
  char threadfile[32];
  int no_file = 0, no_entry = 1;
  int th1 = 0, th2 = 0, th3 = 0;
  int temp_n;

  sprintf (threadfile, "%s threads.txt", dev.name);
  threads = fopen(threadfile, "r");
  if(threads)
  {
    while(fgets(buf, 132, threads) != NULL)
    {
      sscanf(buf, "%d %d %d %d", &temp_n, &th1, &th2, &th3);
      if(n == temp_n * 1024)
      {
        threads1 = th1;
        threads2 = th2;
        threads3 = th3;
        no_entry = 0;
      }
    }
  }
  else no_file = 1;
  if(no_file || no_entry)
  {
    if(no_file) printf("No %s file found. Using default thread sizes.\n", threadfile);
    else if(no_entry) printf("No entry for fft = %dk found. Using default thread sizes.\n", n / 1024);
    printf("For optimal thread selection, please run\n");
    printf("./CUDAPm1 -cufftbench %d %d r\n", n / 1024, n / 1024);
    printf("for some small r, 0 < r < 6 e.g.\n");
    fflush(NULL);
  }
  return;
}

int init_ffts()
{
  //#define COUNT 139
  FILE *fft;
  char buf[132];
  int next_fft, j = 0, i = 0;
  int first_found = 0;
  #define COUNT 160
  int default_mult[COUNT] = {  //this batch from GTX570 timings
                                2,     8,    10,    14,    16,    18,    20,    32,    36,    42,
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
                            52488, 54432, 55296, 56000, 57344, 60750, 62500, 64000, 64800, 65536 };

  char fftfile[32];

  sprintf (fftfile, "%s fft.txt", dev.name);
  fft = fopen(fftfile, "r");
  if(!fft)
  {
    printf("No %s file found. Using default fft lengths.\n", fftfile);
    printf("For optimal fft selection, please run\n");
    printf("./CUDAPm1 -cufftbench 1 8192 r\n");
    printf("for some small r, 0 < r < 6 e.g.\n");
    fflush(NULL);
    for(j = 0; j < COUNT; j++) multipliers[j] = default_mult[j];
  }
  else
  {
    while(fgets(buf, 132, fft) != NULL)
    {
      int le = 0;

      sscanf(buf, "%d", &le);
      if(next_fft = atoi(buf))
      {
        if(!first_found)
        {
          while(i < COUNT && default_mult[i] < next_fft)
          {
            multipliers[j] = default_mult[i];
            i++;
            j++;
          }
          multipliers[j] = next_fft;
          j++;
          first_found = 1;
        }
        else
        {
          multipliers[j] = next_fft;
          j++;
        }
      }
    }
    while(default_mult[i] < multipliers[j - 1] && i < COUNT) i++;
    while(i < COUNT)
    {
      multipliers[j] = default_mult[i];
      j++;
      i++;
    }
    fclose(fft);
  }
  return j;
}

int
choose_fft_length (int q, int *index)
{
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */

  if( 0 < *index && *index < fft_count ) return 1024*multipliers[*index];
  else if( *index >= fft_count || q == 0)
  { /* override with manual fftlen passed as arg; set pointer to largest index <= fftlen */
    int len, i;
    for(i = fft_count - 1; i >= 0; i--)
    {
      len = 1024*multipliers[i];
      if( len <= *index )
      {
        *index = i;
        return len; /* not really necessary, but now we could decide to override fftlen with this value */
      }
    }
  }
  else
  { // *index < 0, not override, choose length and set pointer to proper index
    int i;
    int estimate = ceil(1.01 * 0.0000358738168878758 * exp (1.0219834608 * log ((double) q)));

    for(i = 0; i < fft_count; i++)
    {
      if(multipliers[i] >= estimate)
      {
        *index = i;
        printf("Index %d\n",*index);
        return  multipliers[i] * 1024;
      }
    }
  }
  return 0;
}

int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
  char* endptr;
  const char* ptr = str;
  int len, mult = 0;
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
  if( !mult ) { // No K or M, treat as before    (PS The Python else clause on loops I mention in parse.c would be useful here :) )
    mult = 1;
  }
  len = (int) strtoul(str, &endptr, 10)*mult;
  if( endptr != ptr ) { // The K or M must directly follow the num (or the num must extend to the end of the str)
    fprintf (stderr, "can't parse fft length \"%s\"\n\n", str);
    exit (2);
  }
  return len;
}

//From apsen
void
print_time_from_seconds (int sec)
{
  if (sec > 3600)
    {
      printf ("%d", sec / 3600);
      sec %= 3600;
      printf (":%02d", sec / 60);
    }
  else
    printf ("%d", sec / 60);
  sec %= 60;
  printf (":%02d", sec);
}

void
init_device (int device_number)
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
  cudaSetDeviceFlags (cudaDeviceBlockingSync);
  cudaGetDeviceProperties (&dev, device_number);
  // From Iain
  if (dev.major == 1 && dev.minor < 3)
  {
    printf("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
    exit (2);
  }
  if (d_f)
    {
      printf ("------- DEVICE %d -------\n",    device_number);
      printf ("name                %s\n",       dev.name);
      printf ("Compatibility       %d.%d\n",    dev.major, dev.minor);
      printf ("clockRate (MHz)     %d\n",       dev.clockRate/1000);
      printf ("memClockRate (MHz)  %d\n",       dev.memoryClockRate/1000);
#ifdef _MSC_VER
      printf ("totalGlobalMem      %Iu\n",      dev.totalGlobalMem);
#else
      printf ("totalGlobalMem      %zu\n",      dev.totalGlobalMem);
#endif
#ifdef _MSC_VER
      printf ("totalConstMem       %Iu\n",      dev.totalConstMem);
#else
      printf ("totalConstMem       %zu\n",      dev.totalConstMem);
#endif
      printf ("l2CacheSize         %d\n",       dev.l2CacheSize);
#ifdef _MSC_VER
      printf ("sharedMemPerBlock   %Iu\n",      dev.sharedMemPerBlock);
#else
      printf ("sharedMemPerBlock   %zu\n",      dev.sharedMemPerBlock);
#endif
      printf ("regsPerBlock        %d\n",       dev.regsPerBlock);
      printf ("warpSize            %d\n",       dev.warpSize);
#ifdef _MSC_VER
      printf ("memPitch            %Iu\n",      dev.memPitch);
#else
      printf ("memPitch            %zu\n",      dev.memPitch);
#endif
      printf ("maxThreadsPerBlock  %d\n",       dev.maxThreadsPerBlock);
      printf ("maxThreadsPerMP     %d\n",       dev.maxThreadsPerMultiProcessor);
      printf ("multiProcessorCount %d\n",       dev.multiProcessorCount);
      printf ("maxThreadsDim[3]    %d,%d,%d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
      printf ("maxGridSize[3]      %d,%d,%d\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
#ifdef _MSC_VER
      printf ("textureAlignment    %Iu\n",      dev.textureAlignment);
#else
      printf ("textureAlignment    %zu\n",      dev.textureAlignment);
#endif
      printf ("deviceOverlap       %d\n\n",     dev.deviceOverlap);
    }
}


void
rm_checkpoint (int q, int ks1)
{
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];

  if(!ks1)
  {
    sprintf (chkpnt_cfn, "c%ds1", q);
    sprintf (chkpnt_tfn, "t%ds1", q);
    (void) unlink (chkpnt_cfn);
    (void) unlink (chkpnt_tfn);
  }
  sprintf (chkpnt_cfn, "c%ds2", q);
  sprintf (chkpnt_tfn, "t%ds2", q);
  (void) unlink (chkpnt_cfn);
  (void) unlink (chkpnt_tfn);
}


int standardize_digits_int (int *x_int, int q, int n, int offset, int num_digits)
{
  int j, digit, stop, qn = q / n, carry = 0;
  int temp;
  int lo = 1 << qn;
  int hi = lo << 1;

  digit = floor(offset * (n / (double) q));
  j = (n + digit - 1) % n;
  while(x_int[j] == 0 && j != digit) j = (n + j - 1) % n;
  if(j == digit && x_int[digit] == 0) return(1);
  else if (x_int[j] < 0) carry = -1;
  {
    stop = (digit + num_digits) % n;
    j = digit;
    do
    {
      x_int[j] += carry;
      carry = 0;
      if (size[j]) temp = hi;
      else temp = lo;
      if(x_int[j] < 0)
      {
        x_int[j] += temp;
        carry = -1;
      }
      j = (j + 1) % n;
    }
    while(j != stop);
  }
  return(0);
}

void balance_digits_int(int* x, int q, int n)
{
  int half_low = (1 << (q / n - 1));
  int low = half_low << 1;
  int high = low << 1;
  int upper, adj, carry = 0;
  int j;

  for(j = 0; j < n; j++)
  {
    if(size[j])
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


unsigned *
read_checkpoint_packed (int q)
{
  //struct stat FileAttrib;
  FILE *fPtr;
  unsigned *x_packed;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;

  x_packed = (unsigned *) malloc (sizeof (unsigned) * (end + 25));

  sprintf (chkpnt_cfn, "c%ds1", q);
  sprintf (chkpnt_tfn, "t%ds1", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
  {
//#ifndef _MSC_VER
    //if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the checkpoint file. Trying the backup file.\n");
//#endif
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 25) , fPtr) != (sizeof (unsigned) * (end + 25)))
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != (unsigned int) q)
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return x_packed;
  }
  fPtr = fopen(chkpnt_tfn, "rb");
  if (!fPtr)
  {
//#ifndef _MSC_VER
//    if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the backup file. Restarting test.\n");
//#endif
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 25) , fPtr) != (sizeof (unsigned) * (end + 25)))
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != (unsigned int) q)
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return x_packed;
  }
  x_packed[end] = q;
  x_packed[end + 1] = 0; // n
  x_packed[end + 2] = 1; // iteration number
  x_packed[end + 3] = 0; // stage
  x_packed[end + 4] = 0; // accumulated time
  x_packed[end + 5] = 0; // b1
  // 6-9 reserved for extending b1
  // 10-24 reserved for stage 2
  int i;
  for(i = 6; i < 25; i++) x_packed[end + i] = 0;

  return x_packed;
}

int read_st2_checkpoint (int q, unsigned *x_packed)
{
  //struct stat FileAttrib;
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;

  sprintf (chkpnt_cfn, "c%ds2", q);
  sprintf (chkpnt_tfn, "t%ds2", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
  {
   // if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the checkpoint file. Trying the backup file.\n");
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 25) , fPtr) != (sizeof (unsigned) * (end + 25)))
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != (unsigned int) q)
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return 1;
  }
  fPtr = fopen(chkpnt_tfn, "rb");
  if (!fPtr)
  {
    //if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the backup file. Restarting test.\n");
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 25) , fPtr) != (sizeof (unsigned) * (end + 25)))
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != (unsigned int) q)
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");;
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return 1;
  }
  return 0;
}

void pack_bits_int(int *x_int, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;

  for(i = 0; i < n; i++)
  {
    temp1 = x_int[i];
    temp2 += (temp1 << k);
    k += qn + size[i];
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

void set_checkpoint_data(unsigned *x_packed, int q, int n, int j, int stage, int time)
{
  int end = (q + 31) / 32;

  x_packed[end + 0] = q;
  x_packed[end + 1] = n;
  x_packed[end + 2] = j;
  x_packed[end + 3] = stage;
  x_packed[end + 4] = time;
}

void
write_checkpoint_packed (unsigned *x_packed, int q)
{
  int end = (q + 31) / 32;
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];

  sprintf (chkpnt_cfn, "c%ds1", q);
  sprintf (chkpnt_tfn, "t%ds1", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr)
  {
    fprintf(stderr, "Couldn't write checkpoint.\n");
    return;
  }
  fwrite (x_packed, 1, sizeof (unsigned) * (end + 25), fPtr);
  fclose (fPtr);
 if (s_f > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, x_packed[end + 2], s_residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, x_packed[end + 2], s_residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr) return;
      fwrite (x_packed, 1, sizeof (unsigned) * (end + 25), fPtr);
      fclose (fPtr);
    }
}

void
write_st2_checkpoint (unsigned *x_packed, int q)
{
  int end = (q + 31) / 32;
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];

  sprintf (chkpnt_cfn, "c%ds2", q);
  sprintf (chkpnt_tfn, "t%ds2", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr)
  {
    fprintf(stderr, "Couldn't write checkpoint.\n");
    return;
  }
  fwrite (x_packed, 1, sizeof (unsigned) * (end + 25), fPtr);
  fclose (fPtr);
 if (s_f > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, x_packed[end + 2], s_residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, x_packed[end + 2], s_residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr) return;
      fwrite (x_packed, 1, sizeof (unsigned) * (end + 25), fPtr);
      fclose (fPtr);
    }
}

int printbits_int (int *x_int, int q, int n, int offset, FILE* fp, char *expectedResidue, int o_f)
{
  int j, k = 0;
  int digit, bit;
  unsigned long long temp, residue = 0;

    digit = floor(offset *  (n / (double) q));
    bit = offset - ceil(digit * (q / (double) n));
    j = digit;
    while(k < 64)
    {
      temp = x_int[j];
      residue = residue + (temp << k);
      k += q / n + size[j % n];
      if(j == digit)
      {
         k -= bit;
         residue >>= bit;
      }
      j = (j + 1) % n;
    }
    sprintf (s_residue, "%016llx", residue);

    printf ("M%d, 0x%s,", q, s_residue);
    //if(o_f) printf(" offset = %d,", offset);
    printf (" n = %dK, %s", n/1024, program);
    if (fp)
    {
      fprintf (fp, "M%d, 0x%s,", q, s_residue);
      if(o_f) fprintf(fp, " offset = %d,", offset);
      fprintf (fp, " n = %dK, %s", n/1024, program);
    }
  return 0;
}

void unpack_bits_int(int *x_int, unsigned *packed_x, int q , int n)
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
    if(k < qn + size[i])
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(size[i]) mask = mask1;
    else mask = mask2;
    x_int[i] = ((int) temp2) & mask;
    temp2 >>= (qn + size[i]);
    k -= (qn + size[i]);
  }
}

int* init_lucas_packed_int(unsigned * x_packed, int q , int *n, int *j, int *stage, int *total_time)
{
  int *x;
  int new_n, old_n;
  int end = (q + 31) / 32;
  int new_test = 0;

  *n = x_packed[end + 1];
  if(*n == 0) new_test = 1;
  *j = x_packed[end + 2];
  *stage = x_packed[end + 3];
  if(total_time) *total_time = x_packed[end + 4];

  old_n = fftlen;
  if(fftlen == 0) fftlen = *n;
  new_n = choose_fft_length(q, &fftlen);
  if(old_n > fft_count) *n = old_n;
  else if (new_test || old_n) *n = new_n;
  init_threads(*n);
  printf("Using threads: norm1 %d, mult %d, norm2 %d.\n", threads1, threads2, threads3);
  if ((*n / (2 * threads1)) > dev.maxGridSize[0])
  {
    fprintf (stderr, "over specifications Grid = %d\n", (int) *n / (2 * threads1));
    fprintf (stderr, "try increasing norm1 threads (%d) or decreasing FFT length (%dK)\n\n",  threads1, *n / 1024);
    return NULL;
  }
  if ((*n / (4 * threads2)) > dev.maxGridSize[0])
  {
    fprintf (stderr, "over specifications Grid = %d\n", (int) *n / (4 * threads2));
    fprintf (stderr, "try increasing mult threads (%d) or decreasing FFT length (%dK)\n\n",  threads2, *n / 1024);
    return NULL;
  }
  if ((*n % (2 * threads1))  != 0)
  {
    fprintf (stderr, "fft length %d must be divisible by 2 * norm1 threads %d\n", *n, threads1);
     return NULL;
  }
  if ((*n % (4 * threads2))  != 0)
  {
    fprintf (stderr, "fft length %d must be divisible by 4 * mult threads %d\n", *n, threads2);
     return NULL;
  }
  if (q  < *n * 0.8 * log((double) *n))
  {
    fprintf (stderr, "The fft length %dK is too large for the exponent %d. Restart with smaller fft.\n", *n / 1024, q);
    return NULL;
  }
  x = (int *) malloc (sizeof (int) * *n);
  alloc_gpu_mem(*n);
  write_gpu_data(q, *n);
  if(!new_test)
  {
    unpack_bits_int(x, x_packed, q, *n);
    balance_digits_int(x, q, *n);
  }
  init_x_int(x, x_packed, q, *n, stage);
  apply_weights <<<*n / (2 * threads1), threads1>>> (g_x, g_xint, g_ttmp);
  return x;
}

int isReasonable(int fft)
{ //From an idea of AXN's mentioned on the forums
  int i;

  while(!(fft & 1)) fft /= 2;
  for(i = 3; i <= 7; i += 2) while((fft % i) == 0) fft /= i;
  return (fft);
}


void threadbench (int n, int passes, int device_number)
{
  float total[216] = {0.0f}, outerTime, maxerr = 0.5f;
  int threads[6] = {32, 64, 128, 256, 512, 1024};
  int t1, t2, t3, i;
  float best_time = 10000.0f;
  int best_t1 = 0, best_t2 = 0, best_t3 = 0;
  int pass;
  cudaEvent_t start, stop;

  printf("CUDA bench, testing various thread sizes for fft %dK, doing %d passes.\n", n, passes);
  fflush(NULL);
  n *= 1024;

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * n / 4));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * n / 4));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * n / 32));
  cutilSafeCall (cudaMalloc ((void **) &g_datai, sizeof (int) * n / 32));
  cutilSafeCall (cudaMalloc ((void **) &g_carryi, sizeof (int) * n / 64));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));

  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
  cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));

  for(t1 = 0; t1 < 6; t1++)
  {
    if(n / (2 * threads[t1]) <= dev.maxGridSize[0] && n % (2 * threads[t1]) == 0)
    {
      for (t2 = 0; t2 < 6; t2++)
      {
        if(n / (4 * threads[t2]) <= dev.maxGridSize[0] && n % (4 * threads[t2]) == 0)
        {
          for (t3 = 0; t3 < 6; t3++)
          {
            for(pass = 1; pass <= passes; pass++)
            {
              cutilSafeCall (cudaEventRecord (start, 0));
              for (i = 0; i < 50; i++)
              {
                cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
                square <<< n / (4 * threads[t2]), threads[t2] >>> (n, g_x, g_ct);
                cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
                norm1a <<< n / (2 * threads[t1]), threads[t1] >>> (g_x, g_datai, g_xint, g_ttmp, g_carryi, g_err, maxerr, 0);
                norm2a <<< (n / (2 * threads[t1]) + threads[t3] - 1) / threads[t3], threads[t3] >>>
                           (g_x, g_xint, n, threads[t1], g_datai, g_carryi, g_ttp1, 0);
              }
              cutilSafeCall (cudaEventRecord (stop, 0));
              cutilSafeCall (cudaEventSynchronize (stop));
              cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
              outerTime /= 50.0f;
              total[36 * t1 + 6 * t2 + t3] += outerTime;
              //if(outerTime > max_diff[i]) max_diff[i] = outerTime;
            }
            printf ("fft size = %dK, ave time = %2.4f msec, Norm1 threads %d, Mult threads %d, Norm2 threads %d\n",
                      n / 1024 , total[36 * t1 + 6 * t2 + t3] / passes, threads[t1], threads[t2], threads[t3]);
            fflush(NULL);
          }
        }
      }
    }
  }

  for (i = 0; i < 216; i++)
  {
    if(total[i] < best_time && total[i] > 0.0f)
    {
      int j = i;
      best_time = total[i];
      best_t3 = j % 6;
      j /= 6;
      best_t2 = j % 6;
      best_t1 = j / 6;
    }
  }
  printf("\nBest time for fft = %dK, time: %2.4f, t1 = %d, t2 = %d, t3 = %d\n",
  n/1024, best_time / passes, threads[best_t1], threads[best_t2], threads[best_t3]);

  cufftSafeCall (cufftDestroy (plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_datai));
  cutilSafeCall (cudaFree ((char *) g_carryi));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));


  char threadfile[32];

  sprintf (threadfile, "%s threads.txt", dev.name);
  FILE *fptr;
  fptr = fopen(threadfile, "a+");
  if(!fptr) printf("Can't open %s threads.txt\n", dev.name);
  else fprintf(fptr, "%5d %4d %4d %4d %8.4f\n", n / 1024, threads[best_t1], threads[best_t2], threads[best_t3], best_time / passes);



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


void cufftbench (int cufftbench_s, int cufftbench_e, int passes, int device_number)
{
  //if(cufftbench_s % 2) cufftbench_s++;

  cudaEvent_t start, stop;
  float outerTime;
  int i, j, k;
  int end = cufftbench_e - cufftbench_s + 1;
  float best_time;
  float *total, *max_diff, maxerr = 0.5f;
  int threads[] = {32, 64, 128, 256, 512, 1024};
  int t1 = 3, t2 = 2, t3 = 2;

  if(end == 1)
  {
    threadbench(cufftbench_e, passes, device_number);
    return;
  }

  printf ("CUDA bench, testing reasonable fft sizes %dK to %dK, doing %d passes.\n", cufftbench_s, cufftbench_e, passes);

  total = (float *) malloc (sizeof (float) * end);
  max_diff = (float *) malloc (sizeof (float) * end);
  for(i = 0; i < end; i++)
  {
    total[i] = max_diff[i] = 0.0f;
  }

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * 256 * cufftbench_e));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * 256 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * 1024 / 32 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_datai, sizeof (int) * 1024 / 32 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_carryi, sizeof (int) * 512 / 32 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));

  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));

  for (j = cufftbench_s; j <= cufftbench_e; j++)
  {
    if(isReasonable(j) < 15)
    {
      int n = j * 1024;
      cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));
      for(k = 0; k < passes; k++)
      {
        cutilSafeCall (cudaEventRecord (start, 0));
        for (i = 0; i < 50; i++)
  	    {
          cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          square <<< n / (4 * threads[t2]), threads[t2] >>> (n, g_x, g_ct);
          cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          norm1a <<< n / (2 * threads[t1]), threads[t1] >>> (g_x, g_datai, g_xint, g_ttmp, g_carryi, g_err, maxerr, 0);
          norm2a <<< (n / (2 * threads[t1]) + threads[t3] - 1) / threads[t3], threads[t3] >>> (g_x, g_xint, n, threads[t1], g_datai, g_carryi, g_ttp1, 0);
        }
        cutilSafeCall (cudaEventRecord (stop, 0));
        cutilSafeCall (cudaEventSynchronize (stop));
        cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
        i = j - cufftbench_s;
        outerTime /= 50.0f;
        total[i] += outerTime;
        if(outerTime > max_diff[i]) max_diff[i] = outerTime;
      }
      cufftSafeCall (cufftDestroy (plan));
      printf ("fft size = %dK, ave time = %2.4f msec, max-ave = %0.5f\n",
                  j , total[i] / passes, max_diff[i] - total[i] / passes);
      fflush(NULL);
    }
  }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_datai));
  cutilSafeCall (cudaFree ((char *) g_carryi));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));

  i = end - 1;
  best_time = 10000.0f;
  while(i >= 0)
  {
    if(total[i] > 0.0f && total[i] < best_time) best_time = total[i];
    else total[i] = 0.0f;
    i--;
  }
  char fftfile[32];
  FILE *fptr;

  sprintf (fftfile, "%s fft.txt", dev.name);
  fptr = fopen(fftfile, "w");
  if(!fptr)
  {
    printf("Cannot open %s.\n",fftfile);
    printf ("Device              %s\n", dev.name);
    printf ("Compatibility       %d.%d\n", dev.major, dev.minor);
    printf ("clockRate (MHz)     %d\n", dev.clockRate/1000);
    printf ("memClockRate (MHz)  %d\n", dev.memoryClockRate/1000);
    printf("\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
         int tl = (int) (exp(0.9784876919 * log ((double)cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        printf("%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fflush(NULL);
  }
  else
  {
    fprintf (fptr, "Device              %s\n", dev.name);
    fprintf (fptr, "Compatibility       %d.%d\n", dev.major, dev.minor);
    fprintf (fptr, "clockRate (MHz)     %d\n", dev.clockRate/1000);
    fprintf (fptr, "memClockRate (MHz)  %d\n", dev.memoryClockRate/1000);
    fprintf(fptr, "\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
        int tl = (int) (exp(0.9784876919 * log ((double)cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        fprintf(fptr, "%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fclose(fptr);
    printf("Optimal fft lengths saved in %s.\nPlease email a copy to james@mersenne.ca.\n", fftfile);
    fflush(NULL);
   }

  free ((char *) total);
  free ((char *) max_diff);
}

void
SetQuitting (int sig)
{
  quitting = 1;
  sig==SIGINT ? printf( "\tSIGINT") : (sig==SIGTERM ? printf( "\tSIGTERM") : printf( "\tUnknown signal")) ;
  printf( " caught, writing checkpoint.\n");
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

int interact(void); // defined below everything else

int get_bit(int location, unsigned *control)
{
  int digit = location / 32;
  int bit = location % 32;

  bit = 1 << bit;
  bit = control[digit] & bit;
  if(bit) bit /= bit;
  return(bit);
}

int round_off_test(int q, int n, int *j, unsigned *control, int last)
{
  int k;
  float totalerr = 0.0;
  float terr, avgerr, maxerr = 0.0;
  float max_err = 0.0, max_err1 = 0.0;
  int bit;

      printf("Running careful round off test for 1000 iterations. If average error > 0.25, the test will restart with a longer FFT.\n");
      for (k = 0; k < 1000 && k < last; k++)
	    {
        bit = get_bit(last - k - 1, control);
        terr = lucas_square (q, n, k, last, &maxerr, 1, bit, 1, k == 999);
        if(terr > maxerr) maxerr = terr;
        if(terr > max_err) max_err = terr;
        if(terr > max_err1) max_err1 = terr;
        totalerr += terr;
        reset_err(&maxerr, 0.85);
        if(terr >= 0.35)
        {
	        printf ("Iteration = %d < 1000 && err = %5.5f >= 0.35, increasing n from %dK\n", k, terr, n/1024);
	        fftlen++;
	        return 1;
        }
	      if(k && (k % 100 == 0))
        {
	        printf( "Iteration  %d, average error = %5.5f, max error = %5.5f\n", k, totalerr / k, max_err);
	        max_err = 0.0;
	      }
	    }
      avgerr = totalerr/1000.0;
      if( avgerr > 0.25)
      {
        printf("Iteration 1000, average error = %5.5f > 0.25 (max error = %5.5f), increasing FFT length and restarting\n", avgerr, max_err);
        fftlen++;
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
        reset_err(&maxerr, 0.85);
      }
      *j += 1000;
      return 0;
}

unsigned *get_control(int *j, int lim1, int lim2, int q)
{
  mpz_t result;
  int p = 2;
  int limit;
  int prime_power = 1;
  unsigned *control = NULL;

	mpz_init(result);
  if(lim2 == 0)
  {
    mpz_set_ui (result, 2 * q);
    limit = lim1 / p;
    while (prime_power <= limit) prime_power *= p;
    mpz_mul_ui(result, result, prime_power);
    p = 3;
    while (p <= lim1)
    {
      while(p <= lim1 && !isprime(p)) p += 2;
      limit = lim1 / p;
      prime_power = p;
      while (prime_power <= limit) prime_power *= p;
      mpz_mul_ui(result, result, prime_power);
      p += 2;
    }
  }
  else
  {
    p = lim1;
    if(!(lim1 & 1)) p++;
    mpz_set_ui (result, 1);
    while (p <= lim2)
    {
      while(p <= lim2 && !isprime(p)) p += 2;
      mpz_mul_ui(result, result, p);
      printf("prime_power: %d, %d\n", prime_power, p);
      p += 2;
    }

  }
  *j = mpz_sizeinbase (result, 2);
  control = (unsigned *) malloc (sizeof (unsigned) * ((*j + 31) / 32));
  mpz_export (control, NULL, -1, 4, 0, 0, result);
  mpz_clear (result);
  return control;
}

int get_gcd(unsigned *x_packed, int q, int n, int stage)
{
  	  mpz_t result, prime, prime1;
	    int end = (q + 31) / 32;
          int rv = 0;

	    mpz_init2( result, q);
	    mpz_init2( prime, q);
	    mpz_init2( prime1, q);
	    mpz_import (result, end, -1, sizeof(x_packed[0]), 0, 0, x_packed);
	    if(stage == 1) mpz_sub_ui (result, result, 1);
	    mpz_setbit (prime, q);
	    mpz_sub_ui (prime, prime, 1);
	    if (mpz_cmp_ui (result, 0))
	    {
	      mpz_gcd (prime1, prime, result);
	      if (mpz_cmp_ui (prime1, 1))
	      {
	        rv = 1;
		printf( "M%d has a factor: ", q);
	        mpz_out_str (stdout, 10, prime1);
                if (stage==1) printf (" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,b1,g_e,n/1024, program); // Found in stage 1
                else printf (" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	        fprintf (fp, "M%d has a factor: ", q);
	        mpz_out_str (fp, 10, prime1);
		if  (AID[0] && strncasecmp(AID, "N/A", 3)) {
                   if (stage==1) fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1,b1,g_e,n/1024, AID, program);
                   else fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1,g_b2,g_e,n/1024, AID, program);
		}
                else {
                   if (stage==1) fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,b1,g_e,n/1024, program);
                   else fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                }
                unlock_and_fclose(fp);
	      }
	     }
	   if (rv == 0) {
                printf( "M%d Stage %d found no factor", q, stage);
                printf (" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                if (stage==2) {
		  FILE* fp = fopen_and_lock(RESULTSFILE, "a");
                  fprintf (fp, "M%d found no factor", q);
                  if  (AID[0] && strncasecmp(AID, "N/A", 3))
                    fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1,g_b2,g_e,n/1024, AID, program);
		  else
                    fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                  unlock_and_fclose(fp);
	      }

	    }
	    mpz_clear (result);
	    mpz_clear (prime);
	    mpz_clear (prime1);
      return rv;
}

/**************************************************************/
/* Routines to compute optimal and test to optimal P-1 bounds */
/*    Stolen from Woltman's Prime95 and adapted to CUDAPm1    */
/**************************************************************/

/* This table gives the values of Dickman's function given an input */
/* between 0.000 and 0.500.  These values came from a different program */
/* that did a numerical integration. */

static double savedF[501] = {
	0, 0, 0, 0, 0, 0, 3.3513e-215, 5.63754e-208, 4.00865e-201,
	1.65407e-194, 4.53598e-188, 8.93587e-182, 1.33115e-175,
	1.55557e-169, 1.46609e-163, 1.13896e-157, 7.42296e-152,
	3.80812e-146, 1.56963e-140, 5.32886e-135, 1.51923e-129,
	3.69424e-124, 7.76066e-119, 1.42371e-113, 2.30187e-108,
	3.30619e-103, 4.24793e-098, 4.80671e-093, 4.78516e-088,
	4.22768e-083, 3.33979e-078, 2.37455e-073, 1.52822e-068,
	8.94846e-064, 4.78909e-059, 4.65696e-057, 4.49802e-055, 4.31695e-053,
	4.07311e-051, 3.81596e-049, 3.61043e-047, 1.73046e-045, 8.26375e-044,
	3.9325e-042, 1.86471e-040, 8.8102e-039, 4.14402e-037, 1.99497e-035,
	1.83001e-034, 1.59023e-033, 1.45505e-032, 1.24603e-031, 1.15674e-030,
	9.70832e-030, 9.23876e-029, 4.20763e-028, 4.24611e-027, 1.61371e-026,
	6.59556e-026, 3.17069e-025, 1.12205e-024, 4.65874e-024, 2.01267e-023,
	6.2941e-023, 3.02604e-022, 7.84622e-022, 2.3526e-021, 6.7049e-021,
	1.88634e-020, 4.59378e-020, 1.37233e-019, 4.00682e-019, 8.34209e-019,
	2.21612e-018, 4.84252e-018, 1.02457e-017, 2.03289e-017, 4.07704e-017,
	1.33778e-016, 2.4263e-016, 4.14981e-016, 7.0383e-016, 1.20511e-015,
	3.85644e-015, 6.52861e-015, 1.06563e-014, 1.67897e-014, 2.79916e-014,
	4.54319e-014, 9.83296e-014, 1.66278e-013, 2.61858e-013, 4.03872e-013,
	5.98967e-013, 1.09674e-012, 1.70553e-012, 2.56573e-012, 3.72723e-012,
	6.14029e-012, 9.33636e-012, 1.36469e-011, 1.89881e-011, 2.68391e-011,
	4.12016e-011, 5.94394e-011, 8.43746e-011, 1.12903e-010, 1.66987e-010,
	2.36959e-010, 3.11726e-010, 4.28713e-010, 5.90781e-010, 7.79892e-010,
	1.05264e-009, 1.4016e-009, 1.87506e-009, 2.42521e-009, 3.14508e-009,
	4.38605e-009, 5.43307e-009, 6.96737e-009, 8.84136e-009, 1.16286e-008,
	1.42343e-008, 1.79697e-008, 2.30867e-008, 2.88832e-008, 3.52583e-008,
	4.31032e-008, 5.46444e-008, 6.66625e-008, 8.06132e-008, 1.00085e-007,
	1.20952e-007, 1.4816e-007, 1.80608e-007, 2.13125e-007, 2.5324e-007,
	3.094e-007, 3.64545e-007, 4.31692e-007, 5.19078e-007, 6.03409e-007,
	7.21811e-007, 8.53856e-007, 9.71749e-007, 1.13949e-006, 1.37042e-006,
	1.53831e-006, 1.79066e-006, 2.15143e-006, 2.40216e-006, 2.76872e-006,
	3.20825e-006, 3.61263e-006, 4.21315e-006, 4.76404e-006, 5.43261e-006,
	6.2041e-006, 6.96243e-006, 7.94979e-006, 8.89079e-006, 1.01387e-005,
	1.13376e-005, 1.2901e-005, 1.44183e-005, 1.59912e-005, 1.79752e-005,
	1.99171e-005, 2.22665e-005, 2.47802e-005, 2.7678e-005, 3.0492e-005,
	3.34189e-005, 3.71902e-005, 4.12605e-005, 4.54706e-005, 4.98411e-005,
	5.48979e-005, 6.06015e-005, 6.61278e-005, 7.22258e-005, 7.97193e-005,
	8.66574e-005, 9.48075e-005, 0.00010321, 0.000112479, 0.000121776,
	0.000133344, 0.000144023, 0.000156667, 0.000168318, 0.000183192,
	0.000196527, 0.00021395, 0.000228389, 0.000249223, 0.000264372,
	0.000289384, 0.000305707, 0.000333992, 0.000353287, 0.000379868,
	0.000408274, 0.00043638, 0.000465319, 0.000496504, 0.000530376,
	0.000566008, 0.000602621, 0.000642286, 0.000684543, 0.000723853,
	0.000772655, 0.000819418, 0.000868533, 0.000920399, 0.000975529,
	0.00103188, 0.00109478, 0.00115777, 0.00122087, 0.00128857,
	0.00136288, 0.00143557, 0.00151714, 0.00159747, 0.00167572,
	0.00176556, 0.00186199, 0.00195063, 0.00205239, 0.00216102,
	0.00225698, 0.00236962, 0.00249145, 0.00259636, 0.00272455,
	0.00287006, 0.00297545, 0.00312346, 0.0032634, 0.00340298,
	0.00355827, 0.00371195, 0.00387288, 0.00404725, 0.00420016,
	0.00439746, 0.00456332, 0.00475936, 0.00495702, 0.00514683,
	0.00535284, 0.00557904, 0.00578084, 0.00601028, 0.00623082,
	0.00647765, 0.00673499, 0.00696553, 0.00722529, 0.00748878,
	0.00775537, 0.00803271, 0.00832199, 0.00861612, 0.00889863,
	0.00919876, 0.00953343, 0.00985465, 0.0101993, 0.0105042, 0.0108325,
	0.0112019, 0.0115901, 0.0119295, 0.0123009, 0.0127191, 0.0130652,
	0.0134855, 0.0139187, 0.0142929, 0.0147541, 0.0151354, 0.0156087,
	0.0160572, 0.0165382, 0.0169669, 0.0174693, 0.017946, 0.0184202,
	0.0189555, 0.0194336, 0.0200107, 0.0204863, 0.0210242, 0.0216053,
	0.0221361, 0.0226858, 0.0232693, 0.0239027, 0.0244779, 0.025081,
	0.0257169, 0.0263059, 0.0269213, 0.0275533, 0.0282065, 0.0289028,
	0.029567, 0.0302268, 0.0309193, 0.0316619, 0.0323147, 0.0330398,
	0.0338124, 0.0345267, 0.0353038, 0.0360947, 0.0368288, 0.0376202,
	0.0383784, 0.0391894, 0.0399684, 0.0408148, 0.0416403, 0.042545,
	0.0433662, 0.0442498, 0.0451003, 0.046035, 0.0468801, 0.0478059,
	0.0487442, 0.0496647, 0.0505752, 0.0515123, 0.0524792, 0.0534474,
	0.0544682, 0.0554579, 0.0565024, 0.0574619, 0.0584757, 0.0595123,
	0.0605988, 0.0615874, 0.062719, 0.0637876, 0.064883, 0.0659551,
	0.0670567, 0.0681256, 0.0692764, 0.0704584, 0.0715399, 0.0727237,
	0.0738803, 0.0750377, 0.0762275, 0.0773855, 0.0785934, 0.0797802,
	0.0810061, 0.0822205, 0.0834827, 0.084714, 0.0858734, 0.0871999,
	0.0884137, 0.0896948, 0.090982, 0.0922797, 0.093635, 0.0948243,
	0.0961283, 0.0974718, 0.0988291, 0.100097, 0.101433, 0.102847,
	0.104222, 0.105492, 0.106885, 0.10833, 0.109672, 0.111048, 0.112438,
	0.113857, 0.115311, 0.11673, 0.118133, 0.119519, 0.12099, 0.122452,
	0.123905, 0.125445, 0.126852, 0.128326, 0.129793, 0.131277, 0.132817,
	0.134305, 0.135772, 0.137284, 0.138882, 0.140372, 0.14192, 0.143445,
	0.14494, 0.146515, 0.148145, 0.149653, 0.151199, 0.152879, 0.154368,
	0.155958, 0.157674, 0.159211, 0.160787, 0.16241, 0.164043, 0.165693,
	0.167281, 0.168956, 0.170589, 0.172252, 0.173884, 0.175575, 0.177208,
	0.178873, 0.180599, 0.18224, 0.183975, 0.185654, 0.187363, 0.189106,
	0.190729, 0.19252, 0.194158, 0.195879, 0.197697, 0.199391, 0.201164,
	0.202879, 0.204602, 0.206413, 0.20818, 0.209911, 0.211753, 0.213484,
	0.215263, 0.21705, 0.218869, 0.220677, 0.222384, 0.224253, 0.226071,
	0.227886, 0.229726, 0.231529, 0.233373, 0.235234, 0.237081, 0.238853,
	0.240735, 0.242606, 0.244465, 0.246371, 0.248218, 0.250135, 0.251944,
	0.253836, 0.255708, 0.257578, 0.259568, 0.261424, 0.263308, 0.265313,
	0.26716, 0.269073, 0.271046, 0.272921, 0.274841, 0.276819, 0.278735,
	0.280616, 0.282653, 0.284613, 0.286558, 0.288478, 0.290472, 0.292474,
	0.294459, 0.296379, 0.298382, 0.300357, 0.302378, 0.30434, 0.306853
};

/* This evaluates Dickman's function for any value.  See Knuth vol. 2 */
/* for a description of this function and its use. */

double F (double x)
{
	int	i;

	if (x >= 1.0) return (1.0);
	if (x >= 0.5) return (1.0 + log (x));
	i = (int) (x * 1000.0);
	return (savedF[i] + (x * 1000.0 - i) * (savedF[i+1] - savedF[i]));
}

/* Analyze how well P-1 factoring will perform */

void guess_pminus1_bounds (
	int guess_exp,	/* N in K*B^N+C. Exponent to test. */
	int	how_far_factored,	/* Bit depth of trial factoring */
	int	tests_saved,		/* 1 if doublecheck, 2 if first test */
	int vals,
	int *bound1,
	int *bound2,
	double	*success_rate)
{
	int guess_B1, guess_B2, /*vals,*/ i;
	double	h, pass1_squarings, pass2_squarings;
	double	logB1, logB2, kk, logkk, temp, logtemp, log2;
	double	prob, gcd_cost, ll_tests, numprimes;
	struct {
		int B1;
		int B2;
		double	prob;
		double	pass1_squarings;
		double	pass2_squarings;
	} best[2];


        for (i=0; i<2; i++) {
	   best[i].B1=0;
	   best[i].B2=0;
	   best[i].prob=0;
	   best[i].pass1_squarings=0;
	   best[i].pass2_squarings=0;
	}
/* Guard against wild tests_saved values.  Huge values will cause this routine */
/* to run for a very long time.  This shouldn't happen as auxiliaryWorkUnitInit */
/* now has the exact same test. */

	if (tests_saved > 10) tests_saved = 10;

/* Balance P-1 against 1 or 2 LL tests (actually more since we get a */
/* corrupt result reported some of the time). */

	ll_tests = (double) tests_saved + 2 * 0.018;

/* Precompute the cost of a GCD.  We used Excel to come up with the */
/* formula GCD is equivalent to 861 * Ln (p) - 7775 transforms. */
/* Since one squaring equals two transforms we get the formula below. */
/* NOTE: In version 22, the GCD speed has approximately doubled.  I've */
/* adjusted the formula accordingly. */

	gcd_cost = (430.5 * log ((double) guess_exp) - 3887.5) / 2.0;
	if (gcd_cost < 50.0) gcd_cost = 50.0;

/* Compute how many temporaries we can use given our memory constraints. */
/* Allow 1MB for code and data structures. */

	// vals = cvt_mem_to_estimated_gwnums (max_mem (thread_num), k, b, n, c);
	// if (vals < 1) vals = 1;
	//vals = 176;

/* Find the best B1 */

	log2 = log ((double) 2.0);
	for (guess_B1 = 10000; ; guess_B1 += 5000) {

/* Constants */

	logB1 = log ((double) guess_B1);

/* Compute how many squarings will be required in pass 1 */

	pass1_squarings = ceil (1.44 * guess_B1);

/* Try a lot of B2 values */

	for (guess_B2 = guess_B1; guess_B2 <= guess_B1 * 100; guess_B2 += guess_B1 >> 2) {

/* Compute how many squarings will be required in pass 2.  In the */
/* low-memory cases, assume choose_pminus1_plan will pick D = 210, E = 1 */
/* If more memory is available assume choose_pminus1_plan will pick */
/* D = 2310, E = 2.  This will provide an accurate enough cost for our */
/* purposes even if different D and E values are picked.  See */
/* choose_pminus1_plan for a description of the costs of P-1 stage 2. */

/* For cudapm1, we're not set up for e = 1, assume e = 2 in both cases*/

	logB2 = log ((double) guess_B2);
	numprimes = (unsigned long) (guess_B2 / (logB2 - 1.0) - guess_B1 / (logB1 - 1.0));
	if (guess_B2 <= guess_B1) {
		pass2_squarings = 0.0;
	} else if (vals <= 8) {		/* D = 210, E = 1, passes = 48/temps */
		unsigned long num_passes;
		num_passes = (unsigned long) ceil (48.0 / (vals - 3));
		pass2_squarings = ceil ((guess_B2 - guess_B1) / 210.0) * num_passes;
		pass2_squarings += numprimes * 1.1;
	} else {
		unsigned long num_passes;
		double	numpairings;
		num_passes = (unsigned long) ceil (480.0 / (vals - 3));
		numpairings = (unsigned long)
			(numprimes / 2.0 * numprimes / ((guess_B2-guess_B1) * 480.0/2310.0));
		pass2_squarings = 2400.0 + num_passes * 90.0; /* setup costs */
		pass2_squarings += ceil ((guess_B2-guess_B1) / 4620.0) * 2.0 * num_passes; /*number of base changes per pass * e with e = 2*/
		pass2_squarings += numprimes - numpairings; /*these are the sub_mul operations*/
	}

/* Pass 2 FFT multiplications seem to be at least 20% slower than */
/* the squarings in pass 1.  This is probably due to several factors. */
/* These include: better L2 cache usage and no calls to the faster */
/* gwsquare routine.  Nov, 2009:  On my Macbook Pro, with exponents */
/* around 45M and using 800MB memory, pass2 squarings are 40% slower. */

/* Owftheevil reports that CUDA squarings are only about 2% slower. */
/* New normaliztion kernels benefit stage 1 more than stage 2, back to 9% */

	pass2_squarings *= 1.09;  // was 1.35

/* What is the "average" value that must be smooth for P-1 to succeed? */
/* Ordinarily this is 1.5 * 2^how_far_factored.  However, for Mersenne */
/* numbers the factor must be of the form 2kp+1.  Consequently, the */
/* value that must be smooth (k) is much smaller. */

	kk = 1.5 * pow (2.0, how_far_factored);
	// if (k == 1.0 && b == 2 && c == -1)
	kk = kk / 2.0 / guess_exp;
	logkk = log (kk);

/* Set temp to the number that will need B1 smooth if k has an */
/* average-sized factor found in stage 2 */

	temp = kk / ((guess_B1 + guess_B2) / 2);
	logtemp = log (temp);

/* Loop over increasing bit lengths for the factor */

	prob = 0.0;
	for (h = how_far_factored; ; ) {
		double	prob1, prob2;

/* If temp < 1.0, then there are no factor to find in this bit level */

		if (logtemp > 0.0) {

/* See how many smooth k's we should find using B1 */
/* Using Dickman's function (see Knuth pg 382-383) we want k^a <= B1 */

			prob1 = F (logB1 / logkk);

/* See how many smooth k's we should find using B2 */
/* Adjust this slightly to eliminate k's that have two primes > B1 and < B2 */
/* Do this by assuming the largest factor is the average of B1 and B2 */
/* and the remaining cofactor is B1 smooth */

			prob2 = prob1 + (F (logB2 / logkk) - prob1) *
				        (F (logB1 / logtemp) / F (logB2 / logtemp));
			if (prob2 < 0.0001) break;

/* Add this data in to the total chance of finding a factor */

			prob += prob2 / (h + 0.5);
		}

/* Move to next bit level */

		h += 1.0;
		logkk += log2;
		logtemp += log2;
	}

/* See if this is a new best case scenario */

	if (guess_B2 == guess_B1 ||
	    prob * ll_tests * guess_exp - pass2_squarings >
			best[0].prob * ll_tests * guess_exp - best[0].pass2_squarings){
		best[0].B2 = guess_B2;
		best[0].prob = prob;
		best[0].pass2_squarings = pass2_squarings;
		if (vals < 4) break;
		continue;
	}

	if (prob * ll_tests * guess_exp - pass2_squarings <
		0.9 * (best[0].prob * ll_tests * guess_exp - best[0].pass2_squarings))
		break;
	continue;
	}

/* Is this the best B1 thusfar? */

	if (guess_B1 == 10000 ||
	    best[0].prob * ll_tests * guess_exp -
			(pass1_squarings + best[0].pass2_squarings) >
		best[1].prob * ll_tests * guess_exp -
			(best[1].pass1_squarings + best[1].pass2_squarings)) {
		best[1].B1 = guess_B1;
		best[1].B2 = best[0].B2;
		best[1].prob = best[0].prob;
		best[1].pass1_squarings = pass1_squarings;
		best[1].pass2_squarings = best[0].pass2_squarings;
		continue;
	}
	if (best[0].prob * ll_tests * guess_exp -
			(pass1_squarings + best[0].pass2_squarings) <
	    0.9 * (best[1].prob * ll_tests * guess_exp -
			(best[1].pass1_squarings + best[1].pass2_squarings)))
		break;
	continue;
	}

/* Return the final best choice */

	if (best[1].prob * ll_tests * guess_exp >
		best[1].pass1_squarings + best[1].pass2_squarings + gcd_cost) {
		*bound1 = best[1].B1;
		*bound2 = best[1].B2;
		// *squarings = (unsigned long)
		//	(best[1].pass1_squarings +
		//	 best[1].pass2_squarings + gcd_cost);
		*success_rate = best[1].prob;
	} else {
		*bound1 = 0;
		*bound2 = 0;
		// *squarings = 0;
		*success_rate = 0.0;
	}
}


/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int stage2_init_param3(int e, int n, int trans, float *err)
{
  int j, i, k = 0, base;
  mpz_t exponent;
  long long b[7];

  for(i = 0; i <= e/2; i++)
  {
    base = 2 * i + 1;
    b[i] = 1;
    for(j = 0; j < e / 2; j++) b[i] *= base;
    b[i] *= b[i];
  }
  for(i = e/2; i > 0; i--)
  {
    while (k < i)
    {
      j = i;
      while(j > k)
      {
        b[j] = b[j] - b[j-1];
        j--;
      }
      k++;
      j = i;
      while(j > k)
      {
        b[j] = b[j] - b[j-1];
        j--;
      }
    }
  }
  mpz_init(exponent);
  for(i = 0; i <= e / 2; i++)
  {
    mpz_set_ui (exponent, b[i]);
    trans = E_to_the_p(&e_data[2 * i * n], g_x, exponent, n, trans, err);
    if(i > 0)
    {
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) &e_data[2 * i * n], (cufftDoubleComplex *) &e_data[2 * i * n], CUFFT_INVERSE));
	    copy_kernel<<<n / (2*threads1), threads1>>>(&e_data[(2 * i - 1) * n], &e_data[2 * i * n]);
	    trans++;
    }
  }
  E_pre_mul(&e_data[e * n], &e_data[e * n], n, 0);
  E_pre_mul(&e_data[0], &e_data[0], n, 1);
  trans++;
  mpz_clear(exponent);
  return trans;
}

int next_base1(int k, int e, int n, int trans, float *err)
{
  int j;

  if(k == 1) return(stage2_init_param3(e, n, trans,  err));
  if(k > 3)
  {
    if(k <= e + 1)
    {
      E_mul(&e_data[(k - 3) * n], &e_data[(k - 2) * n], &e_data[(k - 3) * n], n, *err, 0);
      j = e + 3 - k;
      trans += 2 * (k - 3);
    }
    else
    {
      E_half_mul(&e_data[(e-1) * n], &e_data[(e-1) * n], &e_data[e * n], n, *err);
      j = 1;
      trans += 2 * ( e - 1);
    }
    for(; j < e-1; j++) E_mul(&e_data[(e-j-1) * n], &e_data[(e-j) * n], &e_data[(e-j-1) * n], n, *err, 1);
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) &e_data[1 * n], (cufftDoubleComplex *) &e_data[1 * n], CUFFT_INVERSE));
  }
  E_half_mul(&e_data[0], &e_data[1 * n], &e_data[0], n, *err);
  E_pre_mul(&e_data[0], &e_data[0], n, 1);
  trans += 2;
  return trans;
}

int stage2_init_param1(int k, int base, int e, int n, int trans, float *err)
{
  int i, j;

  if(base > 1)
  {
    mpz_t exponent;
    mpz_init(exponent);
    mpz_ui_pow_ui (exponent, base, e);
    trans = E_to_the_p(&e_data[0], g_x, exponent, n, trans, err);
    E_pre_mul(g_x, &e_data[0], n, 1);
    trans++;
    mpz_clear(exponent);
  }

  if(k < 2 * e)
    for(j = 1; j <= k; j += 2)
    {
      trans = next_base1(j, e, n, trans, err);
      cutilSafeThreadSync();
    }
  else
  {
    mpz_t *exponents;

    exponents = (mpz_t *) malloc((e + 1) * sizeof(mpz_t));
    for(j = 0; j <= e; j++) mpz_init(exponents[j]);
    for(j = e; j >= 0; j--) mpz_ui_pow_ui (exponents[j], (k - j * 2), e);
    for(j = 0; j < e; j++)
      for(i = e; i > j; i--) mpz_sub(exponents[i], exponents[i-1], exponents[i]);
    for(j = 0; j <= e; j++) trans = E_to_the_p(&e_data[j * n], g_x, exponents[j], n, trans, err);
    for(j = 0; j <= e; j++) mpz_clear(exponents[j]);
    E_pre_mul(&e_data[0], &e_data[0], n, 1);
    E_pre_mul(&e_data[e * n], &e_data[e * n], n, 1);
    for(j = 1; j < e; j++)
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) &e_data[j * n], (cufftDoubleComplex *) &e_data[j * n], CUFFT_INVERSE));
    trans += e + 1;
  }
  return trans;
}

int stage2_init_param2(int num, int cur_rp, int base, int e, int n, uint8 *gaps, int trans, float *err)
{
  int rp = 1, j = 0, i;
  mpz_t exponent;

  mpz_init(exponent);

  while(j < cur_rp)
  {
    j++;
    rp += 2 * gaps[j];
  }
  for(i = 0; i < num; i++)
  {
      mpz_ui_pow_ui (exponent, rp, e);
      trans = E_to_the_p(&rp_data[i * n], g_x, exponent, n, trans, err);
      E_pre_mul(&rp_data[i * n], &rp_data[i * n], n, 1);
      trans++;
      j++;
      if(rp < base - 1) rp += 2 * gaps[j];
  }

  mpz_clear(exponent);

  return trans;
}

int stage2_init_param4(int num, int cur_rp, int base, int e, int n, uint8 *gaps, int trans, float *err)
{
  int rp = 1, j = 0, i, k = 1;

  while(j < cur_rp)
  {
    j++;
    rp += 2 * gaps[j];
  }
  trans = stage2_init_param1(rp, 1, e, n, trans, err);
  copy_kernel<<<n / (2*threads1), threads1>>>(&rp_data[0], &e_data[0]);
  k = rp + 2;
  for(i = 1; i < num; i++)
  {
    j++;
    rp += 2 * gaps[j];
    while(k <= rp)
    {
      trans = next_base1(k, e, n, trans, err);
      cutilSafeThreadSync();
      k += 2;
    }
    copy_kernel<<<n / (2*threads1), threads1>>>(&rp_data[i * n], &e_data[0]);

  }

  return trans;
}

int rp_init_count1(int k, int base, int e, int n)
{
  int i, j, trans = 0;
  int numb[6] = {10,38,102,196,346,534};
  int numb1[11] = {2,8,18,32,50,72,96,120,144,168,192};
  mpz_t exponent;

  mpz_init(exponent);
  mpz_ui_pow_ui (exponent, base, e);
  trans += (int) mpz_sizeinbase (exponent, 2) + (int) mpz_popcount(exponent) - 2;
  mpz_clear(exponent);

  if(k < 2 * e)
  {
    trans = 2 * trans + 1;
    trans += numb[e / 2 - 1] + numb1[k/2-1];
    return(trans);
  }
  else
  {
    mpz_t *exponents;
    exponents = (mpz_t *) malloc((e+1) * sizeof(mpz_t));
    for(j = 0; j <= e; j++) mpz_init(exponents[j]);
    for(j = e; j >= 0; j--) mpz_ui_pow_ui (exponents[j], (k - j * 2), e);
    for(j = 0; j < e; j++)
      for(i = e; i > j; i--) mpz_sub(exponents[i], exponents[i-1], exponents[i]);
    for(j = 0; j <= e; j++)
	  {
	      trans += (int) mpz_sizeinbase (exponents[j], 2) + (int) mpz_popcount(exponents[j]) - 2;
    }
    for(j = 0; j <= e; j++) mpz_clear(exponents[j]);
    return 2 * (trans + e + 2) - 1;
  }
}

int rp_init_count1a(int k, int base, int e, int n)
{
  int trans;
  int numb[6] = {10,38,102,196,346,534};
  int numb1[12] = {0,2,8,18,32,50,72,96,120,144,168,192};

  trans = (int) (e * log2((double)base) * 3.0 );
  if(k < 2 * e)
  {
    trans += numb[e/2-1] + numb1[(k+1)/2-1];
  }
  else
  {
    if(e == 2) trans += (int) (9.108 * log2((double)k) + 10.7);
    else if(e == 4) trans += (int) (30.349 * log2((double)k) + 50.5);
    else if(e == 6) trans += (int) (64.560 * log2((double)k) + 137.6);
    else if(e == 8) trans += (int) (110.224 * log2((double)k) + 265.2);
    else if(e == 10) trans += (int) (168.206 * log2((double)k) + 478.6);
    else trans += (int) (237.888 * log2((double)k) + 731.5);
  }
  return trans;
}

int rp_init_count2(int num, int cur_rp, int e, int n, uint8 *gaps)
{
  int rp = 1, j = 0, i, trans = 0;
  int numb[6] = {10,38,102,196,346,534};

  while(j < cur_rp)
  {
    j++;
    rp += 2 * gaps[j];
  }
  if(cur_rp == 0) trans -= e * e / 2 - 1;
  cur_rp = rp;
  if(rp == 1) trans += numb[e/2-1];
  else trans = rp_init_count1(rp, 1, e, n);
  for(i = 1; i < num; i++)
  {
      j++;
      rp += 2 * gaps[j];
  }
  trans += e * (rp - cur_rp);

  return trans;
}

int rp_init_count2a(int cur_rp, int e, int n, uint8 *gaps)
{
  int rp = 1, j = 0, trans = 0;
  int numb[6] = {10,38,102,196,346,534};

  while(j < cur_rp)
  {
    j++;
    rp += 2 * gaps[j];
  }
  if(cur_rp == 0) trans -= e * e / 2 - 1;
  cur_rp = rp;
  if(rp == 1) trans += numb[e/2-1];
  else trans = rp_init_count1a(rp, 1, e, n);

  return trans;
}


int stage2(int *x_int, unsigned *x_packed, int q, int n, int nrp, float err)
{
  int j, i = 0, t;
  int e, d, b2 = g_b2;
  int rpt = 0, rp;
  int ks, ke, m = 0, k;
  int last = 0;
  uint8 *bprimes = NULL;
  int prime, prime_pair;
  uint8 *rp_gaps = NULL;
  int sprimes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 43, 47, 53, 0};
  uint8 two_to_i[] = {1, 2, 4, 8, 16, 32, 64, 128};
  int count0 = 0, count1 = 0, count2 = 0;
  mpz_t control;
  timeval time0, time1;

  {
    int best_guess = 0x01111111;
    int best_d = 0, best_e = 0, best_nrp = 0;
    int guess;
    int passes;
    int su;
    int nrpe = 0;
    int start_e = 2, end_e = 12;
    int start_d = 9240, d_div = 1;

    if(g_e)
    {
      start_e = g_e;
      end_e = g_e;
    }
    if(g_d)
    {
      start_d = g_d;
      d_div = g_d;
    }
    for(d = start_d; d > 1;d /= d_div)
    {
    if(d >= 2310)
    {
      rpt = d / 2310 * 480;
      i = 4;
    }
    else if(d >= 210)
    {
      rpt = d / 210 * 48;
      i = 3;
    }
    else if(d >= 30)
    {
      rpt = d / 30 * 8;
      i = 2;
    }
    //else if(d >= 6)
   // {
    //  rpt = d / 6 * 2;
    //  i = 1;
    //}
    if(b1 * sprimes[i] * 53 < b2) ks = ((((b1 * 53 + 1) >> 1) + d - 1) / d - 1) * d;
    else if(b1 * sprimes[i] < b2) ks = ((((b2 / sprimes[i] + 1) >> 1) + d - 1) / d - 1) * d;
    else ks = ((((b1 + 1) >> 1) + d - 1) / d - 1) * d;
    ke = ((((b2 + 1) >> 1) + d - 1) / d) * d;
    ks = ((ks / d) << 1) + 1;
    ke = (ke / d) << 1;

    for(e = start_e; e <= end_e; e +=2)
    {
      nrpe = nrp - e - 1;
      if(nrpe <= 0) break;
      passes = (rpt + nrpe - 1) / nrpe;
      while(nrpe > 1 && passes == (rpt + nrpe - 2) / (nrpe - 1)) nrpe--;
      guess = rp_init_count1a(ks, d, e, n) * passes;
      for(su = 0; su < rpt; su += nrpe)guess += rp_init_count1a((su * d / rpt) | 1, 1, e, n);
      guess += 2 * e * (d/2 - passes) - e * e / 2;
      double numprimes = (double) ke*d / (log ((double) ke*d) - 1.0) - (double) b1 / (log ((double) b1) - 1.0);
	    double	numpairings =	numprimes / 2.0 *	numprimes / ((double) ((ke - ks)*d) * (double) rpt / d);
      guess += e * (ke - ks) * passes + (2.2) * (int)(numprimes-numpairings);
      if(e == 4) guess = (int) guess * 0.95;
      if(e == 6) guess = (int) guess * 0.90;
      if(e == 12) guess = (int) guess * 0.85;
      if(guess < best_guess)
      {
        best_guess = guess;
        best_d = d;
        best_e = e;
        best_nrp = nrpe;
      }
    }
    if(d>2310) d -= 2310;
    else if(d>210) d -= 210;
    else if(d>=30) d -= 30;
    //else if(d>=6) d -= 6;
    }
    d = best_d;
    e = best_e;
    nrp = best_nrp;
  }
  if(d == 0) exit(3);

 int end = (q + 31) / 32;
 if(x_packed[end + 10] == 0)
 {
   x_packed[end + 10] = b2;
   x_packed[end + 11] = d;
   x_packed[end + 12] = e;
   x_packed[end + 13] = nrp;
   x_packed[end + 14] = 0; // m = number of relative primes already finished
   x_packed[end + 15] = 0; // k = how far done with currect crop of relative primes
   x_packed[end + 16] = 0; // t = where to find next relativel prime in the bit array
   x_packed[end + 17] = 0; // extra initialization transforms from starting in the middle of a pass
 }
 else
 {
   b1 = x_packed[end + 5];
   b2 = x_packed[end + 10];
   d = x_packed[end + 11];
   e = x_packed[end + 12];
   nrp = x_packed[end + 13];
 }
 g_e = e;
 printf("Using b1 = %d, b2 = %d, d = %d, e = %d, nrp = %d\n",b1, b2,d,e,nrp);


 if(d % 2310 == 0)
	{
	  i = 4;
	  rpt = 480 * d / 2310;
	}
	else if(d % 210 == 0)
	{
	  i = 3;
	  rpt =  48 * d / 210;
	}
	else if(d % 30 == 0)
	{
	  i = 2;
	  rpt = 8 * d / 30;
	}
	else
	{
	  i = 1;
	  rpt = 2 * d / 6;
	}

  if(b1 * sprimes[i] * 53 < b2) ks = ((((b1 * 53 + 1) >> 1) + d - 1) / d - 1) * d;
  else if(b1 * sprimes[i] < b2) ks = ((((b2 / sprimes[i] + 1) >> 1) + d - 1) / d - 1) * d;
  else ks = ((((b1 + 1) >> 1) + d - 1) / d - 1) * d;
  ke = ((((b2 + 1) >> 1) + d - 1) / d) * d;

  bprimes = (uint8*) malloc(ke * sizeof(uint8));
  if(!bprimes)
  {
    printf("failed to allocate bprimes\n");
    exit (1);
  }
  for (j = 0; j < ke; j++) bprimes[j] = 0;
  gtpr(2 * ke, bprimes);
  for(j = 0; j < 10; j++) bprimes[j] = 1;
  bprimes[0] = bprimes[4] = bprimes[7] = 0;

  cutilSafeCall (cudaMalloc ((void **) &e_data, sizeof (double) * n * (e + 1)));
  cutilSafeCall (cudaMalloc ((void **) &rp_data, sizeof (double) * n * nrp));

  for( j = (b1 + 1) >> 1; j < ks; j++)
  {
    if(bprimes[j] == 1)
    {
      m = i;
      last = j;
      while(sprimes[m])
      {
        prime = sprimes[m] * j + (sprimes[m] >> 1);
        m++;
        if(prime < ks) continue;
        if(prime > ke) break;
        prime_pair = prime + d - 1 - ((prime % d) << 1);
        bprimes[last] = 0;
        bprimes[prime] = 1;
        if(bprimes[prime_pair]) break;
        last = prime;
      }
    }
  }

	rp_gaps = (uint8*) malloc(rpt * sizeof(uint8));
  if(!rp_gaps)
  {
    printf("failed to allocate rp_gaps\n");
    exit (1);
  }
  j = 0;
  k = 0;

  for(rp = 1; rp < d; rp += 2)
  {
    k++;
    for (m = 0; m < i; m++)
      if((rp % sprimes[m]) == 0) break;
    if(m == i)
    {
      rp_gaps[j] = k;
      j++;
      k = 0;
    }
  }

	k = ks + (d >> 1);
  m = k - 1;
  j = 0;
  rp = 0;
  uint8 *tprimes = (uint8*) malloc(rpt / 8 * sizeof(uint8));
  int l = 0;
  while(m < ke)
  {
    tprimes[l] = 0;
    for(i = 0; i < 8; i++)
    {
      m += rp_gaps[j];
      k -= rp_gaps[j];
      if (bprimes[m] || bprimes[k])
      {
        tprimes[l] |= two_to_i[i];
        count1++;
      }
      else count0++;
      if (bprimes[m] && bprimes[k]) count2++;
      j++;
      if(j == rpt)
      {
        j = 0;
        m += (d >> 1);
        k = m + 1;
      }
    }
    l++;
    if(l * 8 == rpt)
    {
      for(t = 0; t < (rpt >> 3); t++) bprimes[rp + t] = tprimes[t];
      l = 0;
      rp += rpt >> 3;
    }
  }
  free(tprimes);
  printf("Zeros: %d, Ones: %d, Pairs: %d\n", count0, count1, count2);


  mpz_init(control);
  mpz_import(control, (ke - ks) / d * rpt / sizeof(bprimes[0]) , -1, sizeof(bprimes[0]), 0, 0, bprimes);
	free(bprimes);

  unpack_bits_int(x_int, x_packed, q, n);
  balance_digits_int(x_int, q, n);
  cudaMemcpy (&g_xint[n], x_int, sizeof (int) * n , cudaMemcpyHostToDevice);

  int fp = 1;
  int num_tran = 0, temp_tran;
  int tran_save;
  int itran_tot;
  int ptran_tot;
  int itran_done = 0;
  int ptran_done = 0;
  double checkpoint_int, checkpoint_bnd;
  double time, ptime = 0.0, itime = 0.0;

  ks = ((ks / d) << 1) + 1;
  ke = (ke / d) << 1;
  m = x_packed[end + 14];
  k = x_packed[end + 15];
  t = x_packed[end + 16];
  if(m + k > 0) // some stage 2 has already been done
  {
    itran_done = x_packed[end + 18] + x_packed[end + 17];
    ptran_done = x_packed[end + 19];
    itime = x_packed[end + 20];
    ptime = x_packed[end + 21];
  }

  ptran_tot = (ke - ks - 1) * e * ((rpt + nrp - 1) / nrp) + count1 * 2;
  int passes;
  passes = (rpt + nrp - 1) / nrp;
  itran_tot = rp_init_count1(ks, d, e, n) * passes + x_packed[end + 17];
  int su = 0;
  while (su < rpt)
  {
    if(rpt - su > nrp)
    {
      itran_tot += rp_init_count2(nrp, su, e, n, rp_gaps);
    }
    else
    {
      itran_tot += rp_init_count2(rpt - su, su, e, n, rp_gaps);
    }
    su += nrp;
  }

  if (k == 0) k = ks;
  if(nrp > rpt - m) nrp = rpt - m;
  gettimeofday (&time0, NULL);
  do
  {
    printf("Processing %d - %d of %d relative primes.\n", m + 1, m + nrp, rpt);
    printf("Inititalizing pass... ");
    apply_weights <<<n / (2 * threads1), threads1>>> (g_x, &g_xint[0], g_ttmp);
    E_pre_mul(g_x, g_x, n, 1);
    num_tran = stage2_init_param4(nrp, m, d, e, n, rp_gaps, num_tran, &err);
    temp_tran = num_tran;
    num_tran = stage2_init_param1(k, d, e, n, num_tran, &err);
    apply_weights <<<n / (2 * threads1), threads1>>> (g_x, &g_xint[n], g_ttmp);
    temp_tran = num_tran - temp_tran;
    itran_done += num_tran;
    if((m > 0 || k > ks) && fp)
    {
      x_packed[end + 17] += num_tran;
      itran_tot += num_tran;
    }
    fp = 0;
    cutilSafeCall (cudaMemcpy (&err, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    gettimeofday (&time1, NULL);
    time = 1000000.0 * (double)(time1.tv_sec - time0.tv_sec) + time1.tv_usec - time0.tv_usec;
    itime += time / 1000000.0;

    if(!quitting)
    {
      printf("done. transforms: %d, err = %0.5f, (%0.2f real, %0.4f ms/tran,  ETA ", num_tran,  err, time / 1000000.0, time / (float) (num_tran * 1000));
      if(m == 0 && k == ks) printf("NA");
      else print_time_from_seconds((int) (itime * ((double) itran_tot/itran_done - 1) + ptime * ((double) ptran_tot / ptran_done - 1)));
    }
    printf(")\n");

	  time0.tv_sec = time1.tv_sec;
	  time0.tv_usec = time1.tv_usec;
	  num_tran = 0;
    tran_save = 0;

    checkpoint_int = (ke - ks) / 2 * e + count1 * nrp / (double) rpt;
    int chkp_per_pass;
    chkp_per_pass = RINT_x86(checkpoint_int / checkpoint_iter);
    if(chkp_per_pass == 0) chkp_per_pass = 1;
    int next_checkpoint = ke - 1;
    checkpoint_int = (ke - ks + 1) / (double) chkp_per_pass;
    checkpoint_bnd = ks - 2.0;
    while((int) checkpoint_bnd < k) checkpoint_bnd += checkpoint_int;
    next_checkpoint = RINT_x86(checkpoint_bnd);
    next_checkpoint |= 1;

    for( ; k < ke && !quitting; k += 2)
    {
      int t_last = -1;
      {
        i = nrp - 1;
        while(i && !mpz_tstbit (control, t + i)) i--;
        if(i) t_last = t + i;
      }
      for(j = 0; j < nrp; j++)
      {
        if(mpz_tstbit (control, t))
        {
          E_sub_mul(g_x, g_x, &e_data[0], &rp_data[j * n], n, err, t == t_last);
          num_tran += 2;
          if(num_tran % 200 == 0)
          {
            cutilSafeCall (cudaMemcpy (&err, g_err, sizeof (float), cudaMemcpyDeviceToHost));
            if(err > 0.4) quitting = 2;
          }
          else if(polite_f && num_tran % (2 * polite) == 0) cutilSafeThreadSync();
        }
        t++;
	    }
	    _kbhit();
	    t += rpt - nrp;
	    if(!quitting)
	    {
	      if(k < ke - 1) num_tran = next_base1(k, e, n, num_tran, &err);
        if(num_tran % 200 < 2 * e)
        {
          cutilSafeCall (cudaMemcpy (&err, g_err, sizeof (float), cudaMemcpyDeviceToHost));
          if(err > 0.4) quitting = 2;
        }
        else if(polite_f && num_tran % (2 * polite) < 2 * e) cutilSafeThreadSync();
      }
      if(k == next_checkpoint || quitting == 1)
      {
        checkpoint_bnd += checkpoint_int;
        next_checkpoint = RINT_x86(checkpoint_bnd);
        next_checkpoint |= 1;
        if(quitting == 1) cutilSafeCall (cudaMemcpy (&err, g_err, sizeof (float), cudaMemcpyDeviceToHost));
        if(err <= 0.4f)
        {
          cutilSafeCall (cudaMemcpy (x_int, &g_xint[n], sizeof (int) * n, cudaMemcpyDeviceToHost));
          standardize_digits_int(x_int, q, n, 0, n);
	        pack_bits_int(x_int, x_packed, q, n);
          x_packed[end + 13] = nrp;
          if(k < ke - 1)
          {
            x_packed[end + 14] = m;
            x_packed[end + 15] = k + 2;
            x_packed[end + 16] = t;
          }
          else
          {
            x_packed[end + 14] = m + nrp;
            x_packed[end + 15] = ks;
            x_packed[end + 16] = m + nrp;
          }
          gettimeofday (&time1, NULL);
          time = 1000000.0 * (double)(time1.tv_sec - time0.tv_sec) + time1.tv_usec - time0.tv_usec;
          ptime += time / 1000000.0;
          x_packed[end + 18] = itran_done;
          x_packed[end + 19] = ptran_done + num_tran;
          x_packed[end + 20] = itime;
          x_packed[end + 21] = ptime;
          write_st2_checkpoint(x_packed, q);
          printf ("Transforms: %5d ", num_tran - tran_save);
          printbits_int (x_int, q, n, 0, 0, NULL, 0);
	        printf (" err = %5.5f (", err);
	        print_time_from_seconds ((int) time1.tv_sec - time0.tv_sec);
	        printf (" real, %4.4f ms/tran, ETA ", time / 1000.0 / (num_tran - tran_save));
          print_time_from_seconds((int) itime * ((double) itran_tot/itran_done - 1) + ptime * ((double) ptran_tot / (ptran_done + num_tran) - 1));
          printf(")\n");
          fflush(stdout);
	        tran_save = num_tran;
	        time0.tv_sec = time1.tv_sec;
	        time0.tv_usec = time1.tv_usec;
          reset_err(&err, 0.85f);
        }
      }
    }
    k = ks;
    m += nrp;
    t = m;
    if(rpt - m < nrp) nrp = rpt - m;
	  ptran_done += num_tran;
	  num_tran = 0;
	  printf("\n");
  }
  while(m < rpt && !quitting);
  if(quitting < 2)
  {
    if(!quitting) printf("Stage 2 complete, %d transforms, estimated total time = ", ptran_done + itran_done);
    else printf("Quitting, estimated time spent = ");
    print_time_from_seconds((int) itime + ptime);
    printf("\n");
  }
  else if (quitting == 2) printf ("err = %5.5g >= 0.40, quitting.\n", err);
  free(rp_gaps);
  cutilSafeCall (cudaFree ((char *) e_data));
  cutilSafeCall (cudaFree ((char *) rp_data));
  mpz_clear (control);
  return 0;
}

int
check_pm1 (int q, char *expectedResidue)
{
  int n, j, last = 0;
  int error_flag, checkpoint_flag;
  int *x_int = NULL;
  unsigned *x_packed = NULL;
  float maxerr = 0.0f, terr;
  int restarting = 0;
  timeval time0, time1;
  int total_time = 0, start_time;
  int j_resume = 0;
  int bit;
  unsigned *control = NULL;
  int stage = 0, st1_factor = 0;
  size_t global_mem, free_mem, use_mem;
  int nrp = g_nrp;

  signal (SIGTERM, SetQuitting);
  signal (SIGINT, SetQuitting);

    cudaMemGetInfo(&free_mem, &global_mem);
#ifdef _MSC_VER
    printf("CUDA reports %IuM of %IuM GPU memory free.\n",free_mem/1024/1024, global_mem/1024/1024);
#else
    printf("CUDA reports %zuM of %zuM GPU memory free.\n",free_mem/1024/1024, global_mem/1024/1024);
#endif

  do
  {				/* while (restarting) */
    maxerr = 0.0;

    if(stage == 0)
    {
      if(!x_packed) x_packed = read_checkpoint_packed(q);
      x_int = init_lucas_packed_int (x_packed, q, &n, &j, &stage, &total_time);
    }
    if(!x_int) exit (2);
    if(stage == 2)
    {
      if(read_st2_checkpoint(q, x_packed))
      {
        printf("Stage 2 checkpoint found.\n");
        int end = (q + 31) / 32;
        b1 = x_packed[end + 5];
      }
      else printf("No stage 2 checkpoint.\n");
    }

    g_d = g_d_commandline;
    if(g_nrp == 0) nrp = ((free_mem - (size_t) unused_mem * 1024 * 1024)/ n / 8 - 7);
#ifdef _MSC_VER
    if (nrp > (4096/sizeof(double))*1024*1024/n)
        nrp = (4096/sizeof(double))*1024*1024/n; // Max single allocation of 4 GB on Windows?
#endif
    if(nrp < 4) nrp = 4;
    use_mem = (size_t) (8 * (nrp + 7)* (size_t) n);
#ifdef _MSC_VER
    printf("Using up to %IuM GPU memory.\n",use_mem/1024/1024);
#else
    printf("Using up to %zuM GPU memory.\n",use_mem/1024/1024);
#endif
    if (free_mem < use_mem)
       printf("WARNING:  There may not be enough GPU memory for stage 2!\n");

    double successrate = 0.0;
    if ((g_b1_commandline == 0) || (g_b2_commandline == 0)) {
      guess_pminus1_bounds(q, tfdepth, llsaved, nrp, &b1, &g_b2, &successrate);
    }
    if (g_b1_commandline > 0) b1 = g_b1_commandline;
    if (g_b2_commandline > 0) g_b2 = g_b2_commandline;
    if ((g_b1_commandline == 0) && (g_b2_commandline == 0))
      printf("Selected B1=%d, B2=%d, %0.3g%% chance of finding a factor\n",b1,g_b2,successrate*100);

    if(x_packed[(q + 31)/32 + 5] == 0 || restarting)  x_packed[(q + 31)/32 + 5] = b1;
    else
    {
      b1 = x_packed[(q + 31)/32 + 5];
      printf("Using B1 = %d from savefile.\n", b1);
      fflush(stdout);
    }

    if (g_b2 > 1000000000) printf("WARNING:  Expected failure with B2 > 1000000000!\n"); //max B2 supported?
    fflush(stdout);

    if(stage == 1)
    {
      if(control) free(control);
      control = get_control(&last, b1, 0, q);
    }
    gettimeofday (&time0, NULL);
    start_time = time0.tv_sec;

    restarting = 0;
    if(j == 1)
    {
      printf ("Starting stage 1 P-1, M%d, B1 = %d, B2 = %d, fft length = %dK\n", q, b1, g_b2, n/1024);
      printf ("Doing %d iterations\n", last);
      //restarting = round_off_test(q, n, &j, control, last);
      //if(restarting) stage = 0;
    }
    else
    {
      if(stage == 1)
      {
        printf ("Continuing stage 1 from a partial result of M%d fft length = %dK, iteration = %d\n", q, n/1024, j);
        j_resume = j % checkpoint_iter - 1;
      }
      else
      {
        printf ("Continuing stage 2 from a partial result of M%d fft length = %dK\n", q, n/1024);
      }
    }
    fflush (stdout);

   for (; !restarting && j <= last; j++) // Main LL loop
    {
	    if ((j % 100) == 0) error_flag = 1;
	    else error_flag = 0;
      if ((j % checkpoint_iter == 0) || j == last) checkpoint_flag = 1;
      else checkpoint_flag = error_flag;
      bit = get_bit(last - j, control);
      terr = lucas_square (q, n, j, last, &maxerr, error_flag, bit, stage, checkpoint_flag);
      if(quitting == 1 && !checkpoint_flag)
	    {
	      j++;
        bit = get_bit(last - j, control);
	      terr = lucas_square (q, n, j, last, &maxerr, 1, bit, stage, 1);
	    }
      if (error_flag || quitting == 1)
	    {
	      if (terr >= 0.40)
		    {
		      printf ("Iteration = %d, err = %5.5g >= 0.40, quitting.\n", j, terr);
          quitting = 2;
	      }
	    }
	    if ((j % checkpoint_iter) == 0 || quitting)
      {
	      if(quitting < 2)
	      {
	        cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
          standardize_digits_int(x_int, q, n, 0, n);
	        gettimeofday (&time1, NULL);
          total_time += (time1.tv_sec - start_time);
          start_time = time1.tv_sec;
          set_checkpoint_data(x_packed, q, n, j + 1, stage, total_time);
	        pack_bits_int(x_int, x_packed, q, n);
          write_checkpoint_packed (x_packed, q);
        }
        if(quitting == 0)
        {
          printf ("Iteration %d ", j);
          printbits_int (x_int, q, n, 0, 0, NULL, 0);
	        long long diff = time1.tv_sec - time0.tv_sec;
	        long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	        long long diff2 = (last - j) * diff1 / ((checkpoint_iter - j_resume) *  1e6);
	        gettimeofday (&time0, NULL);
	        printf (" err = %5.5f (", maxerr);
	        print_time_from_seconds ((int) diff);
	        printf (" real, %4.4f ms/iter, ETA ", diff1 / 1000.0 / (checkpoint_iter - j_resume));
	        print_time_from_seconds ((int) diff2);
	        printf (")\n");
	        fflush (stdout);
	        if(j_resume) j_resume = 0;
	        reset_err(&maxerr, 0.85); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
	      }
	      else
	      {
	        printf("Estimated time spent so far: ");
	        print_time_from_seconds(total_time);
	        printf("\n\n");
		      j = last + 1;
	      }
	    }
      if ( k_f && !quitting && (!(j & 15)) && _kbhit()) interact(); // abstracted to clean up check()
	    fflush (stdout);
	  }

    if (!restarting && !quitting)
	  { // done with stage 1
	    if(stage == 1)
	    {
        free ((char *) control);
	      gettimeofday (&time1, NULL);
	      cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
        standardize_digits_int(x_int, q, n, 0, n);
        if(g_eb1 > b1) stage = 3;
        else if(g_b2 > b1) stage = 2;
        set_checkpoint_data(x_packed, q, n, j + 1, stage, total_time);
        pack_bits_int(x_int, x_packed, q, n);
        write_checkpoint_packed (x_packed, q);
        printbits_int (x_int, q, n, 0, NULL , 0, 1);
        total_time += (time1.tv_sec - start_time);
        printf ("\nStage 1 complete, estimated total time = ");
        print_time_from_seconds(total_time);
        fflush (stdout);
        printf("\nStarting stage 1 gcd.\n");
        st1_factor = get_gcd(/*x,*/ x_packed, q, n, 1);
      }
      if(!st1_factor)
      {
        if (stage == 3)
        {
          printf("Here's where we put the b1 extension calls\n");
          stage = 2;
        }
        if(stage == 2)
        {
          printf("Starting stage 2.\n");
          stage2(x_int, x_packed, q, n, nrp, maxerr);
          if(!quitting)
          {
            printf("Starting stage 2 gcd.\n");
            get_gcd(x_packed, q, n, 2);
            rm_checkpoint(q, keep_s1);
          }
        }
      }
	    printf("\n");
	  }
    close_lucas (x_int);
  }
  while (restarting);
  free ((char *) x_packed);
  return (0);
}

void parse_args(int argc, char *argv[], int* q, int* device_numer,
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d);
		/* The rest of the opts are global */
int main (int argc, char *argv[])
{
  printf("%s\n",program);
  quitting = 0;
#define THREADS_DFLT 256
#define CHECKPOINT_ITER_DFLT 10000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define T_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define POLITE_DFLT 1
#define UNMEM_DFLT 100;
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"

  /* "Production" opts to be read in from command line or ini file */
  int q = -1;
  int device_number = -1, f_f = 0;
  checkpoint_iter = -1;
  threads1 = -1;
  fftlen = -1;
  unused_mem = -1;
  s_f = t_f = d_f = k_f = -1;
  polite_f = polite = -1;
  AID[0] = input_filename[0] = RESULTSFILE[0] = 0; /* First character is null terminator */
  char fft_str[132] = "\0";

  /* Non-"production" opts */
  r_f = 0;
  int cufftbench_s, cufftbench_e, cufftbench_d;
  cufftbench_s = cufftbench_e = cufftbench_d = 0;

  parse_args(argc, argv, &q, &device_number, &cufftbench_s, &cufftbench_e, &cufftbench_d);
  /* The rest of the args are globals */

  if (file_exists(INIFILE))
  {
   if( checkpoint_iter < 1 && 		!IniGetInt(INIFILE, "CheckpointIterations", &checkpoint_iter, CHECKPOINT_ITER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
   if( threads1 < 1 && 			!IniGetInt(INIFILE, "Threads", &threads1, THREADS_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: %d\n", THREADS_DFLT);
   if( s_f < 0 && 			!IniGetInt(INIFILE, "SaveAllCheckpoints", &s_f, S_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
   if( 		     	     s_f > 0 && !IniGetStr(INIFILE, "SaveFolder", folder, SAVE_FOLDER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
   if( t_f < 0 && 			!IniGetInt(INIFILE, "CheckRoundoffAllIterations", &t_f, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option CheckRoundoffAllIterations; using default: off\n");
   if(!IniGetInt(INIFILE, "KeepStage1SaveFile", &keep_s1, 0) )
    keep_s1 = 0;
   if( polite < 0 && 			!IniGetInt(INIFILE, "Polite", &polite, POLITE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT);
   if( k_f < 0 && 			!IniGetInt(INIFILE, "Interactive", &k_f, 0) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
   if( device_number < 0 &&		!IniGetInt(INIFILE, "DeviceNumber", &device_number, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n");
   if( d_f < 0 &&			!IniGetInt(INIFILE, "PrintDeviceInfo", &d_f, D_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
   if( !input_filename[0] &&		!IniGetStr(INIFILE, "WorkFile", input_filename, WORKFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
    /* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
   if( !RESULTSFILE[0] && 		!IniGetStr(INIFILE, "ResultsFile", RESULTSFILE, RESULTSFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
   if( fftlen < 0 && 			!IniGetStr(INIFILE, "FFTLength", fft_str, "\0") )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
   if( unused_mem < 0 && 			!IniGetInt(INIFILE, "UnusedMem", &unused_mem, 100) )
    printf("Warning: Couldn't parse ini file option UnusedMem; using default.\n");
  }
  else // no ini file
    {
      fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");
      if( checkpoint_iter < 1 ) checkpoint_iter = CHECKPOINT_ITER_DFLT;
      if( threads1 < 1 ) threads1 = THREADS_DFLT;
      if( fftlen < 0 ) fftlen = 0;
      if( s_f < 0 ) s_f = S_F_DFLT;
      if( t_f < 0 ) t_f = T_F_DFLT;
      if( k_f < 0 ) k_f = K_F_DFLT;
      if( device_number < 0 ) device_number = 0;
      if( d_f < 0 ) d_f = D_F_DFLT;
      if( polite < 0 ) polite = POLITE_DFLT;
      if( unused_mem < 0 ) unused_mem = UNMEM_DFLT;
      if( !input_filename[0] ) sprintf(input_filename, WORKFILE_DFLT);
      if( !RESULTSFILE[0] ) sprintf(RESULTSFILE, RESULTSFILE_DFLT);
  }

  if( fftlen < 0 ) { // possible if -f not on command line
      fftlen = fft_from_str(fft_str);
  }
  if (polite == 0) {
    polite_f = 0;
    polite = 1;
  } else {
    polite_f = 1;
  }
  if (threads1 != 32 && threads1 != 64 && threads1 != 128
	      && threads1 != 256 && threads1 != 512 && threads1 != 1024)
  {
    fprintf(stderr, "Error: thread count is invalid.\n");
    fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
    exit(2);
  }
  f_f = fftlen; // if the user has given an override... then note this length must be kept between tests


  init_device (device_number);
  fft_count = init_ffts();

   if (cufftbench_d)
    cufftbench (cufftbench_s, cufftbench_e, cufftbench_d, device_number);
  else
    {
      if (s_f)
	{
#ifndef _MSC_VER
	  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
	  if (mkdir (folder, mode) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#else
	  if (_mkdir (folder) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#endif
	}
      if (q <= 0)
        {
          int error;

	  #ifdef EBUG
	  printf("Processed INI file and console arguments correctly; about to call get_next_assignment().\n");
	  #endif
	  do
            { // while(!quitting)

	      fftlen = f_f; // fftlen and AID change between tests, so be sure to reset them
	      AID[0] = 0;

  	      error = get_next_assignment(input_filename, &q, &fftlen, &tfdepth, &llsaved, &AID);
               /* Guaranteed to write to fftlen ONLY if specified on workfile line, so that if unspecified, the pre-set default is kept. */
	      if( error > 0) exit (2); // get_next_assignment prints warning message
	      #ifdef EBUG
	      printf("Gotten assignment, about to call check().\n");
	      #endif

              check_pm1 (q, 0);

	      if(!quitting) // Only clear assignment if not killed by user, i.e. test finished
	        {
	          error = clear_assignment(input_filename, q);
	          if(error) exit (2); // prints its own warnings
	        }

	    }
          while(!quitting);
      }
    else // Exponent passed in as argument
      {
	      if (!valid_assignment(q, fftlen)) {printf("\n");} //! v_a prints warning
	      else { //int trft = 0;
              //while(!trft)
              {
                check_pm1 (q, 0);
                //q += 2;
                //while(!isprime(q)) q += 2;
              }
        }
      }
  } // end if(-r) else if(-cufft) else(workfile)
} // end main()

void parse_args(int argc, char *argv[], int* q, int* device_number,
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d)
{
while (argc > 1)
    {
      if (strcmp (argv[1], "-t") == 0)
	{
	  t_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-h") == 0)
        {
      	  fprintf (stderr,
	       "$ CUDAPm1 -h|-v\n\n");
      	  fprintf (stderr,
	       "$ CUDAPm1 [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-t] [-polite iteration] [-k] exponent|input_filename\n\n");
      	  fprintf (stderr,
	       "$ CUDAPm1 [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
      	  fprintf (stderr,
	       "$ CUDAPm1 [-d device_number] [-info] -cufftbench start end distance\n\n");
	  fprintf (stderr,
	       "                       -h          print this help message\n");
	  fprintf (stderr,
	       "                       -v          print version number\n");
	  fprintf (stderr,
	       "                       -info       print device information\n");
	  fprintf (stderr,
	       "                       -i          set .ini file name (default = \"CUDAPm1.ini\")\n");
      	  fprintf (stderr,
	       "                       -threads    set threads number (default = 256)\n");
      	  fprintf (stderr,
	       "                       -f          set fft length (if round off error then exit)\n");
      	  fprintf (stderr,
	       "                       -s          save all checkpoint files\n");
      	  fprintf (stderr,
	       "                       -t          check round off error all iterations\n");
      	  fprintf (stderr,
	       "                       -polite     GPU is polite every n iterations (default -polite 1) (-polite 0 = GPU aggressive)\n");
      	  fprintf (stderr,
	       "                       -cufftbench exec CUFFT benchmark (Ex. $ ./CUDAPm1 -d 1 -cufftbench 1179648 6291456 32768 )\n");
      	  fprintf (stderr,
      	       "                       -r          exec residue test.\n");
      	  fprintf (stderr,
	       "                       -k          enable keys (p change -polite, t disable -t, s change -s)\n\n");
      	  fprintf (stderr,
	       "                       -b2         set b2\n\n");
      	  fprintf (stderr,
	       "                       -d2         Brent-Suyama coefficient (multiple of 30, 210, or 2310) \n\n");
      	  fprintf (stderr,
	       "                       -e2         Brent-Suyama exponent (2-12) \n\n");
      	  //fprintf (stderr,  // Now an internal parameter
	  //     "                       -nrp2       Relative primes per pass (divisor of 8, 48, or 480)\n\n");
      	  exit (2);
      	}
      else if (strcmp (argv[1], "-v") == 0)
        {
          printf("%s\n\n", program);
          exit (2);
        }
      else if (strcmp (argv[1], "-polite") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -polite option\n\n");
	      exit (2);
	    }
	  polite = atoi (argv[2]);
	  if (polite == 0)
	    {
	      polite_f = 0;
	      polite = 1;
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-r") == 0)
	{
	  r_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-k") == 0)
	{
	  k_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-d") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d option\n\n");
	      exit (2);
	    }
	  *device_number = atoi (argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-i") == 0)
	{
	  if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -i option\n\n");
	      exit (2);
	    }
	  sprintf (INIFILE, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-info") == 0)
        {
          d_f = 1;
          argv++;
          argc--;
        }
      else if (strcmp (argv[1], "-cufftbench") == 0)
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
      else if (strcmp (argv[1], "-threads") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threads option\n\n");
	      exit (2);
	    }
	  threads1 = atoi (argv[2]);
	  if (threads1 != 32 && threads1 != 64 && threads1 != 128
	      && threads1 != 256 && threads1 != 512 && threads1 != 1024)
	    {
	      fprintf(stderr, "Error: thread count is invalid.\n");
	      fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-c") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  checkpoint_iter = atoi (argv[2]);
	  if (checkpoint_iter == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-f") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -f option\n\n");
	      exit (2);
	    }
	  fftlen = fft_from_str(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-b1") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -b1 option\n\n");
	      exit (2);
	    }
	  g_b1_commandline = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-e2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -e2 option\n\n");
	      exit (2);
	    }
	  g_e = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-d2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d2 option\n\n");
	      exit (2);
	    }
	  g_d_commandline = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-b2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -b2 option\n\n");
	      exit (2);
	    }
	  g_b2_commandline = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-nrp2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -nrp option\n\n");
	      exit (2);
	    }
	  g_nrp = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-s") == 0)
	{
	  s_f = 1;
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -s option\n\n");
	      exit (2);
	    }
	  sprintf (folder, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-eb1") == 0)
	{
	  s_f = 1;
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -eb1 option\n\n");
	      exit (2);
	    }
	  g_eb1 = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  if (*q != -1 || strcmp (input_filename, "") != 0 )
	    {
	      fprintf (stderr, "can't parse options\n\n");
	      exit (2);
	    }
	  int derp = atoi (argv[1]);
	  if (derp == 0) {
	    sprintf (input_filename, "%s", argv[1]);
	  }
    else {
      *q = derp;
      *q |= 1;
      while(!isprime(*q)) *q += 2;
      }
	  argv++;
	  argc--;
	}
    }
    if (g_d_commandline%30 != 0) {
	printf("-d2 must be a multiple of 30, 210, or 2310.\n");
	exit(3);
    }
    if ((g_e%2 != 0) || (g_e < 0) || (g_e > 12)) {
	printf("-e2 must be 2, 4, 6, 8, 10, or 12.\n");
	exit(3);
    }
}

int interact(void)
{
  int c = getchar ();
  if (c == 'p')
    if (polite_f)
	  {
	    polite_f = 0;
	    printf ("   -polite 0\n");
	  }
    else
     {
      polite_f = 1;
      printf ("   -polite %d\n", polite);
     }
  else if (c == 't')
    {
      t_f = 0;
      printf ("   disabling -t\n");
    }
  else if (c == 's')
    if (s_f == 1)
      {
        s_f = 2;
        printf ("   disabling -s\n");
      }
    else if (s_f == 2)
      {
        s_f = 1;
        printf ("   enabling -s\n");
      }
   if (c == 'F')
      {
         printf(" -- Increasing fft length.\n");
         fftlen++;
          return 1;
      }
   if (c == 'f')
      {
         printf(" -- Decreasing fft length.\n");
         fftlen--;
         return 1;
      }
   if (c == 'k')
      {
         printf(" -- fft length reset cancelled.\n");
         return 2;
      }
   fflush (stdin);
   return 0;
}
