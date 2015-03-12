build with 

make

run with e.g.

./CUDAPm1 61408363 -b1 600000 -b2 12000000 -f 3360k

I am primarily interested in

1. Does it run without errors
2. Does it give correct results
and much less important at this stage
3. How long does it take with various d, e, nrp

LIMITATIONS

b2 is the upper bound for stage2
d = multiples of 30, 210, or 2310. I have only tested with d = 30, 210, and 2310, 
not any higher multiples yet.
e = 2, 4, 6, 8, 10, or 12, the B-S exponent.
nrp = number of relative primes per pass.

The program can be terminated with ^C during stage1, not during stage2.

No sanity check is done on the values e and rn which affect memory use. 
If these values are too high, Stage2 will teminate immediately with a cuda or cufft error, 
possibly cufftSafeCall() CUFFT error 2: CUFFT_ALLOC_FAILED. 
Current memory use is m = ~(75 + 8*(nrp + e + 1)) * n bytes where n is the fft size.
cufft uses additional memory. Its probably safe to keep 
m < total memory - 100Mb - (base usage if used to drive the display)


KNOWN ISSUES

rn must be a divisor of d, haven't tried it, but a seg fault is likely otherwise.

with p = smallest prime that does not divide d: 
b1 <  b2 / p / 53 will not pair some smaller primes, so will possibly give incorrect results.
b2 / p < 2 * e + 1 will give incorrect results
b2 / p < b1 will produce a seg fault at the onset of stage2.

A few exponents with known factors, with min b1 and b2 necessary to find them, and a reasonable fft length to use:
50001781  94,709	4,067,587  2688k
51558151	 5,953	2,034,041  2880k
54447193	 1,181	  682,009  3072k
58610467	70,843	  694,201  3200k
61012769	10,273	1,572,097  3360k

Feel free to pick your own.
