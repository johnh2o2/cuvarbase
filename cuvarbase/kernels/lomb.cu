#include <pycuda-complex.hpp>
#include <stdio.h>
//{CPP_DEFS}

#define EPSILON 1E-8
#define PI 3.141592653589793238462643383279502884f
#ifdef DOUBLE_PRECISION
	#define FLT double
#else
	#define FLT float
#endif



__device__ FLT cossum(FLT *t, FLT *y, int n, FLT freq){
	FLT C = 0;
	for(int i = 0; i < n; i++)
		C += y[i] * cos((t[i] + 0.5f) * freq * 2.f * PI);

	return C;
}


__device__ FLT sinsum(FLT *t, FLT *y, int n, FLT freq){
	FLT S = 0;
	for(int i = 0; i < n; i++)
		S += y[i] * sin((t[i] + 0.5f) * freq * 2.f * PI);

	return S;
}

__device__ FLT lpow(FLT C, FLT S, FLT C2, FLT S2, FLT Ch, FLT Sh, FLT YY){

	FLT tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S));

	FLT C2wInv2 = 1.f + tan_2omega_tau * tan_2omega_tau;

	FLT C2w = 1.f / sqrt(C2wInv2);
	FLT S2w = tan_2omega_tau * C2w;

	FLT Cw = sqrt(0.5f * (1.f + C2w));
	FLT Sw = sqrt(0.5f * (1.f - C2w));

	if (S2w < 0)
		Sw *= -1.f;

    FLT YC = Ch * Cw + Sh * Sw;
	FLT YS = Sh * Cw - Ch * Sw;
	FLT CC = 0.5f * (1.f + C2 * C2w + S2 * S2w);
	FLT SS = 0.5f * (1.f - C2 * C2w - S2 * S2w);

    CC -= (C * Cw + S * Sw) * (C * Cw + S * Sw);
    SS -= (S * Cw - C * Sw) * (S * Cw - C * Sw);

    FLT P = ((YC * YC) / CC + (YS * YS) / SS)/YY;

    if (isnan(P) || isinf(P) || P < 0 || P > 1)
    	P = -1.;

    return P;
}


__global__ void lomb_dirsum(FLT *t, FLT *yw, FLT *w,
							FLT *lsp,
							int nfreq, int n, FLT YY, FLT df, FLT fmin){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c)
	if (i < nfreq){

		FLT frq = fmin + i * df;

		FLT C = cossum(t, w, n, frq);
		FLT S = sinsum(t, w, n, frq);

		FLT C2 = cossum(t, w, n, 2.f * frq);
		FLT S2 = sinsum(t, w, n, 2.f * frq);

		FLT Ch = cossum(t, yw, n, frq);
		FLT Sh = sinsum(t, yw, n, frq);

   		lsp[i] = lpow(C, S, C2, S2, Ch, Sh, YY);
   	}
}

__global__ void lomb_dirsum_custom_frq(FLT *t, FLT *w, FLT *yw, FLT *freqs,
							FLT *lsp,
							int nfreq, int n, FLT YY){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c)
	if (i < nfreq){

		FLT frq = freqs[i];

		FLT C = cossum(t, w, n, frq);
		FLT S = sinsum(t, w, n, frq);

		FLT C2 = cossum(t, w, n, 2.f * frq);
		FLT S2 = sinsum(t, w, n, 2.f * frq);

		FLT Ch = cossum(t, yw, n, frq);
		FLT Sh = sinsum(t, yw, n, frq);

   		lsp[i] = lpow(C, S, C2, S2, Ch, Sh, YY);
   	}
}

__global__ void lomb(pycuda::complex<FLT>  *sw,
					 pycuda::complex<FLT>  *syw,
					 FLT *lsp,
					 int nfreq, FLT YY){

	// least squares (lomb scargle with FLTing mean)

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c)
	if (i < nfreq && i > 0){
		pycuda::complex<FLT> SW, SW2, SYW;
		SW = sw[i];
		SW2 = sw[2 * i];
		SYW = syw[i];

		FLT C = SW.real();
		FLT S = SW.imag();

		FLT C2 = SW2.real();
		FLT S2 = SW2.imag();

		FLT Ch = SYW.real();
		FLT Sh = SYW.imag();

		lsp[i-1] = lpow(C, S, C2, S2, Ch, Sh, YY);
	}
}
