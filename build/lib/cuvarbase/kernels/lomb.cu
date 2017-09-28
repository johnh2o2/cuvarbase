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

#define STANDARD 0
#define FLOATING_MEAN 1
#define WINDOW 2



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

__device__ FLT lspow_flmean(FLT C, FLT S, 
	                        FLT C2, FLT S2, 
	                        FLT YCh, FLT YSh, 
	                        FLT YY, FLT Y, 
	                        FLT *reg){
	FLT r0 = 0.f, r1 = 0.f, r2 = 0.f;
	if (reg != NULL){
		r0 = reg[0];
		r1 = reg[1];
		r2 = reg[2];
	}
	FLT tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S));

	FLT C2wInv2 = 1.f + tan_2omega_tau * tan_2omega_tau;

	FLT C2w = 1.f / sqrt(C2wInv2);
	FLT S2w = tan_2omega_tau * C2w;

	FLT Cw = sqrt(0.5f * (1.f + C2w));
	FLT Sw = sqrt(0.5f * (1.f - C2w));

	if (S2w < 0.f)
		Sw *= -1.f;

	FLT Cshft = C * Cw + S * Sw;
	FLT Sshft = S * Cw - C * Sw;

	FLT CC = 0.5f * (1.f + C2 * C2w + S2 * S2w);
	FLT SS = 0.5f * (1.f - C2 * C2w - S2 * S2w);

	CC -= Cshft * Cshft;
    SS -= Sshft * Sshft;

    FLT xreg = r2 / (1.f + r2);

    CC += Cshft * Cshft * xreg + r0;
    SS += Sshft * Sshft * xreg + r1;

    FLT YC = (YCh + Y * C * xreg) * Cw + (YSh + Y * S * xreg) * Sw;
	FLT YS = (YSh + Y * S * xreg) * Cw - (YCh + Y * C * xreg) * Sw;
    
    FLT P = ((YC * YC) / CC + (YS * YS) / SS) / YY;

    if (isnan(P) || isinf(P) || P < 0.f)
    	P = -1.;

    return P;
}

__device__ FLT lspow0(FLT C, FLT S, 
					  FLT C2, FLT S2, 
					  FLT YCh, FLT YSh, 
					  FLT YY, FLT Y,
					  FLT *reg){

	FLT tan_2omega_tau = S2 / C2;
	FLT r0 = 0.f, r1 = 0.f;
	if (reg != NULL){
		r0 = reg[0];
		r1 = reg[1];
	}

	FLT C2wInv2 = 1.f + tan_2omega_tau * tan_2omega_tau;

	FLT C2w = 1.f / sqrt(C2wInv2);
	FLT S2w = tan_2omega_tau * C2w;

	FLT Cw = sqrt(0.5f * (1.f + C2w));
	FLT Sw = sqrt(0.5f * (1.f - C2w));

	if (S2w < 0)
		Sw *= -1.f;

    FLT YC = (YCh + Y * C) * Cw + (YSh + Y * S) * Sw;
	FLT YS = (YSh + Y * S) * Cw - (YCh + Y * C) * Sw;

	FLT CC = 0.5f * (1.f + C2 * C2w + S2 * S2w) + r0;
	FLT SS = 0.5f * (1.f - C2 * C2w - S2 * S2w) + r1;

    FLT P = ((YC * YC) / CC + (YS * YS) / SS) / (YY + Y * Y);

    if (isnan(P) || isinf(P) || P < 0.f){
    	//printf("%e, %e, %e, %e, %e: %e\n", C, S, CC, SS, YY + Y*Y, P);
    	P = -1.f;
    }

    return P;
}


__device__ FLT lspow(FLT C, FLT S, 
	                 FLT C2, FLT S2, 
	                 FLT YCh, FLT YSh, 
	                 FLT YY, FLT Y, 
	                 FLT *reg, int mode){
	switch(mode){
	 	case STANDARD:
		 	return lspow0(C, S, C2, S2, YCh, YSh, YY, Y, reg);
		case FLOATING_MEAN:
			return lspow_flmean(C, S, C2, S2, YCh, YSh, YY, Y, reg);
		case WINDOW:
			return lspow0(C, S, C2, S2, C, S, 0.f, 1.f, NULL);
		default:
			return -1.f;
	}
}


__global__ void lomb_dirsum(FLT *t, FLT *yw, FLT *w,
							FLT *lsp, FLT *reg,
							int nfreq, int n, FLT YY, FLT Y, FLT df, 
							FLT fmin, int mode){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c)
	if (i < nfreq){

		FLT frq = fmin + i * df;

		FLT C = cossum(t, w, n, frq);
		FLT S = sinsum(t, w, n, frq);

		FLT C2 = cossum(t, w, n, 2.f * frq);
		FLT S2 = sinsum(t, w, n, 2.f * frq);

		FLT YCh = cossum(t, yw, n, frq);
		FLT YSh = sinsum(t, yw, n, frq);

		lsp[i] = lspow(C, S, C2, S2, YCh, YSh, YY, Y, reg, mode);
   	}
}

__global__ void lomb_dirsum_custom_frq(FLT *t, FLT *w, FLT *yw, FLT *freqs,
							FLT *lsp, FLT *reg,
							int nfreq, int n, FLT YY, FLT Y, int mode){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c)
	if (i < nfreq){

		FLT frq = freqs[i];

		FLT C = cossum(t, w, n, frq);
		FLT S = sinsum(t, w, n, frq);

		FLT C2 = cossum(t, w, n, 2.f * frq);
		FLT S2 = sinsum(t, w, n, 2.f * frq);

		FLT YCh = cossum(t, yw, n, frq);
		FLT YSh = sinsum(t, yw, n, frq);

   		lsp[i] = lspow(C, S, C2, S2, YCh, YSh, YY, Y, reg, mode);
   	}
}

__global__ void lomb(pycuda::complex<FLT>  *sw,
					 pycuda::complex<FLT>  *syw,
					 FLT *lsp,
					 FLT *reg,
					 int nfreq, 
					 FLT YY, 
					 FLT Y, 
					 int k0, 
					 int mode){

	// least squares (lomb scargle with FLTing mean)

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c)
	if (i < nfreq){
		pycuda::complex<FLT> SW, SW2, SYW;
		SW = sw[i];
		SW2 = sw[2 * i + k0];
		SYW = syw[i];

		FLT C = SW.real();
		FLT S = SW.imag();

		FLT C2 = SW2.real();
		FLT S2 = SW2.imag();

		FLT YCh = SYW.real();
		FLT YSh = SYW.imag();

        lsp[i] = lspow(C, S, C2, S2, YCh, YSh, YY, Y, reg, mode);
	}
}