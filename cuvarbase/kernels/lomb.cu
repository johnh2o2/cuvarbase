#include <pycuda-complex.hpp>


__global__ void lomb(pycuda::complex<float>  *sw, 
					 pycuda::complex<float>  *syw, 
					 float *lsp,
					 int nfreq, float YY, float *reg){

	// least squares (lomb scargle with floating mean)

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// reg = (lambda_a, lambda_b, lambda_c) 
	if (i < nfreq && i > 0){
		pycuda::complex<float> SW, SW2, SYW;
		SW = sw[2 * nfreq + i];
		SW2 = sw[2 * nfreq + 2 * i];
		SYW = syw[nfreq + i];

		float C = SW.real();
		float S = SW.imag();

		float C2 = SW2.real();
		float S2 = SW2.imag();

		float YC = SYW.real();
		float YS = SYW.imag();

		//float CC = reg[0] + 0.5f * ( 1.f + C2 ) - C * C / (1.f + reg[2]);
		//float CS = 0.5f * S2 - C * S / (1.f + reg[2]);
		//float SS = reg[1] + 0.5f * ( 1.f - C2 ) - S * S / (1.f + reg[2]);
		float CC= 0.5f * (1.f + C2) - C * C;
		float CS = 0.5f * S2 - C * S;
		float SS = 0.5f * (1.f - C2) - S * S;

		float D = CC * SS - CS * CS;

		lsp[i-1] = (SS * YC * YC + CC * YS * YS - 2 * CS * YC * YS) / (YY * D);
	}
}
