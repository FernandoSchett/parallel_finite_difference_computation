#include "ptsrc.h"
#include "cwp.h"

/* prototype of subroutine used internally */
float ricker (float t, float fpeak);

/*void ptsrc (float xs, float zs,
	int nx, float dx,
	int nz, float dz,
	float ts, float **s)*/

void ptsrc (int xs, int zs, int nx, int nz, float ts, float **s)
/*****************************************************************************
ptsrc - update source pressure function for a point source
******************************************************************************
Input:
xs		x coordinate of point source
zs		z coordinate of point source
nx		number of x samples
dx		x sampling interval
nz		number of z samples
dz		z sampling interval
ts		ricker wavelet

Output:
tdelay		time delay of beginning of source function
s		array[nx][nz] of source pressure at time t+dt
******************************************************************************
Author:  Dave Hale, Colorado School of Mines, 03/01/90
******************************************************************************/
{
	int ix,iz,ixs,izs;
	float xn,zn,xsn,zsn;
	
	/* zero source array */
	/*for (ix=0; ix<nx; ++ix)
		for (iz=0; iz<nz; ++iz)
			s[ix][iz] = 0.0;*/
	
	/* let source contribute within limited distance */
	/*xsn = xs/dx;
	zsn = zs/dz;
	ixs = NINT(xsn);
	izs = NINT(zsn);*/

	ixs = xs;
	izs = zs;
	xsn = xs;
	zsn = zs;

	for (ix=MAX(0,ixs-3); ix<=MIN(nx-1,ixs+3); ++ix) {
		for (iz=MAX(0,izs-3); iz<=MIN(nz-1,izs+3); ++iz) {
			xn = ix-xsn;
			zn = iz-zsn;
			s[ix][iz] += ts*exp(-xn*xn-zn*zn);
		}
	}
}

float ricker (float t, float fpeak)
/*****************************************************************************
ricker - Compute Ricker wavelet as a function of time
******************************************************************************
Input:
t		time at which to evaluate Ricker wavelet
fpeak		peak (dominant) frequency of wavelet
******************************************************************************
Notes:
The amplitude of the Ricker wavelet at a frequency of 2.5*fpeak is 
approximately 4 percent of that at the dominant frequency fpeak.
The Ricker wavelet effectively begins at time t = -1.0/fpeak.  Therefore,
for practical purposes, a causal wavelet may be obtained by a time delay
of 1.0/fpeak.
The Ricker wavelet has the shape of the second derivative of a Gaussian.
******************************************************************************
Author:  Dave Hale, Colorado School of Mines, 04/29/90
******************************************************************************/
{
	float x,xx;
	
	x = PI*fpeak*t;
	xx = x*x;
	/* return (-6.0+24.0*xx-8.0*xx*xx)*exp(-xx); */
	/* return PI*fpeak*(4.0*xx*x-6.0*x)*exp(-xx); */
	return exp(-xx)*(1.0-2.0*xx);
}

void ricker_wavelet(int ns, float dt, float peak, float *s){
	int it;

	for(it = 0; it < ns; it++){
		if(it*dt > 2.0/peak){
			s[it] = 0.0;
		}
		else{
			s[it] = ricker(it*dt - 1.0/peak, peak);
		}
	}
}
