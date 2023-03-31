#include "cwp.h"
#include "fd.h"

static float dx2inv,dz2inv,dt2;
static float **laplace = NULL;
static float *coefs = NULL;
static void makeo2 (float *coef,int order);

float *calc_coefs(int order);

void fd_init(int order, int nx, int nz, float dx, float dz, float dt){
	dx2inv = (1./dx)*(1./dx);
        dz2inv = (1./dz)*(1./dz);
	dt2 = dt*dt;

	coefs = calc_coefs(order);
	laplace = alloc2float(nz,nx);
	
	memset(*laplace,0,nz*nx*sizeof(float));

	return;
}

void fd_step(int order, float **p, float **pp, float **v2, int nz, int nx){
	int ix,iz,io;
	float acm = 0;

	for(ix=order/2;ix<nx-order/2;ix++){
		for(iz=order/2;iz<nz-order/2;iz++){
			for(io=0;io<=order;io++){
				acm += p[ix][iz+io-order/2]*coefs[io]*dz2inv;
				acm += p[ix+io-order/2][iz]*coefs[io]*dx2inv;
			}
			laplace[ix][iz] = acm;
			acm = 0.0;
		}
	}

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nz;iz++){
			pp[ix][iz] = 2.*p[ix][iz] - pp[ix][iz] + v2[ix][iz]*dt2*laplace[ix][iz];
		}
	}

	return;
}

void fd_destroy(){
	free2float(laplace);
	free1float(coefs);
	return;
}

float *calc_coefs(int order){
	float *coef;

	coef = (float*) calloc(order+1,sizeof(float));

	switch(order){
		case 2:
			coef[0] = 1.;
			coef[1] = -2.;
			coef[2] = 1.;
			break;
		case 4:
			coef[0] = -1./12.;
			coef[1] = 4./3.;
			coef[2] = -5./2.;
			coef[3] = 4./3.;
			coef[4] = -1./12.;
			break;
		case 6:
			coef[0] = 1./90.;
			coef[1] = -3./20.;
			coef[2] = 3./2.;
			coef[3] = -49./18.;
			coef[4] = 3./2.;
			coef[5] = -3./20.;
			coef[6] = 1./90.;
			break;
		case 8:
			coef[0] = -1./560.;
			coef[1] = 8./315.;
			coef[2] = -1./5.;
			coef[3] = 8./5.;
			coef[4] = -205./72.;
			coef[5] = 8./5.;
			coef[6] = -1./5.;
			coef[7] = 8./315.;
			coef[8] = -1./560.;
			break;
		default:
			makeo2(coef,order);
	}

	return coef;
}

static void makeo2 (float *coef,int order){
	float h_beta, alpha1=0.0;
	float alpha2=0.0;
	float  central_term=0.0; 
	float coef_filt=0; 
	float arg=0.0; 
	float  coef_wind=0.0;
	int msign,ix; 
	float alpha = .54;
	float beta = 6.;

	h_beta = 0.5*beta;
	alpha1=2.*alpha-1.0;
	alpha2=2.*(1.0-alpha);
	central_term=0.0;

	msign=-1;

	for (ix=1; ix <= order/2; ix++){      
		msign=-msign ;            
		coef_filt = (2.*msign)/(ix*ix); 
		arg = PI*ix/(2.*(order/2+2));
		coef_wind=pow((alpha1+alpha2*cos(arg)*cos(arg)),h_beta);
		coef[order/2+ix] = coef_filt*coef_wind;
		central_term = central_term + coef[order/2+ix]; 
		coef[order/2-ix] = coef[order/2+ix]; 
	}
	
	coef[order/2]  = -2.*central_term;

	return; 
}
