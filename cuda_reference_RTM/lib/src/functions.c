#include "../include/functions.h"

char *file_input;

void read_input(char *file)
{
        file_input = file;	
}

int get_int_input(char* name_val)
{
        FILE *fp;
        fp = fopen(file_input, "r");
        char* line = NULL;
        size_t len = 0;
        if (fp == NULL)
                exit(EXIT_FAILURE);
        while (getline(&line, &len, fp) != -1) 
        {
                if(strstr(line,name_val) != NULL)
                {
                        char *val_char;
                        val_char = strtok(line, "=");
                        val_char = strtok(NULL,"=");
                        return atoi(val_char);
                }
        }
        free(line);
        return -1;
}

float get_float_input(char* name_val)
{
        FILE *fp;
        fp = fopen(file_input, "r");
        char* line = NULL;
        size_t len = 0;
        if (fp == NULL)
                exit(EXIT_FAILURE);
        while (getline(&line, &len, fp) != -1) 
        {
                if(strstr(line,name_val) != NULL)
                {
                        char *val_char;
                        val_char = strtok(line, "=");
                        val_char = strtok(NULL,"=");
                        return atof(val_char);
                }
        }
        free(line);
        return -1.0;
}

char* get_str_input(char* name_val)
{
        FILE *fp;
        fp = fopen(file_input, "r");
        char* line = NULL;
        size_t len = 0;
        if (fp == NULL)
                exit(EXIT_FAILURE);
        while (getline(&line, &len, fp) != -1) 
        {
                if(strstr(line,name_val) != NULL)
                {
                        char *val_char;
                        val_char = strtok(line, "=");
                        val_char = strtok(NULL,"=");
                        val_char[strlen(val_char) - 1] = '\0';
                        return val_char;
                }
        }
        free(line);
        return NULL;
}

// ============================ Aux ============================
float *calc_coefs(int order)
{
        float *coef;

        coef = (float *)calloc(order+1,sizeof(float));
        switch(order)
        {
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

static void makeo2 (float *coef,int order)
{
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

void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

void **alloc2 (size_t n1, size_t n2, size_t size)
{
	size_t i2;
	void **p;

	if ((p=(void**)malloc(n2*sizeof(void*)))==NULL) 
		return NULL;
	if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
		free(p);
		return NULL;
	}
	for (i2=0; i2<n2; i2++)
		p[i2] = (char*)p[0]+size*n1*i2;
	return p;
}

void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
{
	size_t i3,i2;
	void ***p;

	if ((p=(void***)malloc(n3*sizeof(void**)))==NULL)
		return NULL;
	if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void*)malloc(n3*n2*n1*size))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}

	for (i3=0; i3<n3; i3++) {
		p[i3] = p[0]+n2*i3;
		for (i2=0; i2<n2; i2++)
			p[i3][i2] = (char*)p[0][0]+size*n1*(i2+n2*i3);
	}
	return p;
}

float *alloc1float(size_t n1)
{
	return (float*)alloc1(n1,sizeof(float));
}

float **alloc2float(size_t n1, size_t n2)
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

int *alloc1int(size_t n1)
{
	return (int*)alloc1(n1,sizeof(int));
}

void free1 (void *p)
{
	free(p);
}

void free2 (void **p)
{
	free(p[0]);
	free(p);
}

void free3 (void ***p)
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

void free1float(float *p)
{
	free1(p);
}

void free2float(float **p)
{
	free2((void**)p);
}

void free3float(float ***p)
{
	free3((void***)p);
}

void free1int(int *p)
{
	free1(p);
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
	return exp(-xx)*(1.0-2.0*xx);
}

void ricker_wavelet(int nt, float dt, float peak, float *s)
{
	int it;
	for(it = 0; it < nt; it++){
		s[it] = ricker(it*dt - 1.0/peak, peak);
	}
}

void extendvel_linear(int nx,int nz,int nxb,int nzb,float **vel){
	int ix,iz;
	float v=0,v_ave=0,l_lim = 300.,delta = 200.;
	//fprintf(stdout,"\nRAND_MAX = %d\n",RAND_MAX); // Check max randomic number RAND_MAX

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++){ 
			/* borda superior */
			vel[ix+nxb][iz] = vel[ix+nxb][nzb];
			
			/* borda inferior */
			v = vel[ix+nxb][nzb+nz-1];
			v_ave = v - (v - l_lim)*(iz)/(nzb-1);
			vel[ix+nxb][nz+nzb+iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			
		}
	}
	for(iz=0;iz<nz;iz++){
		for(ix=0;ix<nxb;ix++){									
			/* borda esquerda */
			v = vel[nxb][nzb+iz];	
			v_ave = v - (v - l_lim)*(ix)/(nxb-1);
			vel[nxb-1-ix][nzb+iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			
			/* borda direita */
			v = vel[nxb+nx-1][nzb+iz];	
			v_ave = v - (v - l_lim)*(ix)/(nxb-1);
			vel[nxb+nx+ix][nzb+iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;			
		}
	}

	/* Canto superior esquerdo e direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<nxb;ix++){
			vel[ix][iz] = vel[nxb][iz];
			vel[nxb+nx+ix][iz]= vel[nxb+nx-1][iz];
		}
	}

	/* Canto inferior esquerdo */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb][nzb+nz-1];
			v_ave = v - (v - l_lim)*(nxb-1-ix)/(nzb-1);
			vel[ix][nz+2*nzb-1-iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			vel[iz][nz+2*nzb-1-ix] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
		}
	}

	/* Canto inferior direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb+nx-1][nzb+nz-1];
			v_ave = v - (v - l_lim)*(nxb-1-ix)/(nzb-1);
			vel[nx+2*nxb-1-ix][nz+2*nzb-1-iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			vel[nx+2*nxb-1-iz][nz+2*nzb-1-ix] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
		}
	}
}

void taper_init(int nxb,int nzb,float F){
	int i;
	float dfrac;
	taperx = alloc1float(nxb);
	taperz = alloc1float(nzb);

	dfrac = sqrt(-log(F))/(1.*nxb);

	for(i=0;i<nxb;i++){
		taperx[i] = exp(-pow((dfrac*(nxb-i)),2));
	}

	dfrac = sqrt(-log(F))/(1.*nzb);

	for(i=0;i<nzb;i++){
		taperz[i] = exp(-pow((dfrac*(nzb-i)),2));
	}
	return;
}

void taper_destroy(){
	free1float(taperx);
	free1float(taperz);
	return;
}