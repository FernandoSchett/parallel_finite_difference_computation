#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define sizeblock 8
#define PI (3.141592653589793)

static float *taperx=NULL,*taperz=NULL;

void read_input(char *file);
int get_int_input(char* name_val);
float get_float_input(char* name_val);
char* get_str_input(char* name_val);
void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt);
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac);
float *calc_coefs(int order);
static void makeo2 (float *coef,int order);
void *alloc1 (size_t n1, size_t size);
void **alloc2 (size_t n1, size_t n2, size_t size);
float *alloc1float(size_t n1);
float **alloc2float(size_t n1, size_t n2);
float ***alloc3float(size_t n1, size_t n2, size_t n3);
int *alloc1int(size_t n1);
void free1 (void *p);
void free2 (void **p);
void free1float(float *p);
void free2float(float **p);
void free3float(float ***p);
void free1int(int *p);
float ricker (float t, float fpeak);
void ricker_wavelet(int nt, float dt, float peak, float *s);
void taper_init(int nxb,int nzb,float F);
void taper_destroy();
void extendvel_linear(int nx,int nz,int nxb,int nzb,float **vel);
