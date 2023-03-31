#ifndef TAPER_H
#define TAPER_H

void extendvel(int nx,int nz,int nxb,int nzb,float *vel);
void taper_init(int nxb,int nzb,float F);
void taper_apply(float **pp,int nx, int nz, int nxb, int nzb);
void taper_apply2(float **pp,int nx, int nz, int nxb, int nzb); 
void taper_destroy();

#endif

