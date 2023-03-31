#ifndef FD_H
#define FD_H

void fd_init(int order, int nx, int nz, float dx, float dz, float dt);
void fd_step(int order, float **p, float **pp, float **v2, int nz, int nx);
void fd_destroy();
float *calc_coefs(int order);

#endif
