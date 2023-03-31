#ifndef PTSRC_H
#define PTSRC_H

void ptsrc (int xs, int zs, int nx, int nz, float ts, float **s);
void ricker_wavelet(int ns, float dt, float peak, float *s);
float ricker (float t, float fpeak);

#endif
