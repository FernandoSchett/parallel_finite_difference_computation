#include "taper.h"
#include "cwp.h"

static float *taperx=NULL,*taperz=NULL;
/*static void damp(int nbw, float abso, float *bw);*/

void extendvel(int nx,int nz,int nxb,int nzb,float *vel){
	int ix,iz;
	int rnz = nz+2.*nzb;

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++)
			vel[(ix+nxb)*rnz+iz] = vel[(ix+nxb)*rnz+nzb];
		for(iz=nzb+nz;iz<nz+2*nzb;iz++)
			vel[(ix+nxb)*rnz+iz] = vel[(ix+nxb)*rnz+nz+nzb-1];
	}
	for(iz=0;iz<nz+2*nzb;iz++){
		for(ix=0;ix<nxb;ix++)
			vel[ix*rnz+iz] = vel[nxb*rnz+iz];
		for(ix=nxb+nx;ix<nx+2*nxb;ix++)
			vel[ix*rnz+iz] = vel[(nx+nxb-1)*rnz+iz];
	}
}

void taper_init(int nxb,int nzb,float F){
	int i;
	/*float dfrac;*/
	taperx = alloc1float(nxb);
	taperz = alloc1float(nzb);

	/*dfrac = sqrt(-log(F))/(1.*nxb);*/

	for(i=0;i<nxb;i++){
		/*taperx[i] = exp(-pow((dfrac*(nxb-i)),2));*/
		taperx[i] = exp(-pow((F*(nxb-i)),2));
	}

	/*dfrac = sqrt(-log(F))/(1.*nzb);*/

	for(i=0;i<nzb;i++){
		taperz[i] = exp(-pow((F*(nzb-i)),2));
		/*printf("%d , %f \n",i,taperz[i]);*/
	}
	return;
}

void taper_apply(float **pp,int nx, int nz, int nxb, int nzb){
	int itz, itx,i;

	for(itx=0;itx<nx+2*nxb;itx++){
		for(itz=0;itz<nzb;itz++){
			pp[itx][itz] *= taperz[itz];
		}
		for(itz=nzb-1,i=0;itz>-1;itz--,i++){
			pp[itx][nz+nzb+i] *= taperz[itz];
		}
	}
	for(itz=0;itz<nz+2*nzb;itz++){
		for(itx=0;itx<nxb;itx++){
			pp[itx][itz] *= taperx[itx];
		}
		for(itx=nxb-1,i=0;itx>-1;itx--,i++){
			pp[nx+nxb+i][itz] *= taperx[itx];
		}
	}
	return;
}

void taper_apply2(float **pp,int nx, int nz, int nxb, int nzb){
        int itz, itx, itxr;

        for(itx=0;itx<nx+2*nxb;itx++){
                for(itz=0;itz<nzb;itz++){
                        pp[itx][itz] *= taperz[itz];
                }
        }
        for(itx=0,itxr=nx+2*nxb-1;itx<nxb;itx++,itxr--){
                for(itz=0;itz<nzb;itz++){
                        pp[itx][itz]  *= taperx[itx];
                        pp[itxr][itz] *= taperx[itx];
                }
        }
        return;
}


void taper_destroy(){
	free1float(taperx);
	free1float(taperz);
	return;
}

/*static void damp(int nbw, float abso, float *bw)
{
	int i;
	float pi, delta;

	pi = 4. * atan(1.);
	delta = pi / nbw;

	for (i=0; i<nbw; i++) {
		bw[i] = 1.0 - abso * (1.0 + cos(i*delta)) * 0.5;
	}

	return;
}*/

