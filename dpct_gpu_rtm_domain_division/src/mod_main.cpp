/* Acoustic wavefield modeling using finite-difference method
Leonardo Gómez Bernal, Salvador BA, Brazil
August, 2016 */

#include<stdio.h>
#include"su.h"

#include "fd.h"
#include "ptsrc.h"
#include "taper.h"

char *sdoc[] = {	/* self documentation */
	" Seismic migration using acoustic wave equation - RTM ",
	"				                       ",
	NULL};

/* global variables */

/* file names */
char *tmpdir = NULL, *vpfile = NULL, *fdatfile = NULL, file[100];

/* size */
int nz, nx, nt;
float dz, dx, dt;

/* adquisition geometry */
int ns, sz, fsx, ds, gz;

/* boundary */
int nxb, nzb, nxe, nze;
float fac;

/* propagation */
int order; 
float fpeak;

/* arrays */
int *sx;

/* prototypes */

int main (int argc, char **argv){
	/* model file and data pointers */
	FILE *fvp = NULL, *fdat = NULL;

	/* iteration variables */
	int iz, ix, it, is;

	/* arrays */
	float *srce;
	float **vp = NULL;

	/* propagation variables */
	float **PP,**P,**tmp;
	float **vel2, ***data;

	/* initialization admiting self documentation */
	initargs(argc, argv);
	requestdoc(1);

	/*initargs(argc, argv);
	srand(time(0));*/

	/* read parameters */
	MUSTGETPARSTRING("tmpdir",&tmpdir);		// directory for data
	MUSTGETPARSTRING("vpfile",&vpfile);		// vp model
	MUSTGETPARSTRING("datfile",&fdatfile);		// dobs model
	MUSTGETPARINT("nz",&nz); 			// number of samples in z
	MUSTGETPARINT("nx",&nx); 			// number of samples in x
	MUSTGETPARINT("nt",&nt); 			// number of time steps
	MUSTGETPARFLOAT("dz",&dz); 			// sampling interval in z
	MUSTGETPARFLOAT("dx",&dx); 			// sampling interval in x
	MUSTGETPARFLOAT("dt",&dt); 			// sampling interval in t
	MUSTGETPARFLOAT("fpeak",&fpeak); 		// souce peak frequency

	if(!getparint("ns",&ns)) ns = 1;	 	// number of sources
	if(!getparint("sz",&sz)) sz = 0; 		// source depth
	if(!getparint("fsx",&fsx)) fsx = 0; 		// first source position
	if(!getparint("ds",&ds)) ds = 1; 		// source interval
	if(!getparint("gz",&gz)) gz = 0; 		// receivor depth

	if(!getparint("order",&order)) order = 8;	// FD order
	if(!getparint("nzb",&nzb)) nzb = 40;		// z border size
	if(!getparint("nxb",&nxb)) nxb = 40;		// x border size
	if(!getparfloat("fac",&fac)) fac = 0.7;		// damping factor

	fprintf(stdout,"## vp = %s \n",vpfile);
	fprintf(stdout,"## nz = %d, nx = %d, nt = %d \n",nz,nx,nt);
	fprintf(stdout,"## dz = %f, dx = %f, dt = %f \n",dz,dx,dt);
	fprintf(stdout,"## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",ns,sz,fsx,ds,gz);
	fprintf(stdout,"## order = %d, nzb = %d, nxb = %d, F = %f \n",order,nzb,nxb,fac);

	/* create source vector  */
	srce = alloc1float(nt);
	ricker_wavelet(nt, dt, fpeak, srce);

	sx = alloc1int(ns);
	for(is=0; is<ns; is++){
		sx[is] = fsx + is*ds + nxb;
	}
	sz += nzb;
	gz += nzb;

	/* add boundary to models */
	nze = nz + 2 * nzb;
	nxe = nx + 2 * nxb;

	/* read parameter models */
	vp = alloc2float(nz,nx);
	memset(*vp,0,nz*nx*sizeof(float));

	fvp = fopen(vpfile,"r");
	
	fread(vp[0],sizeof(float),nz*nx,fvp);
	fclose(fvp);

	/* initialize velocity */
	vel2 = alloc2float(nze,nxe);

	for(ix=0; ix<nx; ix++){
		for(iz=0; iz<nz; iz++){
			vel2[ix+nxb][iz+nzb] = vp[ix][iz]*vp[ix][iz];
		}
	}

	extendvel(nx,nz,nxb,nzb,*vel2);

	/* initialize wave propagation */
	fd_init(order,nxe,nze,dx,dz,dt);
	taper_init(nxb,nzb,fac);

	PP = alloc2float(nze,nxe);
	P = alloc2float(nze,nxe);
	data = alloc3float(nt,nx,ns);

	fdat = fopen(fdatfile,"w+");

	memset(**data,0,nx*nt*ns*sizeof(float));

	for(is=0; is<ns; is++){
		fprintf(stdout,"** source %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));

		for(it=0; it<nt; it++){
			/* propagate to t+dt */
			fd_step(order,P,PP,vel2,nze,nxe);

			/* add source */
			//PP[sx[is]][sz] += srce[it];
			ptsrc(sx[is],sz,nxe,nze,srce[it],PP);

			/* boundary conditions */
			taper_apply(PP,nx,nz,nxb,nzb);
			taper_apply(P,nx,nz,nxb,nzb);

			/* save data and source wavefield */
			for(ix=0; ix<nx; ix++){
				data[is][ix][it] = P[ix+nxb][gz];
			}

			if(it%100 == 0)fprintf(stdout,"* it = %d / %d \n",it,nt);

			tmp = PP;
			PP = P;
			P = tmp;
		}

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
	}

	fwrite(**data,sizeof(float),ns*nt*nx,fdat);
	/* save stacked image */
	// fwrite(*img,sizeof(float),nz*nx,fimg);

	fclose(fdat);

	fd_destroy();
	taper_destroy();

	/*FILE *fp3;
	sprintf(file,"%s/models.bin\0",tmpdir);
	fp3 = fopen(file,"a+");
	fwrite(*vp,sizeof(float),nx*nz,fp3);
	fclose(fp3);*/

	/*if(rank == 1){	// saving with loops
		FILE *fp2;
		sprintf(file,"%s/input_shots.bin\0",tmpdir);
		fp2 = fopen(file,"a+");
		for(ishot = 0; ishot < ntraces; ishot++){
			fwrite(shot[ishot].data,sizeof(float),ns,fp2);
		}
		fclose(fp2);
	}*/

	/* release memory */
	free1int(sx);
	free1float(srce);
	free2float(vp);
	free2float(P);
	free2float(PP);
	free3float(data);

	return(CWP_Exit());
}
