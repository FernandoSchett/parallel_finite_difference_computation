#include <cuda.h>
#include <stdio.h>
extern "C"{
	#include "functions.h"
}

/* file names */
char *tmpdir = NULL, *vpfile = NULL, *datfile = NULL, *vel_ext_file = NULL;
/* size */
int nz, nx, nt;
float dz, dx, dt;

/* adquisition geometry */
int ns = -1, sz = -1, fsx = -1, ds = -1, gz = -1;

/* boundary */
int nxb = -1, nzb = -1, nxe, nze;
float fac = -1.0;

/* propagation */
int order = -1; 
float fpeak;

/* arrays */
int *sx;

/*aux*/
int iss = -1, rnd, vel_ext_flag=0;

float *d_p, *d_pr, *d_pp, *d_ppr, *d_swap;
float *d_laplace, *d_v2, *d_coefs_x, *d_coefs_z;
float *d_taperx, *d_taperz, *d_sis, *d_img;

size_t mtxBufferLength, brdBufferLength;
size_t imgBufferLength, obsBufferLength;
size_t coefsBufferLength;

float *taper_x, *taper_z;
int nxbin, nzbin;

int gridx, gridz;
int gridBorder_x, gridBorder_z;

static float dx2inv,dz2inv,dt2;
static float **laplace = NULL;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;


// ============================ Kernels ============================
__global__ void kernel_lap(int order, int nx, int nz, float * __restrict__ p, float * __restrict__ lap, float * __restrict__ coefsx, float * __restrict__ coefsz)
{
	int half_order=order/2;
	int i =  half_order + blockIdx.x * blockDim.x + threadIdx.x; // Global row index
	int j =  half_order + blockIdx.y * blockDim.y + threadIdx.y; // Global column index
	int mult = i*nz;
	int aux;
	float acmx = 0, acmz = 0;

	if(i<nx - half_order)
	{
		if(j<nz - half_order)
		{
			for(int io=0;io<=order;io++)
			{
				aux = io-half_order;
				acmz += p[mult + j+aux]*coefsz[io];
				acmx += p[(i+aux)*nz + j]*coefsx[io];
			}
			lap[mult +j] = acmz + acmx;
			acmx = 0.0;
			acmz = 0.0;
		}
	}

}

__global__ void kernel_time(int nx, int nz, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ v2, float *__restrict__ lap, float dt2)
{

  	int i =  blockIdx.x * blockDim.x + threadIdx.x; // Global row index
  	int j =  blockIdx.y * blockDim.y + threadIdx.y; // Global column index
  	int mult = i*nz;

  	if(i<nx){
  		if(j<nz){
			 pp[mult+j] = 2.*p[mult+j] - pp[mult+j] + v2[mult+j]*dt2*lap[mult+j];
		}
  	}
}

__global__ void kernel_tapper(int nx, int nz, int nxb, int nzb, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ taperx, float *__restrict__ taperz)
{

	int i =  blockIdx.x * blockDim.x + threadIdx.x; // nx index
	int j =  blockIdx.y * blockDim.y + threadIdx.y; // nzb index
	int itxr = nx - 1, mult = i*nz;

	if(i<nx){
		if(j<nzb){
			p[mult+j] *= taperz[j];
			pp[mult+j] *= taperz[j];
		}
	}

	if(i<nxb){
		if(j<nzb){
			p[mult+j] *= taperx[i];
			pp[mult+j] *= taperx[i];

			p[(itxr-i)*nz+j] *= taperx[i];
			pp[(itxr-i)*nz+j] *= taperx[i];
		}
	}
}

__global__ void kernel_src(int nz, float * __restrict__ pp, int sx, int sz, float srce)
{
 	pp[sx*nz+sz] += srce;
}

__global__ void kernel_sism(int nx, int nz, int nxb, int nt, int is, int it, int gz, float *__restrict__ d_obs, float *__restrict__ ppr)
{
 	int size = nx-(2*nxb);
	int i = blockIdx.x * blockDim.x + threadIdx.x; //nx index
 	if(i<size)
 		ppr[((i+nxb)*nz) + gz] += d_obs[i*nt + (nt-1-it)];

}

__global__ void kernel_img(int nx, int nz, int nxb, int nzb, float * __restrict__ imloc, float * __restrict__ p, float * __restrict__ ppr)
{
 	int size_x = nx-(2*nxb);
 	int size_z = nz-(2*nzb);
	int i =  blockIdx.x * blockDim.x + threadIdx.x; // Global row index
  	int j =  blockIdx.y * blockDim.y + threadIdx.y; // Global column index
 	if(j<size_z){
      if(i<size_x){
        imloc[i*size_z+j] += p[(i+nxb)*nz+(j+nzb)] * ppr[(i+nxb)*nz+(j+nzb)];
      }
    }
}
// ============================ Initialization ===========================
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac)
{
	float dfrac;
   	nxbin=nxb; nzbin=nzb;
   	brdBufferLength = nxb*sizeof(float);
   	mtxBufferLength = (nxe*nze)*sizeof(float);
   	coefsBufferLength = (order+1)*sizeof(float);
	obsBufferLength = nt*(nxe-(2*nxb))*sizeof(float);
   	imgBufferLength = (nxe-(2*nxb))*(nze-(2*nzb))*sizeof(float);

	taper_x = alloc1float(nxb);
	taper_z = alloc1float(nzb);

	dfrac = sqrt(-log(fac))/(1.*nxb);
	for(int i=0;i<nxb;i++)
	  taper_x[i] = exp(-pow((dfrac*(nxb-i)),2));


	dfrac = sqrt(-log(fac))/(1.*nzb);
	for(int i=0;i<nzb;i++)
	  taper_z[i] = exp(-pow((dfrac*(nzb-i)),2));


	// Create a Device pointers
	cudaMalloc((void **) &d_v2, mtxBufferLength);
	cudaMalloc((void **) &d_p, mtxBufferLength);
	cudaMalloc((void **) &d_pp, mtxBufferLength);
	cudaMalloc((void **) &d_pr, mtxBufferLength);
	cudaMalloc((void **) &d_ppr, mtxBufferLength);
	cudaMalloc((void **) &d_swap, mtxBufferLength);
	cudaMalloc((void **) &d_laplace, mtxBufferLength);

	cudaMalloc((void **) &d_sis, obsBufferLength);
	cudaMalloc((void **) &d_img, imgBufferLength);
	cudaMalloc((void **) &d_coefs_x, coefsBufferLength);
	cudaMalloc((void **) &d_coefs_z, coefsBufferLength);
	cudaMalloc((void **) &d_taperx, brdBufferLength);
	cudaMalloc((void **) &d_taperz, brdBufferLength);

	int div_x, div_z;
	// Set a Grid for the execution on the device
	div_x = (float) nxe/(float) sizeblock;
	div_z = (float) nze/(float) sizeblock;
	gridx = (int) ceil(div_x);
	gridz = (int) ceil(div_z);

	div_x = (float) nxb/(float) sizeblock;
	div_z = (float) nzb/(float) sizeblock;
	gridBorder_x = (int) ceil(div_x);
	gridBorder_z = (int) ceil(div_z);

	div_x = (float) 8/(float) sizeblock;
}

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt)
{
	int io;
	dx2inv = (1./dx)*(1./dx);
	dz2inv = (1./dz)*(1./dz);
	dt2 = dt*dt;

	coefs = calc_coefs(order);
	laplace = alloc2float(nz,nx);

	coefs_z = calc_coefs(order);
	coefs_x = calc_coefs(order);

	// pre calc coefs 8 d2 inv
	for (io = 0; io <= order; io++) {
		coefs_z[io] = dz2inv * coefs[io];
		coefs_x[io] = dx2inv * coefs[io];
	}

	memset(*laplace,0,nz*nx*sizeof(float));

        fd_init_cuda(order,nx,nz,nxb,nzb,nt,ns,fac);

        return;
}

void write_buffers(float **p, float **pp, float **v2, float *taperx, float *taperz, float **d_obs, float **imloc, int is, int flag)
{
    
	if(flag == 0){
		cudaMemcpy(d_p, p[0], mtxBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pp, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, v2[0], mtxBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_coefs_x, coefs_x, coefsBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_coefs_z, coefs_z, coefsBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_taperx, taperx, brdBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_taperz, taperz, brdBufferLength, cudaMemcpyHostToDevice);
	}

	if(flag == 1){
		cudaMemcpy(d_pr, p[0], mtxBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_ppr, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sis, d_obs[is], obsBufferLength, cudaMemcpyHostToDevice);
		cudaMemcpy(d_img, imloc[0], imgBufferLength, cudaMemcpyHostToDevice);
	}
}
// ============================ Propagation ============================
void fd_forward(int order, float **p, float **pp, float **v2, int nz, int nx, int nt, int is, int sz, int *sx, float *srce, int propag)
{
 	dim3 dimGrid(gridx, gridz);
  	dim3 dimGridTaper(gridx, gridBorder_z);

  	dim3 dimGridSingle(1,1);
  	dim3 dimGridUpb(gridx,1);

  	dim3 dimBlock(sizeblock, sizeblock);
  	
	write_buffers(p,pp,v2,taper_x, taper_z,NULL, NULL,is,0);
	   	
   	for (int it = 0; it < nt; it++){
	 	d_swap  = d_pp;
	 	d_pp = d_p;
	 	d_p = d_swap;

	 	kernel_tapper<<<dimGridTaper, dimBlock>>>(nx,nz,nxbin,nzbin,d_p,d_pp,d_taperx,d_taperz);
	 	kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_p,d_laplace,d_coefs_x,d_coefs_z);
	 	kernel_time<<<dimGrid, dimBlock>>>(nx,nz,d_p,d_pp,d_v2,d_laplace,dt2);
	 	kernel_src<<<dimGridSingle, dimBlock>>>(nz,d_pp,sx[is],sz,srce[it]);
		if (it == 750)
		{
 			cudaMemcpy(p[0], d_p, mtxBufferLength, cudaMemcpyDeviceToHost);
			int x, y;
			FILE *teste;
			teste = fopen("file-teste","w");
			for (x = 0; x < nx; x++)
			{
				for (y = 0; y < nz; y++)
				{
					fprintf(teste, "%f\n", p[x][y]);
				}
			}
		}

		if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
 	}
 	cudaMemcpy(p[0], d_p, mtxBufferLength, cudaMemcpyDeviceToHost);
	cudaMemcpy(pp[0], d_pp, mtxBufferLength, cudaMemcpyDeviceToHost);
	
}

void fd_back(int order, float **p, float **pp, float **pr, float **ppr, float **v2, int nz, int nx, int nt, int is, int sz, int gz, float ***snaps, float **imloc, float **d_obs)
{
	int ix, iz, it;

	dim3 dimGrid(gridx, gridz);
  	dim3 dimGridTaper(gridx, gridBorder_z);
  	dim3 dimGridUpb(gridx,1);

  	dim3 dimBlock(sizeblock, sizeblock);
	write_buffers(p,pp,v2,taper_x, taper_z,d_obs,imloc,is,0);
	write_buffers(pr,ppr,v2,taper_x,taper_z,d_obs,imloc,is,1);
	
	for(it=0; it<nt; it++)
	{
		if(it==0 || it==1)
		{
			for(ix=0; ix<nx; ix++)
			{
				for(iz=0; iz<nz; iz++)
				{
					pp[ix][iz] = snaps[1-it][ix][iz];
				}
			}
			cudaMemcpy(d_pp, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
		}
		else
		{
			kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_p,d_laplace,d_coefs_x,d_coefs_z);
			kernel_time<<<dimGrid, dimBlock>>>(nx,nz,d_p,d_pp,d_v2,d_laplace,dt2);
		}

		d_swap = d_pp;
		d_pp = d_p;
		d_p = d_swap;

		kernel_tapper<<<dimGridTaper, dimBlock>>>(nx,nz,nxbin,nzbin,d_pr,d_ppr,d_taperx,d_taperz);
		kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_pr,d_laplace, d_coefs_x, d_coefs_z);
		kernel_time<<<dimGrid, dimBlock>>>(nx,nz,d_pr,d_ppr,d_v2,d_laplace,dt2);
		kernel_sism<<<dimGridUpb, dimBlock>>>(nx,nz,nxbin,nt,is,it,gz,d_sis,d_ppr);
		kernel_img<<<dimGrid, dimBlock>>>(nx,nz,nxbin,nzbin,d_img,d_p,d_ppr);

		d_swap = d_ppr;
		d_ppr = d_pr;
		d_pr = d_swap;

		if((it+1)%100 == 0)
		{
			fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);
		}
	}
	cudaMemcpy(imloc[0], d_img, imgBufferLength, cudaMemcpyDeviceToHost);
}

void init_args()
{
	tmpdir = get_str_input("tmpdir");
	vpfile = get_str_input("vpfile");
	datfile = get_str_input("datfile");
	vel_ext_file = get_str_input("vel_ext_file");
	nz = get_int_input("nz");
	nx = get_int_input("nx");
	nt = get_int_input("nt");
	ns = get_int_input("ns");
	sz = get_int_input("sz");
	fsx = get_int_input("fsx");
	ds = get_int_input("ds");
	gz = get_int_input("gz");
	order = get_int_input("order");
	nzb = get_int_input("nzb");
	nxb = get_int_input("nxb");
	iss = get_int_input("iss");
	rnd = get_int_input("rnd");
	dz = get_float_input("dz");
	dx = get_float_input("dx");
	dt = get_float_input("dt");
	fpeak = get_float_input("fpeak");
	fac = get_float_input("fac");
	if(vel_ext_file != NULL) vel_ext_flag = 1;
	if(iss == -1 ) iss = 0;	 	// save snaps of this source
	if(ns == -1) ns = 1;	 	// number of sources
	if(sz == -1) sz = 0; 		// source depth
	if(fsx == -1) fsx = 0; 	// first source position
	if(ds == -1) ds = 1; 		// source interval
	if(gz == -1) gz = 0; 		// receivor depth
	if(order == -1) order = 8;	// FD order
	if(nzb == -1) nzb = 40;		// z border size
	if(nxb == -1) nxb = 40;		// x border size
	if(fac == -1.0) fac = 0.7;
}

int main (int argc, char **argv)
{
	FILE *fsource = NULL, *fvel_ext = NULL, *fd_obs = NULL, *fvp = NULL, *fsns = NULL,*fsns2 = NULL, *fsnr = NULL, *fimg = NULL, *flim = NULL, *fimg_lap = NULL;

	int iz, ix, it, is;

	float *srce;
	float **vp = NULL, **vpe = NULL, **vpex = NULL;

	float **PP,**P,**PPR,**PR,**tmp;
	float ***swf, ***snaps, **vel2, ***d_obs, ***vel_ext_rnd;
	float **imloc, **img, **img_lap;
	read_input(argv[1]);
	init_args();
	
	printf("## vp = %s, d_obs = %s, vel_ext_file = %s, vel_ext_flag = %d \n",vpfile,datfile,vel_ext_file,vel_ext_flag);
	printf("## nz = %d, nx = %d, nt = %d \n",nz,nx,nt);
	printf("## dz = %f, dx = %f, dt = %f \n",dz,dx,dt);
	printf("## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",ns,sz,fsx,ds,gz);
	printf("## order = %d, nzb = %d, nxb = %d, F = %f, rnd = %d \n",order,nzb,nxb,fac,rnd);
	srce = alloc1float(nt);
	ricker_wavelet(nt, dt, fpeak, srce);
	sx = alloc1int(ns);
	for(is=0; is<ns; is++){
		sx[is] = fsx + (is*ds) + nxb;
	}
	sz += nzb;
	gz += nzb;
	nze = nz + 2 * nzb;
	nxe = nx + 2 * nxb;
	if(vel_ext_flag){
		vel_ext_rnd = alloc3float(nze,nxe,ns);
		memset(**vel_ext_rnd,0,nze*nxe*ns*sizeof(float));
		fvel_ext = fopen(vel_ext_file,"r");
		fread(**vel_ext_rnd,sizeof(float),nze*nxe*ns,fvel_ext);
		fclose(fvel_ext);
	}

	d_obs = alloc3float(nt,nx,ns);
	memset(**d_obs,0,nt*nx*ns*sizeof(float));
	fd_obs = fopen(datfile,"r");
	fread(**d_obs,sizeof(float),nt*nx*ns,fd_obs);
	fclose(fd_obs);
	
	float **d_obs_aux=(float**)malloc(ns*sizeof(float*));
	for(int i=0; i<ns; i++) 
		d_obs_aux[i] = (float*)malloc((nt*nx)*sizeof(float)); 
	
	for(int i=0; i<ns; i++){
		for(int j=0; j<nx; j++){
			for(int k=0; k<nt; k++)
				d_obs_aux[i][j*nt+k] = d_obs[i][j][k]; 
		}
	}

	vp = alloc2float(nz,nx);
	memset(*vp,0,nz*nx*sizeof(float));
	fvp = fopen(vpfile,"r");
	fread(vp[0],sizeof(float),nz*nx,fvp);
	fclose(fvp);
	vpe = alloc2float(nze,nxe);
	vpex = vpe;

	for(ix=0; ix<nx; ix++){
		for(iz=0; iz<nz; iz++){
			vpe[ix+nxb][iz+nzb] = vp[ix][iz]; 
		}
	}

	vel2 = alloc2float(nze,nxe);
	fd_init(order,nxe,nze,nxb,nzb,nt,ns,fac,dx,dz,dt);
	taper_init(nxb,nzb,fac);

	PP = alloc2float(nze,nxe);
	P = alloc2float(nze,nxe);
	PPR = alloc2float(nze,nxe);
	PR = alloc2float(nze,nxe);
	snaps = alloc3float(nze,nxe,2);
	imloc = alloc2float(nz,nx);
	img = alloc2float(nz,nx);
	img_lap = alloc2float(nz,nx);

	char filepath [100];
	sprintf(filepath, "%s/dir.snaps", tmpdir);
	fsns = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.snaps_rec", tmpdir);
	fsns2 = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.snapr", tmpdir);
	fsnr = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.image", tmpdir);
	fimg = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.image_lap", tmpdir);	
	fimg_lap = fopen(filepath,"w");
	
	memset(*img,0,nz*nx*sizeof(float));
	memset(*img_lap,0,nz*nx*sizeof(float));
	FILE *foutput;
	foutput = fopen("image.num", "w");
	for(is=0; is<ns; is++){
		fprintf(stdout,"** source %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		if (vel_ext_flag){
			vpe = vel_ext_rnd[is];					// load hybrid border vpe from file
		}else{
			extendvel_linear(nx,nz,nxb,nzb,vpe); 	// hybrid border (linear randomic)
		}


		for(ix=0; ix<nx+2*nxb; ix++){
			for(iz=0; iz<nz+2*nzb; iz++){
				vel2[ix][iz] = vpe[ix][iz]*vpe[ix][iz];
			}
		}

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
		
		fd_forward(order,P,PP,vel2,nze,nxe,nt,is,sz,sx,srce, is);
		fprintf(stdout,"\n");

		for(iz=0; iz<nze; iz++){
			for(ix=0; ix<nxe; ix++){
				snaps[0][ix][iz] = P[ix][iz];
				snaps[1][ix][iz] = PP[ix][iz];
			}
		}
		
		fprintf(stdout,"** backward propagation %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
		memset(*PPR,0,nze*nxe*sizeof(float));
		memset(*PR,0,nze*nxe*sizeof(float));
		memset(*imloc,0,nz*nx*sizeof(float));


		fd_back(order,P,PP,PR,PPR,vel2,nze,nxe,nt,is,sz,gz,snaps,imloc,d_obs_aux);
		fprintf(stdout,"\n");
		

		fprintf(foutput,"======== %i ========\n", is);
		for(iz=0; iz<nz; iz++){
			for(ix=0; ix<nx; ix++){
				img[ix][iz] += imloc[ix][iz];
				fprintf(foutput, " %f \n", img[ix][iz]);
			}
		}
	}
	// for(iz=0; iz<nz; iz++){
	// 	for(ix=0; ix<nx; ix++){
	// 		fprintf(foutput, " %f \n", img[ix][iz]);
	// 	}
	// }
	fwrite(*img,sizeof(float),nz*nx,fimg);

	fwrite(*img_lap,sizeof(float),nz*nx,fimg_lap);

	fclose(fsns);
	fclose(fsns2);
	fclose(fsnr);
	fclose(fimg);
	fclose(fimg_lap);
        
        //free memory device
	taper_destroy();
	free1float(coefs);
	free1int(sx);
	free1float(srce);
	free2float(laplace);
	free2float(vp);
	free2float(P);
	free2float(PP);
	free2float(PR);
	free2float(PPR);
	free3float(snaps);
	free2float(imloc);
	free2float(img);
	free2float(img_lap);
	free2float(vpex);
	free2float(vel2);
	free3float(d_obs);
	if(vel_ext_flag) free3float(vel_ext_rnd);
	cudaFree(d_p);
	cudaFree(d_pp);
	cudaFree(d_pr);
	cudaFree(d_ppr);
	cudaFree(d_v2);
	cudaFree(d_laplace);
	cudaFree(d_coefs_z);
	cudaFree(d_coefs_x);

	cudaFree(d_taperx);
	cudaFree(d_taperz);

	cudaFree(d_sis);
	cudaFree(d_img);
        return 0;
}
