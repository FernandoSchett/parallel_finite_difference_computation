#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
extern "C"{
#include "functions.h"
#include <cmath>
}
#include <sys/time.h>


struct timeval startCopyMem,endCopyMem;
float execTimeMem = 0.0;
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
void kernel_lap(int order, int nx, int nz, float * __restrict__ p, float * __restrict__ lap, float * __restrict__ coefsx, float * __restrict__ coefsz,
                sycl::nd_item<3> item_ct1)
{
	int half_order=order/2;
        int i = half_order +
                item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2); // Global row index
        int j = half_order +
                item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1); // Global column index
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

void kernel_time(int nx, int nz, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ v2, float *__restrict__ lap, float dt2,
                 sycl::nd_item<3> item_ct1)
{

        int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2); // Global row index
        int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1); // Global column index
        int mult = i*nz;

  	if(i<nx){
  		if(j<nz){
			 pp[mult+j] = 2.*p[mult+j] - pp[mult+j] + v2[mult+j]*dt2*lap[mult+j];
		}
  	}
}

void kernel_tapper(int nx, int nz, int nxb, int nzb, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ taperx, float *__restrict__ taperz,
                   sycl::nd_item<3> item_ct1)
{

        int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2); // nx index
        int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1); // nzb index
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

void kernel_src(int nz, float * __restrict__ pp, int sx, int sz, float srce)
{
 	pp[sx*nz+sz] += srce;
}

void kernel_sism(int nx, int nz, int nxb, int nt, int is, int it, int gz, float *__restrict__ d_obs, float *__restrict__ ppr,
                 sycl::nd_item<3> item_ct1)
{
 	int size = nx-(2*nxb);
        int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2); // nx index
        if(i<size)
 		ppr[((i+nxb)*nz) + gz] += d_obs[i*nt + (nt-1-it)];

}

void kernel_img(int nx, int nz, int nxb, int nzb, float * __restrict__ imloc, float * __restrict__ p, float * __restrict__ ppr,
                sycl::nd_item<3> item_ct1)
{
 	int size_x = nx-(2*nxb);
 	int size_z = nz-(2*nzb);
        int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2); // Global row index
        int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1); // Global column index
        if(j<size_z){
      if(i<size_x){
        imloc[i*size_z+j] += p[(i+nxb)*nz+(j+nzb)] * ppr[(i+nxb)*nz+(j+nzb)];
      }
    }
}
// ============================ Initialization ===========================
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        float dfrac;
   	nxbin=nxb; nzbin=nzb;
   	brdBufferLength = nxb*sizeof(float);
   	mtxBufferLength = (nxe*nze)*sizeof(float);
   	coefsBufferLength = (order+1)*sizeof(float);
	obsBufferLength = nt*(nxe-(2*nxb))*sizeof(float);
   	imgBufferLength = (nxe-(2*nxb))*(nze-(2*nzb))*sizeof(float);

	taper_x = alloc1float(nxb);
	taper_z = alloc1float(nzb);

        dfrac = sqrt(-log(fac)) / (1. * nxb);
        for(int i=0;i<nxb;i++)
          taper_x[i] = exp(-pow((dfrac * (nxb - i)), 2));

        dfrac = sqrt(-log(fac)) / (1. * nzb);
        for(int i=0;i<nzb;i++)
          taper_z[i] = exp(-pow((dfrac * (nzb - i)), 2));

        // Create a Device pointers
        d_v2 = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_p = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_pp = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_pr = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_ppr = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_swap = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_laplace = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);

        d_sis = (float *)sycl::malloc_device(obsBufferLength, q_ct1);
        d_img = (float *)sycl::malloc_device(imgBufferLength, q_ct1);
        d_coefs_x = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
        d_coefs_z = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
        d_taperx = (float *)sycl::malloc_device(brdBufferLength, q_ct1);
        d_taperz = (float *)sycl::malloc_device(brdBufferLength, q_ct1);

        int div_x, div_z;
	// Set a Grid for the execution on the device
	div_x = (float) nxe/(float) sizeblock;
	div_z = (float) nze/(float) sizeblock;
        gridx = (int)ceil(div_x);
        gridz = (int)ceil(div_z);

        div_x = (float) nxb/(float) sizeblock;
	div_z = (float) nzb/(float) sizeblock;
        gridBorder_x = (int)ceil(div_x);
        gridBorder_z = (int)ceil(div_z);

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
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
	gettimeofday(&startCopyMem, NULL);
        if(flag == 0){
                q_ct1.memcpy(d_p, p[0], mtxBufferLength).wait();
                q_ct1.memcpy(d_pp, pp[0], mtxBufferLength).wait();
                q_ct1.memcpy(d_v2, v2[0], mtxBufferLength).wait();
                q_ct1.memcpy(d_coefs_x, coefs_x, coefsBufferLength).wait();
                q_ct1.memcpy(d_coefs_z, coefs_z, coefsBufferLength).wait();
                q_ct1.memcpy(d_taperx, taperx, brdBufferLength).wait();
                q_ct1.memcpy(d_taperz, taperz, brdBufferLength).wait();
        }

	if(flag == 1){
                q_ct1.memcpy(d_pr, p[0], mtxBufferLength).wait();
                q_ct1.memcpy(d_ppr, pp[0], mtxBufferLength).wait();
                q_ct1.memcpy(d_sis, d_obs[is], obsBufferLength).wait();
                q_ct1.memcpy(d_img, imloc[0], imgBufferLength).wait();
        }
	gettimeofday(&endCopyMem, NULL);
        execTimeMem += ((endCopyMem.tv_sec - startCopyMem.tv_sec)*1000000 + (endCopyMem.tv_usec - startCopyMem.tv_usec))/1000;
}
// ============================ Propagation ============================
void fd_forward(int order, float **p, float **pp, float **v2, int nz, int nx, int nt, int is, int sz, int *sx, float *srce, int propag)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        sycl::range<3> dimGrid(1, gridz, gridx);
        sycl::range<3> dimGridTaper(1, gridBorder_z, gridx);

        sycl::range<3> dimGridSingle(1, 1, 1);
        sycl::range<3> dimGridUpb(1, 1, gridx);

        sycl::range<3> dimBlock(1, sizeblock, sizeblock);

        write_buffers(p,pp,v2,taper_x, taper_z,NULL, NULL,is,0);
	   	
   	for (int it = 0; it < nt; it++){
	 	d_swap  = d_pp;
	 	d_pp = d_p;
	 	d_p = d_swap;

                /*
	 	DPCT1049:0: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto nxbin_ct2 = nxbin;
                        auto nzbin_ct3 = nzbin;
                        auto d_p_ct4 = d_p;
                        auto d_pp_ct5 = d_pp;
                        auto d_taperx_ct6 = d_taperx;
                        auto d_taperz_ct7 = d_taperz;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGridTaper * dimBlock,
                                              dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_tapper(nx, nz, nxbin_ct2, nzbin_ct3,
                                                  d_p_ct4, d_pp_ct5,
                                                  d_taperx_ct6, d_taperz_ct7,
                                                  item_ct1);
                            });
                });
                /*
	 	DPCT1049:1: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto d_p_ct3 = d_p;
                        auto d_laplace_ct4 = d_laplace;
                        auto d_coefs_x_ct5 = d_coefs_x;
                        auto d_coefs_z_ct6 = d_coefs_z;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_lap(order, nx, nz, d_p_ct3,
                                               d_laplace_ct4, d_coefs_x_ct5,
                                               d_coefs_z_ct6, item_ct1);
                            });
                });
                /*
	 	DPCT1049:2: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto d_p_ct2 = d_p;
                        auto d_pp_ct3 = d_pp;
                        auto d_v2_ct4 = d_v2;
                        auto d_laplace_ct5 = d_laplace;
                        auto dt2_ct6 = dt2;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_time(nx, nz, d_p_ct2, d_pp_ct3,
                                                d_v2_ct4, d_laplace_ct5,
                                                dt2_ct6, item_ct1);
                            });
                });
                /*
	 	DPCT1049:3: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto d_pp_ct1 = d_pp;
                        auto sx_is_ct2 = sx[is];
                        auto srce_it_ct4 = srce[it];

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGridSingle * dimBlock,
                                              dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_src(nz, d_pp_ct1, sx_is_ct2, sz,
                                               srce_it_ct4);
                            });
                });

                if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
 	}
        q_ct1.memcpy(p[0], d_p, mtxBufferLength).wait();
        q_ct1.memcpy(pp[0], d_pp, mtxBufferLength).wait();
}

void fd_back(int order, float **p, float **pp, float **pr, float **ppr, float **v2, int nz, int nx, int nt, int is, int sz, int gz, float ***snaps, float **imloc, float **d_obs)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        int ix, iz, it;

        sycl::range<3> dimGrid(1, gridz, gridx);
        sycl::range<3> dimGridTaper(1, gridBorder_z, gridx);
        sycl::range<3> dimGridUpb(1, 1, gridx);

        sycl::range<3> dimBlock(1, sizeblock, sizeblock);
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
                        q_ct1.memcpy(d_pp, pp[0], mtxBufferLength).wait();
                }
		else
		{
                        /*
			DPCT1049:9: The workgroup size
                         * passed to the SYCL kernel may exceed the limit. To
                         * get the device limit, query
                         * info::device::max_work_group_size. Adjust the
                         * workgroup size if needed.
			*/
                        q_ct1.submit([&](sycl::handler &cgh) {
                                auto d_p_ct3 = d_p;
                                auto d_laplace_ct4 = d_laplace;
                                auto d_coefs_x_ct5 = d_coefs_x;
                                auto d_coefs_z_ct6 = d_coefs_z;

                                cgh.parallel_for(
                                    sycl::nd_range<3>(dimGrid * dimBlock,
                                                      dimBlock),
                                    [=](sycl::nd_item<3> item_ct1) {
                                            kernel_lap(order, nx, nz, d_p_ct3,
                                                       d_laplace_ct4,
                                                       d_coefs_x_ct5,
                                                       d_coefs_z_ct6, item_ct1);
                                    });
                        });
                        /*
			DPCT1049:10: The workgroup size
                         * passed to the SYCL kernel may exceed the limit. To
                         * get the device limit, query
                         * info::device::max_work_group_size. Adjust the
                         * workgroup size if needed.
			*/
                        q_ct1.submit([&](sycl::handler &cgh) {
                                auto d_p_ct2 = d_p;
                                auto d_pp_ct3 = d_pp;
                                auto d_v2_ct4 = d_v2;
                                auto d_laplace_ct5 = d_laplace;
                                auto dt2_ct6 = dt2;

                                cgh.parallel_for(
                                    sycl::nd_range<3>(dimGrid * dimBlock,
                                                      dimBlock),
                                    [=](sycl::nd_item<3> item_ct1) {
                                            kernel_time(nx, nz, d_p_ct2,
                                                        d_pp_ct3, d_v2_ct4,
                                                        d_laplace_ct5, dt2_ct6,
                                                        item_ct1);
                                    });
                        });
                }

		d_swap = d_pp;
		d_pp = d_p;
		d_p = d_swap;

                /*
		DPCT1049:4: The workgroup size passed to the SYCL
                 * kernel may exceed the limit. To get the device limit, query
                 * info::device::max_work_group_size. Adjust the workgroup size
                 * if needed.
		*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto nxbin_ct2 = nxbin;
                        auto nzbin_ct3 = nzbin;
                        auto d_pr_ct4 = d_pr;
                        auto d_ppr_ct5 = d_ppr;
                        auto d_taperx_ct6 = d_taperx;
                        auto d_taperz_ct7 = d_taperz;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGridTaper * dimBlock,
                                              dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_tapper(nx, nz, nxbin_ct2, nzbin_ct3,
                                                  d_pr_ct4, d_ppr_ct5,
                                                  d_taperx_ct6, d_taperz_ct7,
                                                  item_ct1);
                            });
                });
                /*
		DPCT1049:5: The workgroup size passed to the SYCL
                 * kernel may exceed the limit. To get the device limit, query
                 * info::device::max_work_group_size. Adjust the workgroup size
                 * if needed.
		*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto d_pr_ct3 = d_pr;
                        auto d_laplace_ct4 = d_laplace;
                        auto d_coefs_x_ct5 = d_coefs_x;
                        auto d_coefs_z_ct6 = d_coefs_z;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_lap(order, nx, nz, d_pr_ct3,
                                               d_laplace_ct4, d_coefs_x_ct5,
                                               d_coefs_z_ct6, item_ct1);
                            });
                });
                /*
		DPCT1049:6: The workgroup size passed to the SYCL
                 * kernel may exceed the limit. To get the device limit, query
                 * info::device::max_work_group_size. Adjust the workgroup size
                 * if needed.
		*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto d_pr_ct2 = d_pr;
                        auto d_ppr_ct3 = d_ppr;
                        auto d_v2_ct4 = d_v2;
                        auto d_laplace_ct5 = d_laplace;
                        auto dt2_ct6 = dt2;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_time(nx, nz, d_pr_ct2, d_ppr_ct3,
                                                d_v2_ct4, d_laplace_ct5,
                                                dt2_ct6, item_ct1);
                            });
                });
                /*
		DPCT1049:7: The workgroup size passed to the SYCL
                 * kernel may exceed the limit. To get the device limit, query
                 * info::device::max_work_group_size. Adjust the workgroup size
                 * if needed.
		*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto nxbin_ct2 = nxbin;
                        auto d_sis_ct7 = d_sis;
                        auto d_ppr_ct8 = d_ppr;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGridUpb * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_sism(nx, nz, nxbin_ct2, nt, is, it,
                                                gz, d_sis_ct7, d_ppr_ct8,
                                                item_ct1);
                            });
                });
                /*
		DPCT1049:8: The workgroup size passed to the SYCL
                 * kernel may exceed the limit. To get the device limit, query
                 * info::device::max_work_group_size. Adjust the workgroup size
                 * if needed.
		*/
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto nxbin_ct2 = nxbin;
                        auto nzbin_ct3 = nzbin;
                        auto d_img_ct4 = d_img;
                        auto d_p_ct5 = d_p;
                        auto d_ppr_ct6 = d_ppr;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel_img(nx, nz, nxbin_ct2, nzbin_ct3,
                                               d_img_ct4, d_p_ct5, d_ppr_ct6,
                                               item_ct1);
                            });
                });

                d_swap = d_ppr;
		d_ppr = d_pr;
		d_pr = d_swap;

		if((it+1)%100 == 0)
		{
			fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);
		}
	}
        q_ct1.memcpy(imloc[0], d_img, imgBufferLength).wait();
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
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
	struct timeval start,end;
        FILE *fsource = NULL, *fvel_ext = NULL, *fd_obs = NULL, *fvp = NULL, *fsns = NULL,*fsns2 = NULL, *fsnr = NULL, *fimg = NULL, *flim = NULL, *fimg_lap = NULL;

	int iz, ix, it, is;

	float *srce;
	float **vp = NULL, **vpe = NULL, **vpex = NULL;

	float **PP,**P,**PPR,**PR,**tmp;
	float ***swf, ***snaps, **vel2, ***d_obs, ***vel_ext_rnd;
	float **imloc, **img, **img_lap;
	gettimeofday(&start, NULL);
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
	gettimeofday(&end, NULL);
	float execTime= ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec))/1000000;
        printf("> Copy memory Time    = %.2f (ms)\n",execTimeMem);
	printf("> Exec time = %.2f (s)\n", execTime);
        
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
        sycl::free(d_p, q_ct1);
        sycl::free(d_pp, q_ct1);
        sycl::free(d_pr, q_ct1);
        sycl::free(d_ppr, q_ct1);
        sycl::free(d_v2, q_ct1);
        sycl::free(d_laplace, q_ct1);
        sycl::free(d_coefs_z, q_ct1);
        sycl::free(d_coefs_x, q_ct1);

        sycl::free(d_taperx, q_ct1);
        sycl::free(d_taperz, q_ct1);

        sycl::free(d_sis, q_ct1);
        sycl::free(d_img, q_ct1);
        return 0;
}
