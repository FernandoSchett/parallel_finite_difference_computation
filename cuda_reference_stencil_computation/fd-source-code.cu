#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <string.h>

void fd_init(int order, int nx, int nz, float dx, float dz);
void fd_init_cuda(int order, int nxe, int nze);
float *calc_coefs(int order);
static void makeo2 (float *coef,int order);
void read_input(char *file);
void free_memory();

#define sizeblock 32
#define PI (3.141592653589793)

char *file_path;
float *input_data, *output_data;

float *d_p;
float *d_laplace, *d_coefs_x, *d_coefs_z;

size_t mtxBufferLength, coefsBufferLength;

int gridx, gridz;
int nz, nx, nxb, nzb, nxe, nze, order;
float dz, dx;

static float dx2inv, dz2inv;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;

void read_input(char *file)
{
        FILE *fp;
        fp = fopen(file, "r");
        char *line = NULL;
        size_t len = 0;
        if (fp == NULL)
                exit(EXIT_FAILURE);
        while (getline(&line, &len, fp) != -1) {
                if(strstr(line,"tmpdir") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        file_path = strdup(tok);
                }
                if(strstr(line,"nzb") != NULL)
                {
                        char *nzb_char;
                        nzb_char = strtok(line, "=");
                        nzb_char = strtok(NULL,"=");
                        nzb = atoi(nzb_char);
                }
                if(strstr(line,"nxb") != NULL)
                {
                        char *nxb_char;
                        nxb_char = strtok(line, "=");
                        nxb_char = strtok(NULL,"=");
                        nxb = atoi(nxb_char);
                }
                if(strstr(line,"nz") != NULL)
                {
                        char *nz_char;
                        nz_char = strtok(line, "=");
                        if (strlen(nz_char) <= 2)
                        {
                                nz_char = strtok(NULL,"=");
                                nz = atoi(nz_char);
                        }
                }
                if(strstr(line,"nx") != NULL)
                {
                        char *nx_char;
                        nx_char = strtok(line, "=");
                        if (strlen(nx_char) <= 2)
                        {
                                nx_char = strtok(NULL,"=");
                                nx = atoi(nx_char);
                        }
                }
                if(strstr(line,"dz") != NULL)
                {
                        char *dz_char;
                        dz_char = strtok(line, "=");
                        dz_char = strtok(NULL,"=");
                        dz = atof(dz_char);
                }
                if(strstr(line,"dx") != NULL)
                {
                        char *dx_char;
                        dx_char = strtok(line, "=");
                        dx_char = strtok(NULL,"=");
                        dx = atof(dx_char);
                }
                if(strstr(line,"order") != NULL)
                {
                        char *order_char;
                        order_char = strtok(line, "=");
                        order_char = strtok(NULL,"=");
                        order = atoi(order_char);
                }
        }
        free(line);
}

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

void fd_init_cuda(int order, int nxe, int nze)
{
        mtxBufferLength = (nxe*nze)*sizeof(float);
        coefsBufferLength = (order+1)*sizeof(float);

        // Create a Device pointers
        cudaMalloc(&d_p, mtxBufferLength);
        cudaMalloc(&d_laplace, mtxBufferLength);
        cudaMalloc(&d_coefs_x, coefsBufferLength);
        cudaMalloc(&d_coefs_z, coefsBufferLength);

        int div_x, div_z;
        // Set a Grid for the execution on the device
        int tx = ((nxe - 1) / 32 + 1) * 32;
        int tz = ((nze - 1) / 32 + 1) * 32;

        div_x = (float) tx/(float) sizeblock;
        div_z = (float) tz/(float) sizeblock;

        gridx = (int) ceil(div_x);
        gridz = (int) ceil(div_z);
}

void fd_init(int order, int nx, int nz, float dx, float dz)
{
        int io;
        dx2inv = (1./dx)*(1./dx);
        dz2inv = (1./dz)*(1./dz);

        coefs = calc_coefs(order);

        coefs_z = calc_coefs(order);
        coefs_x = calc_coefs(order);

        // pre calc coefs 8 d2 inv
        for (io = 0; io <= order; io++)
        {
                coefs_z[io] = dz2inv * coefs[io];
                coefs_x[io] = dx2inv * coefs[io];
        }

        fd_init_cuda(order,nx,nz);

        return;
}

void free_memory()
{
        free(coefs_z);
        free(coefs_x);
        free(file_path);
        free(input_data);
        free(output_data);
        cudaFree(d_p);
        cudaFree(d_laplace);
        cudaFree(d_coefs_x);
        cudaFree(d_coefs_z);
}

int main (int argc, char **argv)
{
        read_input(argv[1]);

        printf("Local do arquivo: %s\n", file_path);
        printf("nzb = %i\n", nzb);
        printf("nzb = %i\n", nxb);
        printf("nz = %i\n", nz);
        printf("nx = %i\n", nx);
        printf("dz = %f\n", dz);
        printf("dx = %f\n", dx);
        printf("order = %i\n", order);

        nxe = nx + 2 * nxb;
        nze = nz + 2 * nzb;
        // initialization
        fd_init(order,nxe,nze,dx,dz);

        dim3 dimGrid(gridx, gridz);
        dim3 dimBlock(sizeblock, sizeblock);
        FILE *finput;

        if((finput = fopen(file_path, "rb")) == NULL)
                printf("Unable to open file!\n");
        else
                printf("Input successfully opened for reading.\n");

        input_data = (float*)malloc(mtxBufferLength);
        if(!input_data)
                printf("Input memory allocation error!\n");
        else
                printf("Input memory allocation was successful.\n");

        memset(input_data, 0, mtxBufferLength);

        if( fread(input_data, sizeof(float), nze*nxe, finput) != nze*nxe)
                printf("Input reading error!\n");

        else
                printf("Input reading was successful.\n");
        fclose(finput);

        // data copy
        cudaMemcpy(d_p, input_data, mtxBufferLength, cudaMemcpyHostToDevice);
        cudaMemcpy(d_coefs_x, coefs_x, coefsBufferLength, cudaMemcpyHostToDevice);
        cudaMemcpy(d_coefs_z, coefs_z, coefsBufferLength, cudaMemcpyHostToDevice);

        // kernel utilization
        kernel_lap<<<dimGrid, dimBlock>>>(order,nxe,nze,d_p,d_laplace,d_coefs_x,d_coefs_z);

        output_data = (float*)malloc(mtxBufferLength);
        if(!output_data)
                printf("Output memory allocation error!\n");
        else
                printf("Output memory allocation was successful.\n");
        memset(output_data, 0, mtxBufferLength);
        cudaMemcpy(output_data, d_laplace, mtxBufferLength, cudaMemcpyDeviceToHost);

        // Writing output
        FILE *foutput;
        if((foutput = fopen("../bin/output_cuda.bin", "wb")) == NULL)
                printf("Unable to open file!\n");
        else
                printf("Output successfully opened for writing.\n");

        if( fwrite(output_data, sizeof(float), nze*nxe, foutput) != nze*nxe)
                printf("Output writing error!\n");

        else
                printf("Output writing was successful.\n");
        fclose(foutput);

        // free memory device
        free_memory();
        return 0;
}
