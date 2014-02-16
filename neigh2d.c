/* Copyright 2012.  Los Alamos National Security, LLC. This material was produced
 * under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National 
 * Laboratory (LANL), which is operated by Los Alamos National Security, LLC
 * for the U.S. Department of Energy. The U.S. Government has rights to use,
 * reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS
 * ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 * to produce derivative works, such modified software should be clearly marked,
 * so as not to confuse it with the version available from LANL.   
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy
 * of the License at 
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.”
 *
 * Under this license, it is required to include a reference to this work. We
 * request that each derivative work contain a reference to LANL Copyright 
 * Disclosure C13002/LA-CC-12-022 so that this work’s impact can be roughly
 * measured. In addition, it is requested that a modifier is included as in
 * the following example:
 *
 * //<Uses | improves on | modified from> LANL Copyright Disclosure C13002/LA-CC-12-022
 *
 * This is LANL Copyright Disclosure C13002/LA-CC-12-022
 */

/*
 *  Authors: Bob Robey       XCP-2   brobey@lanl.gov
 *           David Nicholaeff        dnic@lanl.gov, mtrxknight@aol.com
 *           Rachel Robey            rnrobey@gmail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include "kdtree/KDTree2d.h"
#include "gpu.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __APPLE_CC__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#ifdef HAVE_CL_DOUBLE
typedef double real;
typedef cl_double cl_real;
typedef cl_double4 cl_real4;
#define SQRT(x) sqrt(x)
#define ONE 1.0
#define TWO 2.0
#else
typedef float real;
typedef cl_float cl_real;
typedef cl_float4 cl_real4;
#define SQRT(x) sqrtf(x)
#define ONE 1.0f
#define TWO 2.0f
#endif

#define SQR(x) (( (x)*(x) ))

typedef unsigned int uint;

#define CHECK 1
#define TILE_SIZE 256
#define DETAILED_TIMING 0
#define LONG_RUNS 1

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))
#define SWAP_PTR(p1,p2,p3) ((p3=p1), (p1=p2), (p2=p3))
#endif

int powerOfFour(int n) {
   int result = 1;
   int i;
   for(i = 0; i < n; i++) {
      result *= 4;
   }
   return result;
}
void swap_double(double** a, double** b) {
   double* c = *a;
   *a = *b;
   *b = c;
}

void swap_real(real** a, real** b) {
   real* c = *a;
   *a = *b;
   *b = c;
}

void swap_int(int** a, int** b) {
   int* c = *a;
   *a = *b;
   *b = c;
}

struct neighbor2d {
    uint left;
    uint right;
    uint bottom;
    uint top;
};

struct timeval timer;
double t1, t2;

int is_nvidia = 0;
#define BRUTE_FORCE_SIZE_LIMIT 500000

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel init_kernel, hash_setup_kernel, calc_neighbor2d_kernel;

void neighbors2d( uint mesh_size, int levmx );
struct neighbor2d *neighbors2d_bruteforce( uint ncells, int *i, int *j, int *level );
struct neighbor2d *neighbors2d_kdtree( uint ncells, int mesh_size, double *x, double *y, int *level );
struct neighbor2d *neighbors2d_hashcpu( uint ncells, int mesh_size, int levmx, int *i, int *j, int *level );
cl_mem neighbors2d_hashgpu( uint ncells, int mesh_size, int levmx, cl_mem i, cl_mem j, cl_mem level, cl_mem levtable, double *time );
int adaptiveMeshConstructorWij(const int n, const int l, int** level_ptr, double** x_ptr, double** y_ptr, int **i_ptr, int **j_ptr);
void genmatrixfree(void **var);
void **genmatrix(int jnum, int inum, size_t elsize);

int main (int argc, const char * argv[]) {

    cl_int error;

    GPUInit(&context, &queue, &is_nvidia, &program, "neigh2d_kern.cl");

    init_kernel = clCreateKernel(program, "init_kern", &error);
    hash_setup_kernel = clCreateKernel(program, "hash_setup_kern", &error);
    calc_neighbor2d_kernel = clCreateKernel(program, "calc_neighbor2d_kern", &error);

    printf("\n    2D Neighbors Performance Results\n\n");
    printf("Size,   \tncells    \tBrute     \tkDtree   \tHash CPU, \tHash GPU\n");

    for (uint levmx = 0; levmx < 6; levmx++ ){
       printf("\nMax levels is %d\n",levmx);
       for( uint i = 16; i <= 1024; i*=2 ) {
          if (levmx > 3 && i > 512) continue;
          printf("%d,     ", i);
          neighbors2d(i, levmx);
          printf("\n");
       }
    }
}

void neighbors2d( uint mesh_size, int levmx ) 
{

   struct neighbor2d *neigh2d_gold, *neigh2d_test;

   //printf("Mesh Size is %d with a maximum level of refinement of %d.\n", mesh_size, levmx);

   int* level = NULL;
   double* x    = NULL;
   double* y    = NULL;
   int* i     = NULL;
   int* j     = NULL;

   int ncells = adaptiveMeshConstructorWij(mesh_size, levmx, &level, &x, &y, &i, &j);
   printf("\t%8d,", ncells);

#ifdef XXX
   for(int ic = 0; ic < ncells; ic++) {
      //if( (ic % ((ncells+10)/10)) == 0 || ic == ncells - 1)
         printf("Cell %d: x = %f \t y = %f \t i = %3d \t j = %3d \t (level = %3d) \n",
            ic, x[ic], y[ic], i[ic], j[ic], level[ic]);      
   }
#endif

   if (ncells < BRUTE_FORCE_SIZE_LIMIT) {
      gettimeofday(&timer, NULL);
      t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
      neigh2d_gold = neighbors2d_bruteforce(ncells, i, j, level);
      gettimeofday(&timer, NULL);
      t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
      printf("\t%.6lf,", t2 - t1);
   } else {
      printf("\tnot_run,  ");
   }

#ifdef XXX
   for(int ic = 0; ic < ncells; ic++) {
      //if( (ic % ((ncells+10)/10)) == 0 || ic == ncells - 1)
         printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
            ic, neigh2d_gold[ic].left, neigh2d_gold[ic].right, neigh2d_gold[ic].bottom, neigh2d_gold[ic].top, level[ic]);      
   }
#endif

   gettimeofday(&timer, NULL);
   t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
   if (ncells < BRUTE_FORCE_SIZE_LIMIT) {
      neigh2d_test = neighbors2d_kdtree(ncells, mesh_size, x, y, level);
   } else {
      neigh2d_gold = neighbors2d_kdtree(ncells, mesh_size, x, y, level);
   }
   gettimeofday(&timer, NULL);
   t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
   printf("\t%.6lf,", t2 - t1);

   if (ncells < BRUTE_FORCE_SIZE_LIMIT) {
      //printf("\n\nkdtree comparison\n");
      for(int ic = 0; ic < ncells; ic++) {
         if( neigh2d_gold[ic].left   != neigh2d_test[ic].left   ||
             neigh2d_gold[ic].right  != neigh2d_test[ic].right  ||
             neigh2d_gold[ic].bottom != neigh2d_test[ic].bottom ||
             neigh2d_gold[ic].top    != neigh2d_test[ic].top    ) {
                printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
                   ic, neigh2d_gold[ic].left, neigh2d_gold[ic].right, neigh2d_gold[ic].bottom, neigh2d_gold[ic].top, level[ic]);      
                printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
                   ic, neigh2d_test[ic].left, neigh2d_test[ic].right, neigh2d_test[ic].bottom, neigh2d_test[ic].top, level[ic]);      
                printf("\n");
          }
      }
      free(neigh2d_test);
   }


   gettimeofday(&timer, NULL);
   t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
   neigh2d_test = neighbors2d_hashcpu(ncells, mesh_size, levmx, i, j, level);
   gettimeofday(&timer, NULL);
   t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
   printf("\t%.6lf,", t2 - t1);

   //printf("\n\nhash cpu test\n");
   for(int ic = 0; ic < ncells; ic++) {
      if( neigh2d_gold[ic].left   != neigh2d_test[ic].left   ||
          neigh2d_gold[ic].right  != neigh2d_test[ic].right  ||
          neigh2d_gold[ic].bottom != neigh2d_test[ic].bottom ||
          neigh2d_gold[ic].top    != neigh2d_test[ic].top    ) {
             printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
                ic, neigh2d_gold[ic].left, neigh2d_gold[ic].right, neigh2d_gold[ic].bottom, neigh2d_gold[ic].top, level[ic]);      
             printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
                ic, neigh2d_test[ic].left, neigh2d_test[ic].right, neigh2d_test[ic].bottom, neigh2d_test[ic].top, level[ic]);      
             printf("\n");
       }
   }
   free(neigh2d_test);

   cl_int error = 0;
   cl_mem i_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells*sizeof(int), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   cl_mem j_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells*sizeof(int), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   cl_mem level_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells*sizeof(int), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clEnqueueWriteBuffer(queue, i_buffer, CL_TRUE, 0, ncells*sizeof(int), i, 0, NULL, NULL);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clEnqueueWriteBuffer(queue, j_buffer, CL_TRUE, 0, ncells*sizeof(int), j, 0, NULL, NULL);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clEnqueueWriteBuffer(queue, level_buffer, CL_TRUE, 0, ncells*sizeof(int), level, 0, NULL, NULL);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   int *levtable = (int *)malloc(levmx+1);
   for (int lev=0; lev<levmx+1; lev++){
      levtable[lev] = (int)pow(2,lev);
   }
   cl_mem levtable_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (levmx+1)*sizeof(int), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clEnqueueWriteBuffer(queue, levtable_buffer, CL_TRUE, 0, (levmx+1)*sizeof(int), levtable, 0, NULL, NULL);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   cl_mem neigh2d_buffer = neighbors2d_hashgpu(ncells, mesh_size, levmx, i_buffer, j_buffer, level_buffer, levtable_buffer, &t2);
   clReleaseMemObject(i_buffer);
   clReleaseMemObject(j_buffer);
   clReleaseMemObject(level_buffer);
   clReleaseMemObject(levtable_buffer);

   if (neigh2d_buffer != NULL) {
      printf("\t%.6lf,", t2);

      neigh2d_test = (struct neighbor2d *)malloc(ncells*sizeof(struct neighbor2d));
      error = clEnqueueReadBuffer(queue, neigh2d_buffer, CL_TRUE, 0, ncells*sizeof(cl_uint4), neigh2d_test, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      clReleaseMemObject(neigh2d_buffer);

      //printf("\n\nhash gpu test\n");
      for(int ic = 0; ic < ncells; ic++) {
         if( neigh2d_gold[ic].left   != neigh2d_test[ic].left   ||
             neigh2d_gold[ic].right  != neigh2d_test[ic].right  ||
             neigh2d_gold[ic].bottom != neigh2d_test[ic].bottom ||
             neigh2d_gold[ic].top    != neigh2d_test[ic].top    ) {
                //printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
                //   ic, neigh2d_gold[ic].left, neigh2d_gold[ic].right, neigh2d_gold[ic].bottom, neigh2d_gold[ic].top, level[ic]);      
                printf("Cell %d: left = %3d \t right = %3d \t bottom = %3d \t top = %3d \t (level = %3d) \n",
                   ic, neigh2d_test[ic].left, neigh2d_test[ic].right, neigh2d_test[ic].bottom, neigh2d_test[ic].top, level[ic]);      
                printf("\n");
          }
      }
      free(neigh2d_test);
   } else {
      printf("\tnot_run,   ");
   }




   free(neigh2d_gold);
   free(level);
   free(x);
   free(y);
   free(i);
   free(j);
}

struct neighbor2d *neighbors2d_bruteforce( uint ncells, int *i, int *j, int *level )
{
   struct neighbor2d *neigh2d = (struct neighbor2d *)malloc(ncells*sizeof(struct neighbor2d));

   for (uint index1 = 0; index1 < ncells; index1++) {
      int lev = level[index1];
      int ii = i[index1];
      int jj = j[index1];

      int left  = index1;
      for (uint index2 = 0; index2 < ncells; index2++) {
         if ( abs(level[index2] - lev) > 1) continue;
         if ( (level[index2] == lev && i[index2] ==  ii-1    && j[index2] == jj  ) ||
              (level[index2] <  lev && i[index2] == (ii-1)/2 && j[index2] == jj/2) ||
              (level[index2] >  lev && i[index2] ==  ii*2-1  && j[index2] == jj*2) ) {
            left = index2;
            break;
         }
      }
      int right = index1;
      for (uint index2 = 0; index2 < ncells; index2++) {
         if ( abs(level[index2] - lev) > 1) continue;
         if ( (level[index2] == lev && i[index2] ==  ii+1    && j[index2] == jj  ) ||
              (level[index2] <  lev && i[index2] == (ii+1)/2 && j[index2] == jj/2) ||
              (level[index2] >  lev && i[index2] == (ii+1)*2 && j[index2] == jj*2) ) {
            right = index2;
            break;
         }
      }
      int bottom = index1;
      for (uint index2 = 0; index2 < ncells; index2++) {
         if ( abs(level[index2] - lev) > 1) continue;
         if ( (level[index2] == lev && i[index2] == ii   && j[index2] ==  jj-1)    ||
              (level[index2] <  lev && i[index2] == ii/2 && j[index2] == (jj-1)/2) ||
              (level[index2] >  lev && i[index2] == ii*2 && j[index2] ==  jj*2-1)  ) {
            bottom = index2;
            break;
         }
      }
      int top = index1;
      for (uint index2 = 0; index2 < ncells; index2++) {
         if ( abs(level[index2] - lev) > 1) continue;
         if ( (level[index2] == lev && i[index2] == ii   && j[index2] ==  jj+1   ) ||
              (level[index2] < lev && i[index2] == ii/2 && j[index2] == (jj+1)/2) ||
              (level[index2] > lev && i[index2] == ii*2 && j[index2] == (jj+1)*2) ) {
            top = index2;
            break;
         }
      }
      neigh2d[index1].left   = left;
      neigh2d[index1].right  = right;
      neigh2d[index1].bottom = bottom;
      neigh2d[index1].top    = top;
   }

   return(neigh2d);
}

struct neighbor2d *neighbors2d_kdtree( uint ncells, int mesh_size, double *x, double *y, int *level ) 
{
   TKDTree2d tree;

   KDTree_Initialize2d(&tree);

   TBounds2d box;
   for(uint ic = 0; ic < ncells; ic++) {
      double lev_power = pow(2,(double)level[ic]);
      box.min.x = x[ic]-1.0*0.5/lev_power/mesh_size;
      box.max.x = x[ic]+1.0*0.5/lev_power/mesh_size;
      box.min.y = y[ic]-1.0*0.5/lev_power/mesh_size;
      box.max.y = y[ic]+1.0*0.5/lev_power/mesh_size;
      //printf("Adding cell %d : xmin %lf xmax %lf ymin %lf ymax %lf\n",ic,box.min.x,box.max.x,box.min.y,box.max.y);
      KDTree_AddElement2d(&tree, &box);
   }

   struct neighbor2d *neigh2d = (struct neighbor2d *)malloc(ncells*sizeof(struct neighbor2d));

   int index_list[20];
   int num;
   for (uint ic = 0; ic < ncells; ic++) {
      neigh2d[ic].left = ic;
      neigh2d[ic].right = ic;
      neigh2d[ic].bottom = ic;
      neigh2d[ic].top = ic;
      double lev_power = pow(2,(double)level[ic]);
      box.min.x = x[ic]-1.1*1.0*0.5/lev_power/mesh_size;
      box.max.x = x[ic]-1.1*1.0*0.5/lev_power/mesh_size;
      box.min.y = y[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      box.max.y = y[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      KDTree_QueryBoxIntersect2d(&tree, &num, &(index_list[0]), &box);
      if (num == 1) neigh2d[ic].left = index_list[0];

      box.min.x = x[ic]+1.1*1.0*0.5/lev_power/mesh_size;
      box.max.x = x[ic]+1.1*1.0*0.5/lev_power/mesh_size;
      box.min.y = y[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      box.max.y = y[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      KDTree_QueryBoxIntersect2d(&tree, &num, &(index_list[0]), &box);
      if (num == 1) neigh2d[ic].right = index_list[0];

      box.min.x = x[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      box.max.x = x[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      box.min.y = y[ic]-1.1*1.0*0.5/lev_power/mesh_size;
      box.max.y = y[ic]-1.1*1.0*0.5/lev_power/mesh_size;
      KDTree_QueryBoxIntersect2d(&tree, &num, &(index_list[0]), &box);
      if (num == 1) neigh2d[ic].bottom = index_list[0];

      box.min.x = x[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      box.max.x = x[ic]-0.5*1.0*0.5/lev_power/mesh_size;
      box.min.y = y[ic]+1.1*1.0*0.5/lev_power/mesh_size;
      box.max.y = y[ic]+1.1*1.0*0.5/lev_power/mesh_size;
      KDTree_QueryBoxIntersect2d(&tree, &num, &(index_list[0]), &box);
      if (num == 1) neigh2d[ic].top = index_list[0];

   }

   KDTree_Destroy2d(&tree);

   return(neigh2d);
}

struct neighbor2d *neighbors2d_hashcpu( uint ncells, int mesh_size, int levmx, int *i, int *j, int *level )
{
   struct neighbor2d *neigh2d = (struct neighbor2d *)malloc(ncells*sizeof(struct neighbor2d));

   int *levtable = (int *)malloc(levmx+1);
   for (int lev=0; lev<levmx+1; lev++){
      levtable[lev] = (int)pow(2,lev);
   }
   int jmaxsize = mesh_size*levtable[levmx];
   int imaxsize = mesh_size*levtable[levmx];

   int **hash = (int **)genmatrix(jmaxsize, imaxsize, sizeof(int));

   for (int jj = 0; jj<jmaxsize; jj++){
      for (int ii = 0; ii<imaxsize; ii++){
         hash[jj][ii]=-1;
      }
   }

   for(int ic=0; ic<ncells; ic++){
      int lev = level[ic];
      if (lev == levmx) {
         hash[j[ic]][i[ic]] = ic;
      } else {
         for (int jj = j[ic]*levtable[levmx-lev]; jj < (j[ic]+1)*levtable[levmx-lev]; jj++) {
            for (int ii=i[ic]*levtable[levmx-lev]; ii<(i[ic]+1)*levtable[levmx-lev]; ii++) {
               hash[jj][ii] = ic;
            }
         }
      }
   }

   for (int ic=0; ic<ncells; ic++){
      int ii = i[ic];
      int jj = j[ic];
      int lev = level[ic];
      int levmult = levtable[levmx-lev];
      neigh2d[ic].left   = hash[      jj   *levmult               ][MAX(  ii   *levmult-1, 0         )];
      neigh2d[ic].right  = hash[      jj   *levmult               ][MIN( (ii+1)*levmult,   imaxsize-1)];
      neigh2d[ic].bottom = hash[MAX(  jj   *levmult-1, 0)         ][      ii   *levmult               ];
      neigh2d[ic].top    = hash[MIN( (jj+1)*levmult,   jmaxsize-1)][      ii   *levmult               ];
   }

   return(neigh2d);
}

cl_mem neighbors2d_hashgpu( uint ncells, int mesh_size, int levmx, cl_mem i_buffer, cl_mem j_buffer,
      cl_mem level_buffer, cl_mem levtable_buffer, double *time )
{
   cl_mem hash_buffer, neighbor2d_buffer;

   cl_int error = 0;
   long gpu_time = 0;

   int *levtable = (int *)malloc(levmx+1);
   for (int lev=0; lev<levmx+1; lev++){
      levtable[lev] = (int)pow(2,lev);
   }
   int imaxsize = mesh_size*levtable[levmx];
   int jmaxsize = mesh_size*levtable[levmx];

   free(levtable);

   uint hash_size = (uint)(imaxsize*jmaxsize);

   hash_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, hash_size*sizeof(int), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   /******************
    * Init to -1
    ******************/

   error = clSetKernelArg(init_kernel, 0, sizeof(cl_uint), &hash_size);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(init_kernel, 1, sizeof(cl_mem), (void*)&hash_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   size_t global_work_size[1];
   size_t local_work_size[1];

   local_work_size[0] = TILE_SIZE;
   global_work_size[0] = ((hash_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];

   cl_event hash_init_event;

   error = clEnqueueNDRangeKernel(queue, init_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_init_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   global_work_size[0] = ((ncells+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];

   error = clSetKernelArg(hash_setup_kernel, 0, sizeof(cl_uint), &ncells);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 1, sizeof(cl_int), &mesh_size);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 2, sizeof(cl_int), &levmx);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 3, sizeof(cl_mem), &levtable_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 4, sizeof(cl_mem), &i_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 5, sizeof(cl_mem), &j_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 6, sizeof(cl_mem), &level_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_setup_kernel, 7, sizeof(cl_mem), &hash_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   cl_event hash_setup_event;

   error = clEnqueueNDRangeKernel(queue, hash_setup_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_setup_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   neighbor2d_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells*sizeof(cl_uint4), NULL, &error);
   if (error != CL_SUCCESS) {
      //printf("Error is %d at line %d\n",error,__LINE__);
      clReleaseMemObject(hash_buffer);
      clReleaseMemObject(neighbor2d_buffer);
      return(NULL);
   }

   error = clSetKernelArg(calc_neighbor2d_kernel, 0, sizeof(cl_uint), &ncells);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 1, sizeof(cl_int), &mesh_size);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 2, sizeof(cl_int), &levmx);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 3, sizeof(cl_mem), &levtable_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 4, sizeof(cl_mem), &i_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 5, sizeof(cl_mem), &j_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 6, sizeof(cl_mem), &level_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 7, sizeof(cl_mem), &hash_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(calc_neighbor2d_kernel, 8, sizeof(cl_mem), &neighbor2d_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   cl_event calc_neighbor2d_event;

   error = clEnqueueNDRangeKernel(queue, calc_neighbor2d_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &calc_neighbor2d_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);


   long gpu_time_start, gpu_time_end;

   clWaitForEvents(1,&calc_neighbor2d_event);

   clGetEventProfilingInfo(hash_init_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
   clGetEventProfilingInfo(hash_init_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
   gpu_time += gpu_time_end - gpu_time_start;
   clReleaseEvent(hash_init_event);

   if (DETAILED_TIMING) printf("\tinit %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

   clGetEventProfilingInfo(hash_setup_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
   clGetEventProfilingInfo(hash_setup_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
   gpu_time += gpu_time_end - gpu_time_start;
   clReleaseEvent(hash_setup_event);

   if (DETAILED_TIMING) printf("\tsetup %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

   clGetEventProfilingInfo(calc_neighbor2d_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
   clGetEventProfilingInfo(calc_neighbor2d_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
   gpu_time += gpu_time_end - gpu_time_start;
   clReleaseEvent(calc_neighbor2d_event);

   if (DETAILED_TIMING) printf("\tsetup %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

   *time = (double)gpu_time*1.0e-9;

   clReleaseMemObject(hash_buffer);

   return(neighbor2d_buffer);

}

// adaptiveMeshConstructor()
// Inputs: n (width/height of the square mesh), l (maximum level of refinement),
//         pointers for the level, x, and y arrays (should be NULL for all three)
// Output: number of cells in the adaptive mesh
//
int adaptiveMeshConstructorWij(const int n, const int l, 
         int** level_ptr, double** x_ptr, double** y_ptr, int **i_ptr, int **j_ptr) {
   int ncells = SQR(n);

   // ints used for for() loops later
   int ic, xc, yc, xlc, ylc, nlc;

   //printf("\nBuilding the mesh...\n");

   // Initialize Coarse Mesh
   int*  level = (int*)  malloc(sizeof(int)*ncells);
   double* x   = (double*) malloc(sizeof(double)*ncells);
   double* y   = (double*) malloc(sizeof(double)*ncells);
   int*  i     = (int*)  malloc(sizeof(int)*ncells);
   int*  j     = (int*)  malloc(sizeof(int)*ncells);
   for(yc = 0; yc < n; yc++) {
      for(xc = 0; xc < n; xc++) {
         level[n*yc+xc] = 0;
         x[n*yc+xc]     = (real)(TWO*xc+ONE) / (real)(TWO*n);
         y[n*yc+xc]     = (real)(TWO*yc+ONE) / (real)(TWO*n);
         i[n*yc+xc]     = xc;
         j[n*yc+xc]     = yc;
      }
   }
   //printf("Coarse mesh initialized.\n");

   // Randomly Set Level of Refinement
   //unsigned int iseed = (unsigned int)time(NULL);
   //srand (iseed);
   srand (0);
   for(int ii = l; ii > 0; ii--) {
      for(ic = 0; ic < ncells; ic++) {
         int jj = 1 + (int)(10.0*rand() / (RAND_MAX+1.0));
         // XXX Consider distribution across levels: Clustered at 1 level XXX
         if(jj>5) {level[ic] = ii;}
      }
   }

   //printf("Levels of refinement randomly set.\n");

   // Smooth the Refinement
   int newcount = -1;
   while(newcount != 0) {
      newcount = 0;
      int lev = 0;
      for(ic = 0; ic < ncells; ic++) {
         lev = level[ic];
         lev++;
         // Check bottom neighbor
         if(ic - n >= 0) {
            if(level[ic-n] > lev) {
               level[ic] = lev;
               newcount++;
               continue;
            }
         }
         // Check top neighbor
         if(ic + n < ncells) {
            if(level[ic+n] > lev) {
               level[ic] = lev;
               newcount++;
               continue;
            }
         }
         // Check left neighbor
         if((ic%n)-1 >= 0) {
            if(level[ic-1] > lev) {
               level[ic] = lev;
               newcount++;
               continue;
            }
         }
         // Check right neighbor
         if((ic%n)+1 < n) {
            if(level[ic+1] > lev) {
               level[ic] = lev;
               newcount++;
               continue;
            }
         }
      }
   }
   //printf("Refinement smoothed.\n");

   // Allocate Space for the Adaptive Mesh
   newcount = 0;
   for(ic = 0; ic < ncells; ic++) {newcount += (powerOfFour(level[ic]) - 1);}
   int*  level_temp = (int*)  malloc(sizeof(int)*(ncells+newcount));
   double* x_temp   = (double*) malloc(sizeof(double)*(ncells+newcount));
   double* y_temp   = (double*) malloc(sizeof(double)*(ncells+newcount));
   int*  i_temp     = (int*)  malloc(sizeof(int)*(ncells+newcount));
   int*  j_temp     = (int*)  malloc(sizeof(int)*(ncells+newcount));

   // Set the Adaptive Mesh
   int offset = 0;
   for(yc = 0; yc < n; yc++) {
      for(xc = 0; xc < n; xc++) {
         ic = n*yc + xc;
         nlc = (int) SQRT( (real) powerOfFour(level[ic]) );
         for(ylc = 0; ylc < nlc; ylc++) {
            for(xlc = 0; xlc < nlc; xlc++) {
               level_temp[ic + offset + (nlc*ylc + xlc)] = level[ic];
               x_temp[ic + offset + (nlc*ylc + xlc)] = x[ic]-(ONE / (real)(TWO*n))
                                    + ((real)(TWO*xlc+ONE) / (real)(n*nlc*TWO));
               y_temp[ic + offset + (nlc*ylc + xlc)] = y[ic]-(ONE / (real)(TWO*n))
                                    + ((real)(TWO*ylc+ONE) / (real)(n*nlc*TWO));
               i_temp[ic + offset + (nlc*ylc + xlc)] = i[ic]*pow(2,level[ic]) + xlc;
               j_temp[ic + offset + (nlc*ylc + xlc)] = j[ic]*pow(2,level[ic]) + ylc;
            }         
         }
         offset += powerOfFour(level[ic])-1;
      }
   }
   //printf("Adaptive mesh built.\n");

   // Swap pointers and free memory used by Coarse Mesh
   swap_int(&level, &level_temp);
   swap_double(&x, &x_temp);
   swap_double(&y, &y_temp);
   swap_int(&i, &i_temp);
   swap_int(&j, &j_temp);
   free(level_temp);
   free(x_temp);
   free(y_temp);
   free(i_temp);
   free(j_temp);


   //printf("Old ncells: %d", ncells);
   // Update ncells
   ncells += newcount;
   //printf("\tNew ncells: %d\n", ncells);

   // Randomize the order of the arrays
   int* random = (int*) malloc(sizeof(int)*ncells);
   int* temp1 = (int*) malloc(sizeof(int)*ncells);
   real* temp2 = (real*) malloc(sizeof(real)*ncells*2);
   int* temp3 = (int*) malloc(sizeof(int)*ncells*2);
   // XXX Want better randomization? XXX
   // XXX Why is the time between printf() statements the longest part? XXX
   //printf("Shuffling");
   //fflush(stdout);
   for(ic = 0; ic < ncells; ic++) {random[ic] = ic;}
   //iseed = (unsigned int)time(NULL);
   //srand (iseed);
   srand(0);
   nlc = 0;
   for(int ii = 0; ii < 7; ii++) {
      for(ic = 0; ic < ncells; ic++) {
         int jj = (int)( ((real)ncells*rand()) / (RAND_MAX+ONE) );
         nlc = random[jj];
         random[jj] = random[ic];
         random[ic] = nlc;
      }
      //printf(".");
      //fflush(stdout);
   }
   //printf("\n");

   for(ic = 0; ic < ncells; ic++) {
      temp1[ic] = level[random[ic]];
      temp2[2*ic] = x[random[ic]];
      temp2[2*ic+1] = y[random[ic]];
      temp3[2*ic] = i[random[ic]];
      temp3[2*ic+1] = j[random[ic]];
   }
   for(ic = 0; ic < ncells; ic++) {
      level[ic] = temp1[ic];
      x[ic]     = temp2[2*ic];
      y[ic]     = temp2[2*ic+1];
      i[ic]     = temp3[2*ic];
      j[ic]     = temp3[2*ic+1];
   }

   free(temp1);
   free(temp2);
   free(temp3);
   free(random);
   //printf("Adaptive mesh randomized.\n");

   *level_ptr = level;
   *x_ptr = x;
   *y_ptr = y;
   *i_ptr = i;
   *j_ptr = j;

   //printf("Adaptive mesh construction complete.\n");
   return ncells;

}

void **genmatrix(int jnum, int inum, size_t elsize)
{
   void **out;
   int mem_size;

   mem_size = jnum*sizeof(void *);
   out      = (void **)malloc(mem_size);

   mem_size = jnum*inum*elsize;
   out[0]    = (void *)calloc(jnum*inum, elsize);

   for (int i = 1; i < jnum; i++) {
      out[i] = out[i-1] + inum*elsize;
   }

   return (out);
}

void genmatrixfree(void **var)
{
   free(var[0]);
   free(var);
}

