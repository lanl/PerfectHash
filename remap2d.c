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
typedef struct {
    double *x;
    double *y;
    int    *level;
    int    *i;
    int    *j;
} cell;
#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define SQRT(x) sqrt(x)
#else
typedef float real;
typedef cl_float cl_real;
typedef struct {
    float *x;
    float *y;
    int   *level;
    int   *i;
    int   *j;
} cell;
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define SQRT(x) sqrtf(x)
#endif

typedef unsigned int uint;

#define CHECK 1
#define TILE_SIZE 256

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))
#define SWAP_PTR(p1,p2,p3) ((p3=p1), (p1=p2), (p2=p3))
#endif

#define SQ(x) ((x)*(x))

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

// Cartesian Coordinate Indexing
#define two_to_the(ishift)       (1u <<(ishift) )
#define four_to_the(ishift)      (1u << ( (ishift)*2 ) )

#define HASHY (( two_to_the(levmx)*mesh_size ))
#define HASH_MAX (( SQ(HASHY) ))

/* CPU Timing Variables */
struct timeval timer;
double t1, t2;

/* OpenCL vairables */
cl_context context;
cl_command_queue queue;
cl_program program;
int is_nvidia=0;

cl_kernel remap_c_kernel, remap_r_kernel;

/* Declare Functions */
int adaptiveMeshConstructor(const int n, const int l, cell *mesh_ptr);
void remaps2d(int mesh_size, int levmx);
cl_mem parallelRemap2D(cell mesh_a, cl_mem a_x_buffer, cl_mem a_y_buffer, cl_mem a_level_buffer,
                                    cl_mem b_x_buffer, cl_mem b_y_buffer, cl_mem b_level_buffer,
                                    cl_mem V_buffer, int asize, int bsize, int mesh_size, int levmx);
void remap_brute2d(cell mesh_a, cell mesh_b, int asize, int bsize, real* V_a, real* V_remap, int mesh_size);
void remap_kDtree2d(cell mesh_a, cell mesh_b, int asize, int bsize, real* V_a, real* V_remap, int mesh_size, int levmx);


/* Begin Funtion Definitions */
int main (int argc, const char * argv[]) {

    cl_int error;
    int mesh_size, levmx;
    unsigned int iseed = (unsigned int)time(NULL);
    //srand(iseed);
    srand(1);

    GPUInit(&context, &queue, &is_nvidia, &program, "remap2d_kern.cl");

    remap_c_kernel = clCreateKernel(program,"remap_hash_creation_kern", &error);
    remap_r_kernel = clCreateKernel(program,"remap_hash_retrieval_kern", &error);

    printf("\t \t REMAP 2D \t \t\n Mesh Size, levmx, \t ncells_a, \t ncells_b, \t CPU Hash, \t CPU Brute, \t CPU k-D Tree, \t GPU Hash\n");
    for(mesh_size = 16; mesh_size <= 1024; mesh_size *= 2) {
       for(levmx = 1; levmx <= 8; levmx++) {
          if(SQ(mesh_size*two_to_the(levmx)) > two_to_the(28)) {
             levmx = 10;
              continue;
          }
          printf("\t %d, \t %d,",mesh_size,levmx);
          remaps2d(mesh_size, levmx);
       }
    }

}


void remaps2d(int mesh_size, int levmx) {

   cell mesh_a;
   cell mesh_b;

   int ncells_a = adaptiveMeshConstructor(mesh_size, levmx, &mesh_a);
   int ncells_b = adaptiveMeshConstructor(mesh_size, levmx, &mesh_b);
   printf(" \t %8d, \t %8d,",ncells_a,ncells_b);

   int icount = 0;
   if(ncells_a == ncells_b) {
      for(int ic = 0; ic < ncells_a; ic++) {
         if(mesh_a.x[ic] == mesh_b.x[ic] && mesh_a.y[ic] == mesh_b.y[ic])
            icount++;
      }
   }
   if(icount > 0)
      printf("\nThe adaptive mesh constructor not random enough: %d cells identical between Mesh A and Mesh B out of %d.\n \t \t \t \t",icount,ncells_a);

   real* V_a = (real*) malloc(ncells_a*sizeof(real));
   real* V_b = (real*) malloc(ncells_b*sizeof(real));
   real* V_remap = (real*) malloc(ncells_b*sizeof(real));

   for(int ic = 0; ic < ncells_a; ic++) {
      V_a[ic] = ONE / ((real)four_to_the(mesh_a.level[ic])*(real)SQ(mesh_size));
   }
   for(int ic = 0; ic < ncells_b; ic++) {
      V_b[ic] = ONE / ((real)four_to_the(mesh_b.level[ic])*(real)SQ(mesh_size));
   }
   for(int ic = 0; ic < ncells_b; ic++) {
      V_remap[ic] = ZERO;
   }

   /* CPU Hash Remap */
   gettimeofday(&timer, NULL);
   t1 = timer.tv_sec+(timer.tv_usec/1000000.0);

   // Initialize Hash Table
   int hsize = HASH_MAX;
   int* hash_table = (int*) malloc(hsize*sizeof(int));
   // Not needed -- for debug
   //for(int ic = 0; ic < hsize; ic++) {hash_table[ic] = -1;}

   uint i_max = mesh_size*two_to_the(levmx);

   // Fill Hash Table from Mesh A
   for(int ic = 0; ic < ncells_a; ic++) {
        uint lev = mesh_a.level[ic];
        uint i = mesh_a.i[ic];
        uint j = mesh_a.j[ic];
        // If at the maximum level just set the one cell
        if (lev == levmx) {
            hash_table[(j*i_max)+i] = ic;
        } else {
            // Set the square block of cells at the finest level
            // to the index number
            uint lev_mod = two_to_the(levmx - lev);
            for (uint jj = j*lev_mod; jj < (j+1)*lev_mod; jj++) {
                for (uint ii = i*lev_mod; ii < (i+1)*lev_mod; ii++) {
                    hash_table[(jj*i_max)+ii] = ic;
                }
            }
        }
   } 

   // Use Hash Table to Perform Remap
   for(int jc = 0; jc < ncells_b; jc++) {
      int lev = mesh_b.level[jc];
      int levmult = two_to_the(levmx - lev);
      int jbase = mesh_b.j[jc]*levmult;
      int ibase = mesh_b.i[jc]*levmult;
      real val_sum = 0.0;
      for(int jj = jbase; jj < jbase+levmult; jj++) {
         for(int ii = ibase; ii < ibase+levmult; ii++) {
            int ic = hash_table[jj*i_max+ii];
            int leva = mesh_a.level[ic];
            val_sum += V_a[ic] / (real)four_to_the(levmx-leva);
         }
      }
      V_remap[jc] += val_sum;
   }
   free(hash_table);

   gettimeofday(&timer, NULL);
   t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
   printf(" \t %.6lf,", t2 - t1);

   icount = 0;
   for(int ic = 0; ic < ncells_b; ic++) {
      if(V_b[ic] != V_remap[ic]) icount++;
   }
   if(icount > 0)
      printf("Error in the CPU Hash Remap for %d cells out of %d.\n",icount,ncells_b);

   // Reset remap array for Brute Force Remap
   for(int ic = 0; ic < ncells_b; ic++) {V_remap[ic] = ZERO;}

   if (ncells_a < 600000) {
      /* Brute Force Remap */
      gettimeofday(&timer, NULL);
      t1 = timer.tv_sec+(timer.tv_usec/1000000.0);

      remap_brute2d(mesh_a, mesh_b, ncells_a, ncells_b, V_a, V_remap, mesh_size);

      gettimeofday(&timer, NULL);
      t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
      printf(" \t %.6lf,", t2 - t1);

      icount = 0;
      for(int ic = 0; ic < ncells_b; ic++) {
         if(V_b[ic] != V_remap[ic]) icount++;
      }
      if(icount > 0)
         printf("Error in the Bruteforce Remap for %d cells out of %d.\n",icount,ncells_b);
   } else {
      printf(" \t not_run,   ");
   }

   // Reset remap array for k-D Tree Remap
   for(int ic = 0; ic < ncells_b; ic++) {V_remap[ic] = ZERO;}

   /* k-D Tree Remap */
   gettimeofday(&timer, NULL);
   t1 = timer.tv_sec+(timer.tv_usec/1000000.0);

   remap_kDtree2d(mesh_a, mesh_b, ncells_a, ncells_b, V_a, V_remap, mesh_size, levmx);

   gettimeofday(&timer, NULL);
   t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
   printf(" \t %.6lf,", t2 - t1);

   icount = 0;
   for(int ic = 0; ic < ncells_b; ic++) {
      if(V_b[ic] != V_remap[ic]) icount++;
   }
   if(icount > 0)
      printf("Error in the k-D Tree Remap for %d cells out of %d.\n",icount,ncells_b);

   // Reset remap array for GPU Hash Remap
   for(int ic = 0; ic < ncells_b; ic++) {V_remap[ic] = ZERO;}

   size_t needed_gpu_memory = HASH_MAX*sizeof(int)+ncells_a*sizeof(cell)+ncells_b*(sizeof(cell)+sizeof(real));

   //printf("  \t size is %lu",needed_gpu_memory);

   if (needed_gpu_memory < 900000000 || is_nvidia){
      /* GPU Hash Remap */
      cl_int error;
    
      cl_mem a_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_a*sizeof(real), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      cl_mem a_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_a*sizeof(real), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      cl_mem a_level_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_a*sizeof(int), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
      error = clEnqueueWriteBuffer(queue, a_x_buffer, CL_TRUE, 0, ncells_a*sizeof(real), mesh_a.x, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      error = clEnqueueWriteBuffer(queue, a_y_buffer, CL_TRUE, 0, ncells_a*sizeof(real), mesh_a.y, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      error = clEnqueueWriteBuffer(queue, a_level_buffer, CL_TRUE, 0, ncells_a*sizeof(int), mesh_a.level, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
      cl_mem b_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_b*sizeof(real), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      cl_mem b_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_b*sizeof(real), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      cl_mem b_level_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_b*sizeof(int), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
      error = clEnqueueWriteBuffer(queue, b_x_buffer, CL_TRUE, 0, ncells_b*sizeof(real), mesh_b.x, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      error = clEnqueueWriteBuffer(queue, b_y_buffer, CL_TRUE, 0, ncells_b*sizeof(real), mesh_b.y, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      error = clEnqueueWriteBuffer(queue, b_level_buffer, CL_TRUE, 0, ncells_b*sizeof(int), mesh_b.level, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
      cl_mem V_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells_a*sizeof(real), NULL, &error);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

      error = clEnqueueWriteBuffer(queue, V_buffer, CL_TRUE, 0, ncells_a*sizeof(real), V_a, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

      cl_mem remap_buffer = parallelRemap2D(mesh_a, a_x_buffer, a_y_buffer, a_level_buffer,
                                                    b_x_buffer, b_y_buffer, b_level_buffer,
                                                    V_buffer, ncells_a, ncells_b, mesh_size, levmx);
    
      error = clEnqueueReadBuffer(queue, remap_buffer, CL_TRUE, 0, ncells_b*sizeof(real), V_remap, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

      icount = 0;
      for(int ic = 0; ic < ncells_b; ic++) {
         if(V_b[ic] != V_remap[ic]) icount++;
      }
      if(icount > 0)
         printf("Error in the GPU Hash Remap for %d cells out of %d.\n",icount,ncells_b);
    
      clReleaseMemObject(a_x_buffer);
      clReleaseMemObject(a_y_buffer);
      clReleaseMemObject(a_level_buffer);
      clReleaseMemObject(b_x_buffer);
      clReleaseMemObject(b_y_buffer);
      clReleaseMemObject(b_level_buffer);
      clReleaseMemObject(V_buffer);
      clReleaseMemObject(remap_buffer);
   } else {
      printf(" \t not_run,   \n");
   }

   free(V_a);
   free(V_b);
   free(V_remap);

   free(mesh_a.x);
   free(mesh_a.y);
   free(mesh_a.level);
   free(mesh_b.x);
   free(mesh_b.y);
   free(mesh_b.level);
}

cl_mem parallelRemap2D(cell mesh_a, cl_mem a_x_buffer, cl_mem a_y_buffer, cl_mem a_level_buffer,
                                    cl_mem b_x_buffer, cl_mem b_y_buffer, cl_mem b_level_buffer,
                                    cl_mem V_buffer, int asize, int bsize, int mesh_size, int levmx) {
    
    cl_int error = 0;
    
    int temp_size = HASH_MAX;
    
    cl_mem temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, temp_size*sizeof(int), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    size_t global_work_size[1];
    size_t local_work_size[1];
    
    local_work_size[0] = TILE_SIZE;
    global_work_size[0] = ((asize+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
    
    /******************
     * Remap Into Hash Table
     ******************/
    
   error = clSetKernelArg(remap_c_kernel, 0, sizeof(cl_mem), (void*)&temp_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_c_kernel, 1, sizeof(cl_mem), (void*)&a_x_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_c_kernel, 2, sizeof(cl_mem), (void*)&a_y_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_c_kernel, 3, sizeof(cl_mem), (void*)&a_level_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_c_kernel, 4, sizeof(cl_int), &asize);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_c_kernel, 5, sizeof(cl_int), &mesh_size);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_c_kernel, 6, sizeof(cl_int), &levmx);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   cl_event hash_kernel_event;
   
   error = clEnqueueNDRangeKernel(queue, remap_c_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_kernel_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   
   /*****************
    * Remap Out of Hash Table
    *****************/
   
   cl_mem remap_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bsize*sizeof(real), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   
   global_work_size[0] = ((bsize+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];

   error = clSetKernelArg(remap_r_kernel, 0, sizeof(cl_mem),(void*)&remap_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 1, sizeof(cl_mem), (void*)&V_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 2, sizeof(cl_mem), (void*)&temp_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 3, sizeof(cl_mem), (void*)&a_x_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 4, sizeof(cl_mem), (void*)&a_y_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 5, sizeof(cl_mem), (void*)&a_level_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 6, sizeof(cl_mem), (void*)&b_x_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 7, sizeof(cl_mem), (void*)&b_y_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 8, sizeof(cl_mem), (void*)&b_level_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 9, sizeof(cl_int), &bsize);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 10, sizeof(cl_int), &mesh_size);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(remap_r_kernel, 11, sizeof(cl_int), &levmx);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
   cl_event remap_event;
    
   error = clEnqueueNDRangeKernel(queue, remap_r_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &remap_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
        
   long gpu_time_start, gpu_time_end, gpu_time=0;
    
   clWaitForEvents(1, &remap_event);
    
    clGetEventProfilingInfo(hash_kernel_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(hash_kernel_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(hash_kernel_event);
        
    clGetEventProfilingInfo(remap_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(remap_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(remap_event);
    
    printf(" \t %.6lf \n", (double)gpu_time*1.0e-9);
    
    clReleaseMemObject(temp_buffer);
    
    return remap_buffer;

}

// adaptiveMeshConstructor()
// Inputs: n (width/height of the square mesh), l (maximum level of refinement),
//         pointers for the level, x, and y arrays (should be NULL for all three)
// Output: number of cells in the adaptive mesh
//
int adaptiveMeshConstructor(const int n, const int l,
//                          int** level_ptr, real** x_ptr, real** y_ptr) {
                            cell *mesh) {
   int ncells = SQ(n);

   // ints used for for() loops later
   int ic, xc, yc, xlc, ylc, nlc;

   // Initialize Coarse Mesh
   int*  level = (int*)  malloc(sizeof(int)*ncells);
   real* x     = (real*) malloc(sizeof(real)*ncells);
   real* y     = (real*) malloc(sizeof(real)*ncells);
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

   // Randomly Set Level of Refinement
//   unsigned int iseed = (unsigned int)time(NULL);
//   unsigned int iseed = (unsigned int)(time(NULL)/100000.0);
//   srand (iseed);
   for(int ii = l; ii > 0; ii--) {
      for(ic = 0; ic < ncells; ic++) {
         int jj = 1 + (int)(10.0*rand() / (RAND_MAX+1.0));
         // XXX Consider distribution across levels: Clustered at 1 level? XXX
         if(jj>5) {level[ic] = ii;}
      }
   }

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

   // Allocate Space for the Adaptive Mesh
   newcount = 0;
   for(ic = 0; ic < ncells; ic++) {newcount += (four_to_the(level[ic]) - 1);}
   int*  level_temp = (int*)  malloc(sizeof(int)*(ncells+newcount));
   real* x_temp     = (real*) malloc(sizeof(real)*(ncells+newcount));
   real* y_temp     = (real*) malloc(sizeof(real)*(ncells+newcount));
   int*  i_temp     = (int*)  malloc(sizeof(int)*(ncells+newcount));
   int*  j_temp     = (int*)  malloc(sizeof(int)*(ncells+newcount));

   // Set the Adaptive Mesh
   int offset = 0;
   for(yc = 0; yc < n; yc++) {
      for(xc = 0; xc < n; xc++) {
         ic = n*yc + xc;
         nlc = (int) SQRT( (real) four_to_the(level[ic]) );
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
         offset += four_to_the(level[ic])-1;
      }
   }

   // Swap pointers and free memory used by Coarse Mesh
   swap_int(&level, &level_temp);
   swap_real(&x, &x_temp);
   swap_real(&y, &y_temp);
   swap_int(&i, &i_temp);
   swap_int(&j, &j_temp);
   free(level_temp);
   free(x_temp);
   free(y_temp);
   free(i_temp);
   free(j_temp);

   // Update ncells
   ncells += newcount;

   // Randomize the order of the arrays
   int* random = (int*) malloc(sizeof(int)*ncells);
   int* temp1 = (int*) malloc(sizeof(int)*ncells);
   real* temp2 = (real*) malloc(sizeof(real)*ncells*2);
   int* temp3 = (int*) malloc(sizeof(int)*ncells*2);
   // XXX Want better randomization? XXX
   for(ic = 0; ic < ncells; ic++) {random[ic] = ic;}
//   iseed = (unsigned int)time(NULL);
//   srand (iseed);
   nlc = 0;
   for(int ii = 0; ii < 7; ii++) {
      for(ic = 0; ic < ncells; ic++) {
         int jj = (int)( ((real)ncells*rand()) / (RAND_MAX+ONE) );
         nlc = random[jj];
         random[jj] = random[ic];
         random[ic] = nlc;
      }
   }


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

   mesh->x = (real*) malloc(sizeof(real)*ncells);
   mesh->y = (real*) malloc(sizeof(real)*ncells);
   mesh->level = (int*) malloc(sizeof(int)*ncells);
   mesh->i = (int*) malloc(sizeof(int)*ncells);
   mesh->j = (int*) malloc(sizeof(int)*ncells);
   for(ic = 0; ic < ncells; ic++) {
      mesh->x[ic]     = x[ic];
      mesh->y[ic]     = y[ic];
      mesh->level[ic] = level[ic];
      mesh->i[ic]     = i[ic];
      mesh->j[ic]     = j[ic];
   }

   free(x);
   free(y);
   free(level);
   free(i);
   free(j);

   return ncells;

}


void remap_brute2d(cell mesh_a, cell mesh_b, int asize, int bsize, real* V_a, real* V_remap, int mesh_size) {
    
    int ic, jc;
    real xmin_a, xmin_b, ymin_a, ymin_b, xmax_a, xmax_b, ymax_a, ymax_b;
    real radius_a, radius_b;
    real overlap_x, overlap_y;

    for (ic = 0; ic < bsize; ic++) {
       radius_b = ONE / ((real)mesh_size*two_to_the(mesh_b.level[ic]+1));
       xmin_b = mesh_b.x[ic] - radius_b;
       xmax_b = mesh_b.x[ic] + radius_b;
       ymin_b = mesh_b.y[ic] - radius_b;
       ymax_b = mesh_b.y[ic] + radius_b;

       for (jc = 0; jc < asize; jc++) {
          radius_a = ONE / ((real)mesh_size*two_to_the(mesh_a.level[jc]+1));
          xmin_a = mesh_a.x[jc] - radius_a;
          xmax_a = mesh_a.x[jc] + radius_a;
          ymin_a = mesh_a.y[jc] - radius_a;
          ymax_a = mesh_a.y[jc] + radius_a;

          overlap_x = MIN(xmax_a, xmax_b) - MAX(xmin_a, xmin_b);
          overlap_y = MIN(ymax_a, ymax_b) - MAX(ymin_a, ymin_b);

          if(overlap_x > ZERO && overlap_y > ZERO) {
             V_remap[ic] += (V_a[jc] * (overlap_x*overlap_y) / SQ(TWO*radius_a));
          }
       }
    }

    return;

}

void remap_kDtree2d(cell mesh_a, cell mesh_b, int asize, int bsize, real* V_a, real* V_remap, int mesh_size, int levmx) {

    int num;
    int index_list[four_to_the(levmx)*4];
    TKDTree2d tree;

    KDTree_Initialize2d(&tree);

    TBounds2d box;

    for(uint ic = 0; ic < asize; ic++) {
       real radius_a  = ONE / ((real)mesh_size*two_to_the(mesh_a.level[ic]+1));
       box.min.x = mesh_a.x[ic] - radius_a;
       box.max.x = mesh_a.x[ic] + radius_a;
       box.min.y = mesh_a.y[ic] - radius_a;
       box.max.y = mesh_a.y[ic] + radius_a;
       KDTree_AddElement2d(&tree, &box);
    }

    for(uint ic = 0; ic < bsize; ic++) {
       real radius_b  = ONE / ((real)mesh_size*two_to_the(mesh_b.level[ic]+1));
       box.min.x = mesh_b.x[ic] - radius_b;
       box.max.x = mesh_b.x[ic] + radius_b;
       box.min.y = mesh_b.y[ic] - radius_b;
       box.max.y = mesh_b.y[ic] + radius_b;

       KDTree_QueryBoxIntersect2d(&tree, &num, &(index_list[0]), &box);

       for(uint jc = 0; jc < num; jc++) {
          real radius_a  = ONE / ((real)mesh_size*two_to_the(mesh_a.level[index_list[jc]]+1));
          real xmin_a = mesh_a.x[index_list[jc]] - radius_a;
          real xmax_a = mesh_a.x[index_list[jc]] + radius_a;
          real ymin_a = mesh_a.y[index_list[jc]] - radius_a;
          real ymax_a = mesh_a.y[index_list[jc]] + radius_a;

          real overlap_x = MIN(xmax_a, box.max.x) - MAX(xmin_a, box.min.x);
          real overlap_y = MIN(ymax_a, box.max.y) - MAX(ymin_a, box.min.y);

          if(overlap_x > ZERO && overlap_y > ZERO) {
             V_remap[ic] += (V_a[index_list[jc]] * (overlap_x*overlap_y) / SQ(TWO*radius_a));
          }
       }
    }
    KDTree_Destroy2d(&tree);

   return;

}


