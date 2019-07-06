/*
 *  Copyright (c) 2012-2019, Triad National Security, LLC.
 *  All rights Reserved.
 *
 * Copyright 2012-2019.  Triad National Security, LLC. This material was produced
 * under U.S. Government contract 89233218CNA000001 for Los Alamos National 
 * Laboratory (LANL), which is operated by Triad National Security, LLC
 * for the U.S. Department of Energy. The U.S. Government has rights to use,
 * reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
 * TRIAD NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
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
 * This is LANL Copyright Disclosure C13002/LA-CC-12-022
 *
 */

/*
 *  Authors: Bob Robey       XCP-2   brobey@lanl.gov
 *           David Nicholaeff        dnic@lanl.gov, mtrxknight@aol.com
 *           Rachel Robey            rnrobey@gmail.com
 */

/* 2-D Hash Sort */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include "gpu.h"
#include "timer.h"

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
#define SQRT(x) sqrt(x)
#define ONE 1.0
#define TWO 2.0
typedef struct {
   long level;
   double x;
   double y;
} cell;
#else
typedef float real;
typedef cl_float cl_real;
#define SQRT(x) sqrtf(x)
#define ONE 1.0f
#define TWO 2.0f
typedef struct {
   long level;
   float x;
   float y;
} cell;
#endif

typedef unsigned int uint;

#define CHECK 1
#define TILE_SIZE 256
#define DETAILED_TIMING 0
#define LONG_RUNS 1

/* Basic Functions */
#define SQR(x) (( (x)*(x) ))
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
int powerOfTwo(int n) {
   int result = 1;
   int i;
   for(i = 0; i < n; i++) {
      result *= 2;
   }
   return result;
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


struct timespec tstart;
double time_sum;

int mesh_size;
#define MESH_SIZE mesh_size
int levmx;

int adaptiveMeshConstructor(const int n, const int l, cell** mesh_ptr);
void adaptiveMeshDestructor(cell* mesh) {free(mesh);}

#define category cell
int* hashsort2d(const category* array, category* array_sorted, int size,
//                real* key1, real* key2, int* key3, int key4)
                cell* mesh, int key4);

int compare_cells(const void* a, const void* b);

int is_nvidia = 0;

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel init_kernel, hash_kernel, scan1_kernel, scan2_kernel, scan3_kernel;

cl_mem parallelHash(int ncells, int levmx, int mesh_size, cl_mem mesh_buffer, double *time);

/*
// XXX Will comparison operator have access to levmx?
int compare_cells(const void* a, const void* b) {
   return hashkey2d( (*(cell*)a).x, (*(cell*)a).y, (*(cell*)a).level, levmx )
         - hashkey2d( (*(cell*)b).x, (*(cell*)b).y, (*(cell*)a).level, levmx );
}
*/

int main(int argc, const char * argv[]) {

   cl_int error;

   GPUInit(&context, &queue, &is_nvidia, &program, "sort2d_kern.cl");

   init_kernel = clCreateKernel(program, "hash_init_cl", &error);
   hash_kernel = clCreateKernel(program, "hash_build_cl", &error);
   scan1_kernel = clCreateKernel(program, "scan1", &error);
   scan2_kernel = clCreateKernel(program, "scan2", &error);
   scan3_kernel = clCreateKernel(program, "scan3", &error);

   printf("\n    2D Sorting Performance Results\n\n");
#ifdef __APPLE_CC__
   printf("Size,   \tQsort,    \tHeapsort, \tMergesort, \tHash CPU, \tHash GPU\n");
#else
   printf("MeshSize, \tNcells   \tQsort,    \tHash CPU, \tHash GPU\n");
#endif

   int ic;

   for(levmx = 1; levmx <= 6; levmx++) {
      printf("\nMax levels is %d\n",levmx);
      for(mesh_size = 16; mesh_size <= 1024; mesh_size *= 2) {
         //if(SQR(mesh_size*powerOfTwo(levmx)) > powerOfTwo(30)) continue;

         if (levmx > 3 && mesh_size > 512) continue;
         if (levmx > 5 && mesh_size > 256) continue;

         cell* mesh = NULL;

         int icount;

         int ncells = adaptiveMeshConstructor(mesh_size, levmx, &mesh); 
                  
         cell* sorted        = (cell*) malloc(sizeof(cell)*ncells);
         cell* unsorted      = (cell*) malloc(sizeof(cell)*ncells);
         cell* sorted_temp   = (cell*) malloc(sizeof(cell)*ncells);
         for(ic = 0; ic < ncells; ic++) {
            sorted[ic].x      = mesh[ic].x; //(real) hashkey2d(x[ic],y[ic],level[ic],levmx);
            sorted[ic].y      = mesh[ic].y;
            sorted[ic].level  = mesh[ic].level;

            unsorted[ic].x      = mesh[ic].x;
            unsorted[ic].y      = mesh[ic].y;
            unsorted[ic].level  = mesh[ic].level;

            sorted_temp[ic].x     = unsorted[ic].x;
            sorted_temp[ic].y     = unsorted[ic].y;
            sorted_temp[ic].level = unsorted[ic].level;
         }
         printf("%d,      ", mesh_size);
         printf("\t%d,     ", ncells);

         /* Quicksort CPU */
         cpu_timer_start(&tstart);
         qsort(sorted, ncells, sizeof(cell), compare_cells);
         time_sum += cpu_timer_stop(tstart);
         printf("\t%.6lf,", time_sum);

         /* Hashsort CPU */
         int* hash_table = NULL;
	      cpu_timer_start(&tstart);

         hash_table = hashsort2d(unsorted, sorted_temp, ncells, mesh, levmx);
	      time_sum += cpu_timer_stop(tstart);
	      printf("\t%.6lf,", time_sum);
#ifdef CHECK
         icount = 0;
         for(ic = 0; ic < ncells; ic++) {
            if(sorted_temp[ic].x != sorted[ic].x || sorted_temp[ic].y != sorted[ic].y) {
               printf("Check failed for hashsort CPU index %d hashsort x value %lf gold standard %lf \t hashsort y value %lf gold standard %lf\n",ic,sorted_temp[ic].x,sorted[ic].x,sorted_temp[ic].y,sorted[ic].y);
               icount++;
            }
         }
#endif
         free(hash_table); 
         for(ic = 0; ic < ncells; ic++) {
            sorted_temp[ic].x     = unsorted[ic].x;
            sorted_temp[ic].y     = unsorted[ic].y;
            sorted_temp[ic].level = unsorted[ic].level;
         }

         /* Hashsort GPU */
        cl_mem unsorted_mesh_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells*sizeof(cell), NULL, &error);
        cl_mem sorted_mesh_buffer = NULL;
        if (error == CL_SUCCESS) {
           error = clEnqueueWriteBuffer(queue, unsorted_mesh_buffer, CL_TRUE, 0, ncells*sizeof(cell), unsorted, 0, NULL, NULL);
           if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

           sorted_mesh_buffer = parallelHash(ncells, levmx, mesh_size, unsorted_mesh_buffer, &time_sum);
           clReleaseMemObject(unsorted_mesh_buffer);
        }

        if (sorted_mesh_buffer != NULL) {
           error = clEnqueueReadBuffer(queue, sorted_mesh_buffer, CL_TRUE, 0, ncells*sizeof(cell), sorted_temp, 0, NULL, NULL);
           if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
           clReleaseMemObject(sorted_mesh_buffer);

	   printf("\t%.6lf,", time_sum);

// Should not have round-off since we are just moving data
#ifdef CHECK
            icount = 0;
            for(ic = 0; ic < ncells; ic++) {
               if(sorted_temp[ic].x != sorted[ic].x || sorted_temp[ic].y != sorted[ic].y) {
                  printf("Check failed for hashsort GPU index %d hashsort x value %lf gold standard %lf \t hashsort y value %lf gold standard %lf\n",ic,sorted_temp[ic].x,sorted[ic].x,sorted_temp[ic].y,sorted[ic].y);
                  icount++;
               }
               if (icount > 20) exit(0);
            }
            if (icount > 0) printf("\tCheck failed for hashsort GPU on %d cells out of %d.\n",icount,ncells);
#endif
        } else {
	   printf("\tnot_run,   ");
        }


         printf("\n");

         free(sorted);
         free(unsorted);
         free(sorted_temp);
         free(mesh); // adaptiveMeshDestructor(mesh);
      }
   }   

}



// adaptiveMeshConstructor()
// Inputs: n (width/height of the square mesh), l (maximum level of refinement),
//         pointers for the level, x, and y arrays (should be NULL for all three)
// Output: number of cells in the adaptive mesh
//
int adaptiveMeshConstructor(const int n, const int l,
//                          int** level_ptr, real** x_ptr, real** y_ptr) {
                            cell** mesh_ptr) {
   int ncells = SQR(n);

   // ints used for for() loops later
   int i, ic, xc, yc, xlc, ylc, j, nlc;

   // Initialize Coarse Mesh
   int*  level = (int*)  malloc(sizeof(int)*ncells);
   real* x     = (real*) malloc(sizeof(real)*ncells);
   real* y     = (real*) malloc(sizeof(real)*ncells);
   for(yc = 0; yc < n; yc++) {
      for(xc = 0; xc < n; xc++) {
         level[n*yc+xc] = 0;
         x[n*yc+xc]     = (real)(TWO*xc+ONE) / (real)(TWO*n);
         y[n*yc+xc]     = (real)(TWO*yc+ONE) / (real)(TWO*n);
      }
   }

   // Randomly Set Level of Refinement
   unsigned int iseed = (unsigned int)time(NULL);
   srand (iseed);
   for(i = l; i > 0; i--) {
      for(ic = 0; ic < ncells; ic++) {
         j = 1 + (int)(10.0*rand() / (RAND_MAX+1.0));
         // XXX Consider distribution across levels: Clustered at 1 level? XXX
         if(j>5) {level[ic] = i;}
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
   for(ic = 0; ic < ncells; ic++) {newcount += (powerOfFour(level[ic]) - 1);}
   int*  level_temp = (int*)  malloc(sizeof(int)*(ncells+newcount));
   real* x_temp     = (real*) malloc(sizeof(real)*(ncells+newcount));
   real* y_temp     = (real*) malloc(sizeof(real)*(ncells+newcount));

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
            }         
         }
         offset += powerOfFour(level[ic])-1;
      }
   }

   // Swap pointers and free memory used by Coarse Mesh
   swap_int(&level, &level_temp);
   swap_real(&x, &x_temp);
   swap_real(&y, &y_temp);
   free(level_temp);
   free(x_temp);
   free(y_temp);

   // Update ncells
   ncells += newcount;

   // Randomize the order of the arrays
   int* random = (int*) malloc(sizeof(int)*ncells);
   int* temp1 = (int*) malloc(sizeof(int)*ncells);
   real* temp2 = (real*) malloc(sizeof(real)*ncells*2);
   // XXX Want better randomization? XXX
   for(ic = 0; ic < ncells; ic++) {random[ic] = ic;}
   iseed = (unsigned int)time(NULL);
   srand (iseed);
   nlc = 0;
   for(i = 0; i < 7; i++) {
      for(ic = 0; ic < ncells; ic++) {
         j = (int)( ((real)ncells*rand()) / (RAND_MAX+ONE) );
         nlc = random[j];
         random[j] = random[ic];
         random[ic] = nlc;
      }
   }


   for(ic = 0; ic < ncells; ic++) {
      temp1[ic] = level[random[ic]];
      temp2[2*ic] = x[random[ic]];
      temp2[2*ic+1] = y[random[ic]];
   }
   for(ic = 0; ic < ncells; ic++) {
      level[ic] = temp1[ic];
      x[ic]     = temp2[2*ic];
      y[ic]     = temp2[2*ic+1];
   }

   free(temp1);
   free(temp2);
   free(random);

   cell* mesh = (cell*) malloc(sizeof(cell)*ncells);
   for(ic = 0; ic < ncells; ic++) {
      mesh[ic].x     = x[ic];
      mesh[ic].y     = y[ic];
      mesh[ic].level = level[ic];
   }

   free(x);
   free(y);
   free(level);

   *mesh_ptr = mesh;

   return ncells;

}


//#define HASH_FACTOR 10
//#define MESH_SIZE

// Cartesian Coordinate Indexing
#define HASHY (( powerOfTwo(key4)*MESH_SIZE ))
#define XY_TO_IJ(x) (( (x-(ONE/(TWO*(real)MESH_SIZE*(real)powerOfTwo(key3))))*(real)HASHY ))
#define HASH_MAX (( SQR(HASHY) ))
#define HASH_KEY (( XY_TO_IJ(key1) + XY_TO_IJ(key2)*(real)HASHY ))

int hashkey2d(real key1, real key2, int key3, int key4) {

   return (int) HASH_KEY;

}

// XXX Will comparison operator have access to levmx?
int compare_cells(const void* a, const void* b) {
   return hashkey2d( (*(cell*)a).x, (*(cell*)a).y, (*(cell*)a).level, levmx )
         - hashkey2d( (*(cell*)b).x, (*(cell*)b).y, (*(cell*)b).level, levmx );
}
// XXX XXX

// XXX CHANGE INPUT TO cell* XXX
// 2-D Spatial Hash Sort
// key1 is x value, key2 is y value, key3 is level, and key4 is levmx
int* hashsort2d(const category* array, category* array_sorted, int size,
//                real* key1, real* key2, int* key3, int key4) {
                cell* mesh, int key4) {

   // Size of the hash table
//   int hsize = size*HASH_FACTOR;
   int hsize = HASH_MAX; // XXX Needed until compression technique discovered XXX

   int i,j;

   // Allocate and initialize hash table
   int* hash_table = (int*) malloc(sizeof(int)*hsize);
   for(i = 0; i< hsize; i++)
      hash_table[i] = -1;

   // Calculate the hash key, normalizing to the size of the hash table;
   // Map indices of elements in array to the hash table
   int hash_key = -1;
   for(i = 0; i < size; i++) {
//      hash_key = hashkey2d(key1[i], key2[i], key3[i], key4) 
        hash_key = hashkey2d(mesh[i].x, mesh[i].y, mesh[i].level, key4) 
                 ; // * hsize / HASH_MAX; // Normalize hash key
      hash_table[hash_key] = i;
   }


   // Fill sorted array by striding through the hash table
   i = 0;
   for(j = 0; j < hsize; j++) {
//      if(j == hsize - 1)
//         printf("The last cell filled is %d\n",i);
      if(hash_table[j] == -1)
         continue;
      else {
         array_sorted[i].x     = array[hash_table[j]].x;
         array_sorted[i].y     = array[hash_table[j]].y;
         array_sorted[i].level = array[hash_table[j]].level;
         i++;
      }
   }

   return hash_table;

}


// Cartesian Coordinate Indexing
#define HASHY_PARALLEL (( powerOfTwo(levmx)*MESH_SIZE ))
#define HASH_MAX_PARALLEL (( SQR(HASHY_PARALLEL) ))

cl_mem parallelHash(int ncells, int levmx, int mesh_size, cl_mem mesh_buffer, double *time) {

    cl_mem sorted_buffer, temp_buffer, ioffset_buffer;
 
    cl_int error = 0;
    long gpu_time = 0;
 
    int temp_size = HASH_MAX_PARALLEL;
    
    temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, temp_size*sizeof(int), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

/******************
 * Init to -1
 ******************/
  
    error = clSetKernelArg(init_kernel, 0, sizeof(cl_int), &temp_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(init_kernel, 1, sizeof(cl_mem), (void*)&temp_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
 
    size_t global_work_size[1];
    size_t local_work_size[1];
    
    local_work_size[0] = TILE_SIZE;
    global_work_size[0] = ((temp_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
    
    cl_event hash_init_event;
 
    error = clEnqueueNDRangeKernel(queue, init_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_init_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

/******************
 * Hash Kernel
 ******************/
     
    error = clSetKernelArg(hash_kernel, 0, sizeof(cl_mem), (void*)&mesh_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 1, sizeof(cl_int), &levmx);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 2, sizeof(cl_int), &ncells);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 3, sizeof(cl_mem), (void*)&temp_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 4, sizeof(cl_int), &mesh_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
 
    global_work_size[0] = ((ncells+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
 
    cl_event hash_kernel_event;
    
    error = clEnqueueNDRangeKernel(queue, hash_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_kernel_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

/***********************
 * Prefix Scan Kernels
 ***********************/

    /* scan 1 */
    global_work_size[0] = ((temp_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
 
    int group_size = (int)(global_work_size[0]/local_work_size[0]);
    
    ioffset_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, group_size*sizeof(uint), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
  
    error = clSetKernelArg(scan1_kernel, 0, sizeof(cl_uint), &temp_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan1_kernel, 1, sizeof(cl_mem), (void*)&ioffset_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan1_kernel, 2, local_work_size[0]*sizeof(uint), NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan1_kernel, 3, sizeof(cl_mem), (void*)&temp_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
  
    cl_event scan1_event;
    
    error = clEnqueueNDRangeKernel(queue, scan1_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &scan1_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

    /* scan 2 */
    //global_work_size[0] = ((group_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
    global_work_size[0] = local_work_size[0];

    cl_event scan2_event;
    
    int elements_per_thread = (group_size+local_work_size[0]-1)/local_work_size[0];
    //printf("\tgroup_size %d EPT %d\t",group_size,elements_per_thread );

    error = clSetKernelArg(scan2_kernel, 0, local_work_size[0]*sizeof(uint), NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan2_kernel, 1, sizeof(cl_mem), (void*)&ioffset_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan2_kernel, 2, sizeof(uint), &group_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clEnqueueNDRangeKernel(queue, scan2_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &scan2_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

#ifdef XXX
    uint *ioffset = (uint *)malloc(group_size*sizeof(uint));
    error = clEnqueueReadBuffer(queue, ioffset_buffer, CL_TRUE, 0, group_size*sizeof(uint), ioffset, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

    printf("\n");
    for (uint i=0; i<group_size; i++){
       printf("%d ioffset %u\n",i,ioffset[i]);
    }
 
    uint *mailbox = (uint *)malloc(local_work_size[0]*sizeof(uint));
    error = clEnqueueReadBuffer(queue, mailbox_buffer, CL_TRUE, 0, local_work_size[0]*sizeof(int), mailbox, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
#endif
 
    /* scan 3 */ // XXX XXX XXX 
    global_work_size[0] = ((temp_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
        
    sorted_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ncells*sizeof(cell), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    error = clSetKernelArg(scan3_kernel, 0, sizeof(cl_uint), &temp_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 1, sizeof(cl_mem), (void*)&ioffset_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 2, local_work_size[0]*sizeof(uint), NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 3, sizeof(cl_mem), (void*)&temp_buffer) ;
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 4, sizeof(cl_mem), (void *)&mesh_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 5, sizeof(cl_mem), (void *)&sorted_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

    cl_event scan3_event;
    
    if (clEnqueueNDRangeKernel(queue, scan3_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &scan3_event) != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    cell *sorted_temp = (cell *)malloc(ncells*sizeof(cell));
    error = clEnqueueReadBuffer(queue, sorted_buffer, CL_TRUE, 0, ncells*sizeof(cell), sorted_temp, 0, NULL, NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

    long gpu_time_start, gpu_time_end;
    
    clWaitForEvents(1, &scan3_event);
    clGetEventProfilingInfo(hash_init_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(hash_init_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(hash_init_event);
    
    if (DETAILED_TIMING) printf("\tinit %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

    clGetEventProfilingInfo(hash_kernel_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(hash_kernel_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(hash_kernel_event);

    if (DETAILED_TIMING) printf("hash %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

    clGetEventProfilingInfo(scan1_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(scan1_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(scan1_event);
    
    if (DETAILED_TIMING) printf("scan 1 %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

    clGetEventProfilingInfo(scan2_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(scan2_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(scan2_event);
    
    if (DETAILED_TIMING) printf("scan 2 %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

    clGetEventProfilingInfo(scan3_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
    clGetEventProfilingInfo(scan3_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
    gpu_time += gpu_time_end - gpu_time_start;
    clReleaseEvent(scan3_event);

    if (DETAILED_TIMING) printf("scan 3 %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

    *time = (double)gpu_time*1.0e-9;

    /* cleanup */
    clReleaseMemObject(temp_buffer);
    clReleaseMemObject(ioffset_buffer);
    
    return(sorted_buffer);
}

