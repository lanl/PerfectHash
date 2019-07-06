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
#include "kdtree/KDTree1d.h"
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
#define ONE 1.0
#define TWO 2.0
#else
typedef float real;
typedef cl_float cl_real;
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
#endif

struct neighbor {
    uint left;
    uint right;
};

struct timespec tstart;
double time_sum;

int is_nvidia = 0;
#define BRUTE_FORCE_SIZE_LIMIT 500000

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel init_kernel, hash_kernel, get_neighbor_kernel;

void neighbors( uint length, double min_diff, double max_diff, double min_val );
struct neighbor *neighbors_bruteforce( uint length, double *xcoor, double min_val, double max_val);
struct neighbor *neighbors_kdtree( uint length, double *xcoor, double *xmin, double *xmax,
   double min_diff, double max_val, double min_val );
struct neighbor *neighbors_hashcpu( uint length, double *xcoor, double min_diff, double max_val, double min_val );
cl_mem neighbors_hashgpu( uint length, cl_mem data_buffer, double min_diff, double max_val, double min_val, double *time );
double generate_array_wminmax( uint size, double *ptr, double *xmin, double *xmax,
    double mindx, double maxdx, double min, double *max );

int main (int argc, const char * argv[]) {

    cl_int error;

    GPUInit(&context, &queue, &is_nvidia, &program, "neigh_kern.cl");

    init_kernel = clCreateKernel(program, "init_kern", &error);
    hash_kernel = clCreateKernel(program, "hash_kern", &error);
    get_neighbor_kernel = clCreateKernel(program, "get_neighbor_kern", &error);

    printf("\n    Neighbors Performance Results\n\n");
    if (LONG_RUNS == 1)
       printf("Size,   \tBrute,    \tkDtree   \tHash CPU, \tHash GPU\n");
    else
       printf("Size,   \tkDtree   \tHash CPU, \tHash GPU\n");

    for (uint max_mult = 1; max_mult <= 32; max_mult *= 2){
       printf("\nMax diff is %d times min_diff\n",max_mult);
       for( uint i = 64; i <= 5000000; i*=2 ) {
          printf("%d,     ", i);
          neighbors(i, 2.0, (double)max_mult*2.0, 0.0);
          printf("\n");
       }
    }
}

/* find right and left neighbors of element at index index in array of size length */
void neighbors( uint length, double min_diff, double max_diff, double min_val ) 
{
   double *xcoor, *xmin, *xmax;
   double max_val = min_val; //reset in generate array call
   struct neighbor *neigh_gold, *neigh_test;

   xcoor = (double*)malloc(length*sizeof(double));
   xmin  = (double*)malloc(length*sizeof(double));
   xmax  = (double*)malloc(length*sizeof(double));

   generate_array_wminmax(length, xcoor, xmin, xmax, min_diff, max_diff, min_val, &max_val);
   //for (uint i=0; i<length; i++) {printf("i %d xcoor %lf\n",i,xcoor[i]);}

   if (length < BRUTE_FORCE_SIZE_LIMIT) {
      cpu_timer_start(&tstart);
      neigh_gold = neighbors_bruteforce(length, xcoor, min_val, max_val);
      time_sum += cpu_timer_stop(tstart);
      printf("\t%.6lf,", time_sum);

#ifdef XXX
      printf("\n");
      for (uint index=0; index<length; index++){
         int left  = neigh_gold[index].left;
         int right = neigh_gold[index].right;
         printf("%2d: Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
            index, xcoor[index], right, xcoor[right], left, xcoor[left]);
      }
#endif

   } else {
      printf("\tnot_run,  ");
   }

   cpu_timer_start(&tstart);
   if (length < BRUTE_FORCE_SIZE_LIMIT)
      neigh_test = neighbors_kdtree(length, xcoor, xmin, xmax, min_diff, max_val, min_val);
   else
      neigh_gold = neighbors_kdtree(length, xcoor, xmin, xmax, min_diff, max_val, min_val);

   time_sum += cpu_timer_stop(tstart);
   printf("\t%.6lf,", time_sum);

#ifdef XXX
   for (uint index=0; index<length; index++){
      if (neigh_test[index].left != neigh_gold[index].left || neigh_test[index].right != neigh_gold[index].right){
         printf("%2d: neigh_test Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
            index, xcoor[index], neigh_test[index].right, xcoor[neigh_test[index].right], neigh_test[index].left, xcoor[neigh_test[index].left]);
         printf("%2d: neigh_gold Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
            index, xcoor[index], neigh_gold[index].right, xcoor[neigh_gold[index].right], neigh_gold[index].left, xcoor[neigh_gold[index].left]);
         printf("\n");
      }
   }
   if (length < 200000) free(neigh_test);
#endif

   cpu_timer_start(&tstart);
   neigh_test = neighbors_hashcpu(length, xcoor, min_diff, max_val, min_val);
   time_sum += cpu_timer_stop(tstart);
   printf("\t%.6lf,", time_sum);

   for (uint index=0; index<length; index++){
      if (neigh_test[index].left != neigh_gold[index].left || neigh_test[index].right != neigh_gold[index].right){
         printf("%2d: neigh_test Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
            index, xcoor[index], neigh_test[index].right, xcoor[neigh_test[index].right], neigh_test[index].left, xcoor[neigh_test[index].left]);
         printf("%2d: neigh_gold Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
            index, xcoor[index], neigh_gold[index].right, xcoor[neigh_gold[index].right], neigh_gold[index].left, xcoor[neigh_gold[index].left]);
         printf("\n");
      }
   }
   free(neigh_test);


   cl_int error = 0;
   cl_mem data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, length*sizeof(real), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clEnqueueWriteBuffer(queue, data_buffer, CL_TRUE, 0, length*sizeof(real), xcoor, 0, NULL, NULL);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   cl_mem neigh_buffer = neighbors_hashgpu(length, data_buffer, min_diff, max_val, min_val, &time_sum);
   clReleaseMemObject(data_buffer);

   if (neigh_buffer != NULL) {
      printf("\t%.6lf,", time_sum);

      neigh_test = (struct neighbor *)malloc(length*sizeof(struct neighbor));
      error = clEnqueueReadBuffer(queue, neigh_buffer, CL_TRUE, 0, length*sizeof(cl_uint2), neigh_test, 0, NULL, NULL);
      if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
      clReleaseMemObject(neigh_buffer);

      for (uint index=0; index<length; index++){
         if (neigh_test[index].left != neigh_gold[index].left || neigh_test[index].right != neigh_gold[index].right){
            printf("%2d: neigh_test Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
               index, xcoor[index], neigh_test[index].right, xcoor[neigh_test[index].right], neigh_test[index].left, xcoor[neigh_test[index].left]);
            printf("%2d: neigh_gold Element %.2lf  \tRight neighbor: index %2d val %.2lf   \tLeft neighbor index %2d val %.2lf\n",
               index, xcoor[index], neigh_gold[index].right, xcoor[neigh_gold[index].right], neigh_gold[index].left, xcoor[neigh_gold[index].left]);
            printf("\n");
         }
      }
      free(neigh_test);
   } else {
      printf("\tnot_run,  ");
   }

   free(xcoor);
   free(xmin);
   free(xmax);
   free(neigh_gold);
}

struct neighbor *neighbors_bruteforce( uint length, double *xcoor, double min_val, double max_val ) 
{
   double xleft, xright;
   int left=0, right=0;

   struct neighbor *neigh = (struct neighbor *)malloc(length*sizeof(struct neighbor));

   for (uint index1 = 0; index1 < length; index1++) {
      left  = index1;
      right = index1;
      xleft = min_val;
      xright = max_val;
      for (uint index2 = 0; index2 < length; index2++) {
         if (index2 == index1) continue;
         if (xcoor[index2] < xcoor[index1] && xcoor[index2] >= xleft  ) {xleft  = xcoor[index2]; left = index2; }

         if (xcoor[index2] > xcoor[index1] && xcoor[index2] <= xright ) {xright = xcoor[index2]; right = index2;}
      }
      neigh[index1].left = left;
      neigh[index1].right = right;
   }

   return(neigh);
}

struct neighbor *neighbors_kdtree( uint length, double *xcoor, double *xmin, double *xmax,
   double min_diff, double max_val, double min_val ) 
{
   TKDTree1d tree;

   KDTree_Initialize1d(&tree);

   TBounds1d box;
   for(uint i = 0; i < length; i++) {
     box.min.x = xmin[i];
     box.max.x = xmax[i];
     KDTree_AddElement1d(&tree, &box);
   }

   struct neighbor *neigh = (struct neighbor *)malloc(length*sizeof(struct neighbor));

   int index_list[10];
   int num;
   for (uint index = 0; index < length; index++) {
      neigh[index].left = index;
      neigh[index].right = index;
      box.min.x = xmin[index]-min_diff*0.25;
      box.max.x = xmin[index]-min_diff*0.20;
      KDTree_QueryBoxIntersect1d(&tree, &num, &(index_list[0]), &box);
      if (num == 1) neigh[index].left = index_list[0];

      box.min.x = xmax[index]+min_diff*0.20;
      box.max.x = xmax[index]+min_diff*0.25;
      KDTree_QueryBoxIntersect1d(&tree, &num, &(index_list[0]), &box);
      if (num == 1) neigh[index].right = index_list[0];
   }

   KDTree_Destroy1d(&tree);

   return(neigh);
}

/* find right and left neighbors of element at index index in array of size length */
struct neighbor *neighbors_hashcpu( uint length, double *xcoor, double min_diff, double max_val, double min_val ) 
{
   uint hash_size = (uint)((max_val - min_val)/min_diff + 2.5);	//create hash table with buckets of size min_diff -- +2.5 rounds up and adds one space to either side
   int *hash = (int*)malloc(hash_size*sizeof(int));
	
   /* Sort elements into hash array hash */
   memset(hash, -1, hash_size*sizeof(int));			//set all elements of hash array to -1
	
   for(uint i = 0; i < length; i++) { hash[(int)((xcoor[i]+min_val)/min_diff)] = i; }
   //place index of current xcoor element into hash according to where the xcoor value

   struct neighbor *neigh = (struct neighbor *)malloc(length*sizeof(struct neighbor));

   for (uint index = 0; index < length; index++) {
      /* move left and right through hash array from desired element to find its neighbors */
      int idx_new = (int)((xcoor[index]-min_val)/min_diff);	//where the index element is in the hash array
      int left = index, right = index;

      for(int i = idx_new+1; i < hash_size; i++) {	//store index of neigbor in original unsorted array, if greatest/least, than left as -1
         if(hash[i] != -1) {
            right = hash[i];
            break;
         }
      }
      for(int i = idx_new-1; i >= 0; i--) {
         if(hash[i]  != -1) {
            left = hash[i];
            break;
         }
      }
      neigh[index].left  = left;
      neigh[index].right = right;
   }

   free(hash);

   return(neigh);
}

/* find right and left neighbors of element at index index in array of size length */
cl_mem neighbors_hashgpu( uint length, cl_mem data_buffer, double min_diff, double max_val, double min_val, double *time ) 
{
   cl_mem hash_buffer, neighbor_buffer;

   cl_int error = 0;
   long gpu_time = 0;

   uint hash_size = (uint)((max_val - min_val)/min_diff + 2.5);	//create hash table with buckets of size min_diff -- +2.5 rounds up and adds one space to either side

   real min_val_real = (real)min_val;
   real min_diff_real = (real)min_diff;

   hash_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, hash_size*sizeof(int), NULL, &error);
   if (error != CL_SUCCESS) {
      //printf("Error is %d at line %d\n",error,__LINE__);
      return(NULL);
   }

   /******************
    * Init to -1
    *******************/
 
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

   /******************
    * Hash Kernel
    ******************/

   error = clSetKernelArg(hash_kernel, 0, sizeof(real), &min_val_real);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_kernel, 1, sizeof(real), &min_diff_real);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_kernel, 2, sizeof(cl_uint), &length);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_kernel, 3, sizeof(cl_mem), (void*)&data_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(hash_kernel, 4, sizeof(cl_mem), (void*)&hash_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   global_work_size[0] = ((length+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];

   cl_event hash_kernel_event;

   error = clEnqueueNDRangeKernel(queue, hash_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_kernel_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   /******************
    * Get Neighbor Kernel
    ******************/

   neighbor_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, length*sizeof(cl_uint2), NULL, &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   error = clSetKernelArg(get_neighbor_kernel, 0, sizeof(real), &min_val_real);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(get_neighbor_kernel, 1, sizeof(real), &min_diff_real);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(get_neighbor_kernel, 2, sizeof(cl_uint), &length);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(get_neighbor_kernel, 3, sizeof(cl_mem), (void*)&data_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(get_neighbor_kernel, 4, sizeof(cl_mem), (void*)&hash_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(get_neighbor_kernel, 5, sizeof(cl_uint), &hash_size);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
   error = clSetKernelArg(get_neighbor_kernel, 6, sizeof(cl_mem), &neighbor_buffer);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   cl_event get_neighbor_event;

   error = clEnqueueNDRangeKernel(queue, get_neighbor_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &get_neighbor_event);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

   long gpu_time_start, gpu_time_end;

   clWaitForEvents(1,&get_neighbor_event);

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

   clGetEventProfilingInfo(get_neighbor_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
   clGetEventProfilingInfo(get_neighbor_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
   gpu_time += gpu_time_end - gpu_time_start;
   clReleaseEvent(get_neighbor_event);

   if (DETAILED_TIMING) printf("hash %.6lf,", (double)(gpu_time_end - gpu_time_start)*1.0e-9);

   *time = (double)gpu_time*1.0e-9;

   clReleaseMemObject(hash_buffer);

   return(neighbor_buffer);

}

double generate_array_wminmax( uint size, double *ptr, double *xmin, double *xmax,
     double mindx, double maxdx, double min, double *max ) {
	
     double swap;
	int index, front = 0;
    double running_min = maxdx;
		
	struct timeval tim;				//random seeding
	gettimeofday(&tim, NULL);
	//srand(tim.tv_sec*tim.tv_usec);
	
	srand(0);
	
	ptr[0] = min;		//start the array using the minimum value
	
	/* for each element, add a random value between mindx and maxdx to the previous element's value */
	for(int i = 1; i < size; i++) {
		ptr[i] = ptr[i-1] + mindx + ((double)rand() * (maxdx - mindx) / (double)RAND_MAX);
        if(ptr[i]-ptr[i-1] < running_min) running_min = ptr[i]-ptr[i-1];
	}


	*max = ptr[size-1];					//set the max value to the last element's value
	//*max = min + (size-1) * maxdx;	//force the range for timings isolating a different variable
	
        xmin[0] = min;
        for (int i=1; i<size; i++){
           xmin[i] = (ptr[i] + ptr[i-1]) * 0.5;
           xmax[i-1] = xmin[i];
        }
        xmax[size-1]=*max;

	/* Mix up the array by selecting elements from shrinking front portion of array and placing them on back end of array */
	for(int i = 0; (i < size) && (size - i != 0) ; i++) {
		index = rand() % (size - i - front) + front;
		swap = ptr[size-i-1];
		ptr[size-i-1] = ptr[index];
		ptr[index] = swap;
                swap = xmin[size-i-1];
                xmin[size-i-1] = xmin[index];
                xmin[index] = swap;
                swap = xmax[size-i-1];
                xmax[size-i-1] = xmax[index];
                xmax[index] = swap;
	}
    return running_min;
}
