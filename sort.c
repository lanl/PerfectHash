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
#define EPS 1.0e-12
#else
typedef float real;
typedef cl_float cl_real;
#define EPS 1.0e-7
#endif

#define SQR(x) (( (x)*(x) ))

typedef unsigned int uint;

#define CHECK 1
#define TILE_SIZE 256
#define DETAILED_TIMING 0

struct timeval timer;
double t1, t2;

int is_nvidia = 0;

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel init_kernel, hash_kernel, scan1_kernel, scan2_kernel, scan3_kernel;

void sorts( uint length, double min_diff, double max_diff, double min_val );
cl_mem parallelHash( uint length, cl_mem arr, double min_diff, double max_diff, double min_val, double max_val, double *time );
double* hashsort( uint length, double *arr, double min_diff, double min_val, double max_val );
double generate_array( uint size, double *ptr, double mindx, double maxdx, double min, double *max );

//int compare (const void * a, const void * b) { return ( *(double*)a - *(double*)b ); }

int compare (const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

int main (int argc, const char * argv[]) 
{
    cl_int error;

    GPUInit(&context, &queue, &is_nvidia, &program, "sort_kern.cl");

    struct timeval tim;                //random seeding
    gettimeofday(&tim, NULL);
    //srand(tim.tv_sec*tim.tv_usec);

    srand(0);

    init_kernel = clCreateKernel(program, "init_kern", &error);
    hash_kernel = clCreateKernel(program, "hash_kern", &error);
    scan1_kernel = clCreateKernel(program, "scan1", &error);
    scan2_kernel = clCreateKernel(program, "scan2", &error);
    scan3_kernel = clCreateKernel(program, "scan3", &error);

    printf("\n    Sorting Performance Results\n\n");
#ifdef __APPLE_CC__
    printf("Size,   \tQsort,    \tHeapsort, \tMergesort, \tHash CPU, \tHash GPU\n");
#else
    printf("Size,   \tQsort,    \tHash CPU, \tHash GPU\n");
#endif

    uint max_size = 0;
#ifdef HAVE_CL_DOUBLE
    max_size = 100000000;
#else
    max_size = 10000000;
#endif
    //else max_size = 131071;

    for (uint max_mult = 2; max_mult <= 8; max_mult *= 2){
       printf("\nMax diff is %d times min_diff\n",max_mult);
       for( uint i = 1024; i <= max_size; i*=2 ) {
#ifndef HAVE_CL_DOUBLE
          if (max_mult > 2  && i > 5000000) continue;
          if (max_mult > 4  && i > 4000000) continue;
          if (max_mult > 8  && i > 2000000) continue;
          if (max_mult > 16  && i > 1000000) continue;
#endif
          if (max_mult > 10 && i > 50000000) continue;
          if (max_mult > 30 && i > 20000000) continue;
          printf("%d,     ", i);
          sorts(i, 2.0, (double)max_mult*2.0, 0.0);
          printf("\n");
       }
    }

}

void sorts( uint length, double min_diff, double max_diff, double min_val ) {
    int icount;
    cl_int error = 0;
    double max_val = min_val; //reset in generate_array call
    double *sorted=NULL, *sort_test=NULL, *arr=NULL;
    
    arr = (double*)malloc(length*sizeof(double));
    
    //generate randomly shuffled array with given conditions to be sorted
    generate_array(length, arr, min_diff, max_diff, min_val, &max_val);
    
    /* Qsort */
    sorted = (double*)malloc(length*sizeof(double));
    for(uint i = 0; i < length; i++) { sorted[i] = arr[i]; }
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
    qsort(sorted, length, sizeof(double), compare);
    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("\t%.6lf,", t2 - t1);


#ifdef __APPLE_CC__
    /* Heapsort */
    sort_test = (double*)malloc(length*sizeof(double));
    for(uint i = 0; i < length; i++) { sort_test[i] = arr[i]; }
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
    heapsort(sort_test, length, sizeof(double), compare);
    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("\t%.6lf,", t2 - t1);
#ifdef CHECK
    for(uint i = 0; i < length; i++) { if (sort_test[i] != sorted[i]) printf("Check failed for heapsort index %d heapsort value %lf gold standard %lf\n",i,sort_test[i],sorted[i]); }
#endif
    free(sort_test);
    sort_test = NULL;

    /* Mergesort */
    sort_test = (double*)malloc(length*sizeof(double));
    for(uint i = 0; i < length; i++) { sort_test[i] = arr[i]; }
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
    mergesort(sort_test, length, sizeof(double), compare);
    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("\t%.6lf,", t2 - t1);
#ifdef CHECK
    for(uint i = 0; i < length; i++) { if (sort_test[i] != sorted[i]) printf("Check failed for mergesort index %d mergesort value %lf gold standard %lf\n",i,sort_test[i],sorted[i]); }
#endif
    free(sort_test);
    sort_test = NULL;
#endif


    /* Hashsort CPU */
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
    sort_test = hashsort(length, arr, min_diff, min_val, max_val);
    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("\t%.6lf,", t2 - t1);
#ifdef CHECK
    icount=0;
    for(uint i = 0; i < length; i++) {
       if (sort_test[i] != sorted[i]) {
          printf("Check failed for hashsort CPU index %d hashsort value %lf gold standard %lf\n",i,sort_test[i],sorted[i]);
          icount++;
       }
    }
#endif
    free(sort_test);
    sort_test = NULL;


    uint hash_size = (uint)((max_val - min_val)/min_diff + 2.5);
    uint alloc_size = 2*length*sizeof(real)+hash_size*sizeof(int)+(hash_size+hash_size-1)/TILE_SIZE*sizeof(int);
    //printf("\tSize is %lu\t", alloc_size);
    if (is_nvidia || alloc_size < 850000000) {
       /* Hashsort GPU */
       real *arr_real = (real*)malloc(length*sizeof(real));
       for(uint i = 0; i < length; i++) { arr_real[i] = (real)arr[i]; }
       cl_mem xcoor_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, length*sizeof(real), NULL, &error);
       cl_mem sorted_buffer = NULL;
       if (xcoor_buffer != NULL) {
          if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
          error = clEnqueueWriteBuffer(queue, xcoor_buffer, CL_TRUE, 0, length*sizeof(real), arr_real, 0, NULL, NULL);
          if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

          sorted_buffer = parallelHash(length, xcoor_buffer,  min_diff, max_diff, min_val, max_val, &t2);
          clReleaseMemObject(xcoor_buffer);
       }
       free(arr_real);
       if (sorted_buffer != NULL) {

          real *sort_real = (real*)malloc(length*sizeof(real));
          error = clEnqueueReadBuffer(queue, sorted_buffer, CL_TRUE, 0, length*sizeof(real), sort_real, 0, NULL, NULL);
          if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
          clReleaseMemObject(sorted_buffer);

          printf("\t%.6lf,", t2);
          sort_test = (double*)malloc(length*sizeof(double));
          for(uint i = 0; i < length; i++) { sort_test[i] = (double)sort_real[i]; }
          free(sort_real);
#ifdef CHECK

          icount=0;
          for(uint i = 0; i < length; i++) {
             if (fabs(sort_test[i] - sorted[i])/sorted[i] > EPS) {
                printf("Check failed for hashsort GPU index %d hashsort value %lf gold standard %lf\n",i,sort_test[i],sorted[i]);
                icount++;
             }
             if (icount > 20) exit(0);
          }
#endif
          free(sort_test);
          sort_test = NULL;
       } else {
          printf("\tnot_run,  ");
       } 
    } else {
       printf("\tnot_run,  ");
    }


    free(sorted);
    sorted = NULL;
    free(arr);
    arr=NULL;
}

double* hashsort( uint length, double *arr, double min_diff, double min_val, double max_val ) {
    uint hash_size;
    int *hash=NULL;
    double *sorted=NULL;
    
    sorted = (double*)malloc(length*sizeof(double));

    //create hash table with buckets of size min_diff 
    //   -- +2.5 rounds up and adds one space to either side
    hash_size = (uint)((max_val - min_val)/min_diff + 2.5);
    hash = (int*)malloc(hash_size*sizeof(int));

    //set all elements of hash array to -1
    memset(hash, -1, hash_size*sizeof(int));
    
    for(uint i = 0; i < length; i++) {
       //place index of current arr element into hash according to where the arr value
        hash[(int)((arr[i]-min_val)/min_diff)] = i;
    }
    
    int count=0;
    for(uint i = 0; i < hash_size; i++) {
        if(hash[i] >= 0) {
            //sweep through hash and put set values in a sorted array
            sorted[count] = arr[hash[i]];
            count++;
        }
    }
    
    free(hash);
    return sorted;
}

/* generate a randomly mixed up array with size size to be stored in pointer. the elements will have a minimum value min, and
    the difference between elements when sorted will be between mindx and maxdx. the maximum value is recorded in max. */
double generate_array( uint size, double *ptr, double mindx, double maxdx, double min, double *max ) {
    
    double swap;
    int index, front = 0;
    double running_min = maxdx;
        
    ptr[0] = min;        //start the array using the minimum value
    
    /* for each element, add a random value between mindx and maxdx to the previous element's value */
    for(int i = 1; i < size; i++) {
        ptr[i] = ptr[i-1] + mindx + ((double)rand() * (maxdx - mindx) / (double)RAND_MAX);
        if(ptr[i]-ptr[i-1] < running_min) running_min = ptr[i]-ptr[i-1];
    }

    *max = ptr[size-1];                    //set the max value to the last element's value
    //*max = min + (size-1) * maxdx;    //force the range for timings isolating a different variable
    
    /* Mix up the array by selecting elements from shrinking front portion of array and placing them on back end of array */
    for(int i = 0; (i < size) && (size - i != 0) ; i++) {
        index = rand() % (size - i - front) + front;
        swap = ptr[size-i-1];
        ptr[size-i-1] = ptr[index];
        ptr[index] = swap;
    }
    return running_min;
}

cl_mem parallelHash( uint length, cl_mem xcoor_buffer, double min_diff, double max_diff, double min_val, double max_val, double *time ) {

    cl_mem sorted_buffer, hash_buffer, ioffset_buffer;
 
    cl_int error = 0;
    long gpu_time = 0;
 
    uint hash_size = (uint)((max_val - min_val)/min_diff + 2.5);
 
    real min_val_real = (real)min_val;
    real min_diff_real = (real)min_diff;
    
    hash_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, hash_size*sizeof(int), NULL, &error);
    if (error != CL_SUCCESS) {
       //printf("Error is %d at line %d\n",error,__LINE__);
       return(NULL);
    }

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

/******************
 * Hash Kernel
 ******************/
     
    error = clSetKernelArg(hash_kernel, 0, sizeof(real), &min_val_real);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 1, sizeof(real), &min_diff_real);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 2, sizeof(cl_uint), &length);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 3, sizeof(cl_mem), (void*)&xcoor_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(hash_kernel, 4, sizeof(cl_mem), (void*)&hash_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
 
    global_work_size[0] = ((length+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
 
    cl_event hash_kernel_event;
    
    error = clEnqueueNDRangeKernel(queue, hash_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_kernel_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

/***********************
 * Prefix Scan Kernels
 ***********************/

    /* scan 1 */
    global_work_size[0] = ((hash_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
 
    int group_size = (int)(global_work_size[0]/local_work_size[0]);
    
    ioffset_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, group_size*sizeof(uint), NULL, &error);
    if (error != CL_SUCCESS) {
       //printf("Error is %d at line %d\n",error,__LINE__);
       clReleaseMemObject(hash_buffer);
       return(NULL);
    }
  
    error = clSetKernelArg(scan1_kernel, 0, sizeof(cl_uint), &hash_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan1_kernel, 1, sizeof(cl_mem), (void*)&ioffset_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan1_kernel, 2, local_work_size[0]*sizeof(uint), NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan1_kernel, 3, sizeof(cl_mem), (void*)&hash_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
  
    cl_event scan1_event;
    
    error = clEnqueueNDRangeKernel(queue, scan1_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &scan1_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

    //clWaitForEvents(1, &scan1_event);
    //exit(0);

    /* scan 2 */
    //global_work_size[0] = ((group_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
    global_work_size[0] = local_work_size[0];

    cl_event scan2_event;
    
    //printf("\n local: %d global: %d\n", local_work_size[0], global_work_size[0]);

        
    int elements_per_thread = (group_size+local_work_size[0]-1)/local_work_size[0];
    //printf("\ngroup_size %d EPT %d\n",group_size,elements_per_thread );
                
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

    //printf("\n");
    //for (int i=0; i<local_work_size[0]; i++){
    //   printf("%d mailbox %d\n",i,mailbox[i]);
    //}

    //int *hash = (int *)malloc(hash_size*sizeof(int));
    //error = clEnqueueReadBuffer(queue, hash_buffer, CL_TRUE, 0, hash_size*sizeof(int), hash, 0, NULL, NULL);
    //if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);

    //printf("\n");
    //for (int i=0; i<hash_size; i++){
    //   printf("%d hash %d\n",i,hash[i]);
    //}
#endif
 
    /* scan 3 */
    sorted_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, length*sizeof(real), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    global_work_size[0] = ((hash_size+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
        
    error = clSetKernelArg(scan3_kernel, 0, sizeof(cl_uint), &hash_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 1, sizeof(cl_mem), (void*)&ioffset_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 2, local_work_size[0]*sizeof(uint), NULL);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 3, sizeof(cl_mem), (void*)&hash_buffer) ;
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 4, sizeof(cl_mem), (void *)&xcoor_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(scan3_kernel, 5, sizeof(cl_mem), (void *)&sorted_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    cl_event scan3_event;
    
    if (clEnqueueNDRangeKernel(queue, scan3_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &scan3_event) != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
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
    clReleaseMemObject(hash_buffer);
    clReleaseMemObject(ioffset_buffer);
    
    return(sorted_buffer);
}

