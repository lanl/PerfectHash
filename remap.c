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
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include "kdtree/KDTree1d.h"
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
#else
typedef float real;
typedef cl_float cl_real;
#endif

typedef unsigned int uint;

#define CHECK 1
#define TILE_SIZE 256

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))
#endif
#define SWAP_PTR(p1,p2,p3) ((p3=p1), (p1=p2), (p2=p3))

uint seed = 0;

/* Structs to hold higher and lower cell boundaries */
struct cell {
    double low;
    double high;
};
struct rcell {
    real low;
    real high;
};

/* CPU Timing Variables */
struct timeval timer;
double t1, t2;

/* OpenCL variables */
cl_context context;
cl_command_queue queue;
cl_program program;
int is_nvidia=0;
cl_kernel cHash_kernel, remap1_kernel;

/* Declare Functions */
int* hashsort( uint length, real *arr, real min_diff, real min_val, real max_val );
real* hashsort2( uint length, real *arr, real *values, real *sorted_values, real min_diff, real min_val, real max_val );

void remaps( int asize, int bsize, double min_diff, double max_diff, double min_val );
real* remap1( struct rcell *arr_a, real *original_values, struct rcell *arr_b, int asize, int bsize, real max_a, real min_val, real min_diff );
real* remap2( real *arr_old_in, real *original_values, real *arr_new_in, int asize, int bsize, real max_a, real max_b, real min_val, real min_diff );
real* remap_kdTree( struct rcell *x, real* original_values, struct rcell *x_new, int ncells, int new_cells, double min_diff, double max_diff, double min_val, double max_a );
real* remap_bruteforce( struct rcell *x, real* original_values, struct rcell *x_new, int ncells, int new_cells, double min_diff, double max_diff, double min_val, double max_a );

void generateRealCells( int size, struct rcell *ptr, int mindx, int maxdx, real min, real *max );

cl_mem parallelRemap1( cl_mem a_buffer, cl_mem v_buffer, cl_mem b_buffer, uint asize, uint bsize, real max_a, real min_val, real min_diff, double *time );

int compare (const void * a, const void * b) { return ( *(double*)a - *(double*)b ); }

/* Begin Funtion Definitions */
int main (int argc, const char * argv[]) {

    cl_int error;

    GPUInit(&context, &queue, &is_nvidia, &program, "remap_kern.cl");

    cHash_kernel = clCreateKernel(program, "cellHash_kern", &error);
    remap1_kernel = clCreateKernel(program, "remap1_kern", &error);

    printf("             REMAP\n\n");

    if (is_nvidia) 
       printf("size, Brute Force, CPU kD Tree, CPU Hash1, CPU Hash2, NVIDIA Hash1\n");
    else
       printf("size, Brute Force, CPU kD Tree, CPU Hash1, CPU Hash2, ATI Hash1\n");
    for( int levmx = 1; levmx < 10; levmx++) {
       printf("\nlevmx is %d\n\n",levmx);
       for( int i = 1024; i < 50000000; i *= 2) {
           printf("%d, ", i);
           remaps(i, i, 1.0, (double)levmx, 0.0);
           printf("\n");
       }
    }
    
}

int* hashsort( uint length, real *arr, real min_diff, real min_val, real max_val ) {
    
    uint temp_size;
    
    int *index = (int*)malloc(length*sizeof(int));

    temp_size = (uint)((max_val - min_val)/min_diff + 2.5);    //create hash table with buckets of size min_diff -- +2.5 rounds up and adds one space to either side
    int *temp = (int*)malloc(temp_size*sizeof(int));

    memset(temp, -1, temp_size*sizeof(int));            //set all elements of temp hash array to -1
    
    for(uint i = 0; i < length; i++) {
        temp[(int)((arr[i]-min_val)/min_diff)] = i;    //place index of current arr element into temp according to where the arr value
    }
    
    int count=0;
    for(uint i = 0; i < temp_size; i++) {
        if(temp[i] >= 0) {
            index[count] = temp[i];                            //sweep through temp and put set values in a sorted array
            count++;
        }
    }
    
    free(temp);
    return index;
}

real* hashsort2( uint length, real *arr, real *values, real *sorted_values, real min_diff, real min_val, real max_val ) {
    
    uint temp_size;
    
    real *sorted = (real*)malloc(length*sizeof(real));

    temp_size = (uint)((max_val - min_val)/min_diff + 2.5);    //create hash table with buckets of size min_diff -- +2.5 rounds up and adds one space to either side
    int *temp = (int*)malloc(temp_size*sizeof(int));

    memset(temp, -1, temp_size*sizeof(int));            //set all elements of temp hash array to -1
    
    for(uint i = 0; i < length; i++) {
        temp[(int)((arr[i]-min_val)/min_diff)] = i;    //place index of current arr element into temp according to where the arr value
    }
    
    int count=0;
    for(uint i = 0; i < temp_size; i++) {
        if(temp[i] >= 0) {
            sorted[count] = arr[temp[i]];                            //sweep through temp and put set values in a sorted array
            sorted_values[count] = values[temp[i]];                            //sweep through temp and put set values in a sorted array
            count++;
        }
    }
    
    free(temp);
    return sorted;
}

void remaps( int asize, int bsize, double min_diff, double max_diff, double min_val ) {
    struct rcell *arr_a, *arr_b;
    real max_a, max_b;
    int i;
    
    real *remap_gold, *remap_test;
    
    arr_a = (struct rcell*)malloc(asize*sizeof(struct rcell));
    arr_b = (struct rcell*)malloc(bsize*sizeof(struct rcell));
    
    generateRealCells( asize, arr_a, min_diff, max_diff, min_val, &max_a );
    generateRealCells( bsize, arr_b, min_diff, max_diff, min_val, &max_b );
    
    real *original_values = (real *)malloc((asize+1)*sizeof(real));
    for (i = 0; i<asize; i++) { original_values[i] = arr_a[i].high - arr_a[i].low; }

    real *arr_old_in = (real *)malloc((asize+1)*sizeof(real));
    real *arr_new_in = (real *)malloc((bsize+1)*sizeof(real));
    
    for(int a = 0; a < asize; a++) {
        arr_old_in[a] = arr_a[a].low;
    }
    for(int b = 0; b < bsize; b++) {
        arr_new_in[b] = arr_b[b].low;
    }
    arr_old_in[asize] = max_a;
    arr_new_in[bsize] = max_b;
    original_values[asize] = 0.0;


    /* Brute Force remap */
    if (asize < 600000) {
       gettimeofday(&timer, NULL);
       t1 = timer.tv_sec+(timer.tv_usec/1000000.0); //begin timer
    
       remap_gold = remap_bruteforce(arr_a, original_values, arr_b, asize, bsize, min_diff, max_diff, min_val, max_a);

       gettimeofday(&timer, NULL);
       t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
       printf("%.6lf, ", t2 - t1);
    } else {
       printf("not_run, ");
    }

    /* CPU kD Tree */
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0); //begin timer
    
    if (asize < 600000) {
       remap_test = remap_kdTree(arr_a, original_values, arr_b, asize, bsize, min_diff, max_diff, min_val, max_a);
    } else {
       remap_gold = remap_kdTree(arr_a, original_values, arr_b, asize, bsize, min_diff, max_diff, min_val, max_a);
    }

    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("%.6lf, ", t2 - t1);

    if (asize < 600000){
       for(i = 0; i < bsize; i++) {
          if(fabs(remap_gold[i] - remap_test[i]) > 1.0e-6)
             printf("Check failed for remap_kdTree CPU index %d remap value %lf gold standard %lf\n", i, remap_test[i], remap_gold[i]);
       }
       free(remap_test);
    }
    
    /* CPU Hash Remap1 */
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0); //begin timer

    //for (int ic = 0; ic < asize; ic++){
    //   printf("Array In %d %lf %lf value %lf\n",ic,arr_a[ic].low,arr_a[ic].high,original_values[ic]);
    //}

    //for (int ic = 0; ic < bsize; ic++){
    //   printf("Array In %d %lf %lf\n",ic,arr_b[ic].low,arr_b[ic].high);
    //}

    remap_test = remap1( arr_a, original_values, arr_b, asize, bsize, max_a, min_val, min_diff );

    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("%.6lf, ", t2 - t1);
    
    int icount = 0;
    for(i = 0; i < bsize; i++) {
       if(fabs(remap_gold[i] - remap_test[i]) > 1.0e-6){
          printf("Check failed for remap1 CPU index %d remap value %lf gold standard %lf\n", i, remap_test[i], remap_gold[i]);
          icount++;
       }
       if (icount > 20) exit(0);
    }
    free(remap_test);

    /* CPU Hash Remap2 */
    gettimeofday(&timer, NULL);
    t1 = timer.tv_sec+(timer.tv_usec/1000000.0); //begin timer

    remap_test = remap2( arr_old_in, original_values, arr_new_in, asize, bsize, max_a, max_b, min_val, min_diff );
    
    gettimeofday(&timer, NULL);
    t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
    printf("%.6lf, ", t2 - t1);
    
    icount = 0;
    for(i = 0; i < bsize; i++) {
       if(fabs(remap_gold[i] - remap_test[i]) > 1.0e-6){
          printf("Check failed for remap2 CPU index %d remap value %lf gold standard %lf\n", i, remap_test[i], remap_gold[i]);
          icount++;
       }
       if (icount > 20) exit(0);
    }
    free(remap_test);

    if (is_nvidia || (max_diff < 8.0 && asize < 20000000) || asize < 10000000) {

       /* GPU Hash Remap1 */
       cl_int error;
    
       cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, asize*sizeof(struct rcell), NULL, &error);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
       error = clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, asize*sizeof(struct rcell), arr_a, 0, NULL, NULL);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
       cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bsize*sizeof(struct rcell), NULL, &error);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
       error = clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, bsize*sizeof(struct rcell), arr_b, 0, NULL, NULL);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
       cl_mem v_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, asize*sizeof(real), NULL, &error);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
       error = clEnqueueWriteBuffer(queue, v_buffer, CL_TRUE, 0, asize*sizeof(real), original_values, 0, NULL, NULL);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
       cl_mem remap_buffer = parallelRemap1( a_buffer, v_buffer, b_buffer, (uint)asize, (uint)bsize, max_a, min_val, min_diff, &t2 );

       printf("%.6lf, ", t2);
    
       remap_test = (real *)malloc(bsize*sizeof(real));
       error = clEnqueueReadBuffer(queue, remap_buffer, CL_TRUE, 0, bsize*sizeof(real), remap_test, 0, NULL, NULL);
       if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
       clReleaseMemObject(remap_buffer);

       icount = 0;
       for(i = 0; i < bsize; i++) {
          if(fabs(remap_gold[i] - remap_test[i]) > 1.0e-6){
             printf("Check failed for remap1 GPU index %d remap value %lf gold standard %lf\n", i, remap_test[i], remap_gold[i]);
             icount++;
          }
          if (icount > 20) exit(0);
       }
       free(remap_test);

       clReleaseMemObject(a_buffer);
       clReleaseMemObject(b_buffer);
       clReleaseMemObject(v_buffer);
    } else {
       printf("not_run   ");
    }
    
    free(remap_gold);
    free(arr_a);
    free(arr_b);
    free(original_values);
    free(arr_old_in);
    free(arr_new_in);
} 

real* remap_bruteforce( struct rcell *x, real *original_values, struct rcell *x_new, int ncells, int new_cells, double min_diff, double max_diff, double min_val, double max_a ) {
    
    int ic,jc;
    int num;
    
    real* remap = (real*)malloc((new_cells)*sizeof(real));
    
    for (ic = 0; ic < new_cells; ic++){
        remap[ic]=0.0;
        //printf("x new %d low %lf high %lf\n",ic,x_new[ic].low,x_new[ic].high);
        for (jc = 0; jc < ncells; jc++){
            //printf("x original %d low %lf high %lf\n",jc,x[jc].low,x[jc].high);
            real overlap_area = MIN(x[jc].high,x_new[ic].high)-MAX(x[jc].low,x_new[ic].low);
            //printf("overlap is %lf\n",overlap_area);
            if (overlap_area > 0) {
                real original_area = x[jc].high - x[jc].low;
                real mapped_value = overlap_area/original_area * original_values[jc];

                remap[ic] += mapped_value;
            }
        }
    }

    //for (ic = 0; ic < new_cells; ic++){
    //   printf("%d: remap %lf\n",ic,remap[ic]);
    //}

    return remap;
}

real* remap_kdTree( struct rcell *x, real *original_values, struct rcell *x_new, int ncells, int new_cells, double min_diff, double max_diff, double min_val, double max_a ) {
    
    int ic,jc;
    int num;
    int index_list[20];
    TKDTree1d tree;
    
    real* remap = (real*)malloc((new_cells)*sizeof(real));
    
    KDTree_Initialize1d(&tree);
    
    TBounds1d box;
    //printf("\n");
    for (ic = 0; ic < ncells; ic++) {
        box.min.x = x[ic].low;
        box.max.x = x[ic].high;
        //printf("Add box %x min %lf max %lf\n",ic,box.min.x,box.max.x);
        KDTree_AddElement1d(&tree, &box);
    }
    
    for (ic = 0; ic < new_cells; ic++){
        remap[ic]=0.0;
        box.min.x = x_new[ic].low;
        box.max.x = x_new[ic].high;
        //printf("box for %d min %lf max %lf\n",ic,box.min.x,box.max.x);
        KDTree_QueryBoxIntersect1d(&tree, &num, &(index_list[0]), &box);
        for (jc = 0; jc<num; jc++){
            real overlap_area = MIN(x[index_list[jc]].high,x_new[ic].high)-MAX(x[index_list[jc]].low,x_new[ic].low);
            real original_area = x[index_list[jc]].high - x[index_list[jc]].low;
            real mapped_value = overlap_area/original_area * original_values[index_list[jc]];

            remap[ic]+=mapped_value;
        }
    }
    KDTree_Destroy1d(&tree);

    //for (ic = 0; ic < new_cells; ic++){
    //   printf("%d: remap %lf\n",ic,remap[ic]);
    //}

    return remap;
}

real* remap1( struct rcell *arr_a, real* original_values, struct rcell *arr_b, int asize, int bsize, real max_a, real min_val, real min_diff ) {

    int a, b, i;
    int start, end;
    
    int temp_size = (uint)((max_a - min_val)/min_diff);
    real *remap = (real*)malloc(bsize*sizeof(real));
    int* temp = malloc(temp_size*sizeof(int));
    
    /* Create a Hash Table for the first (old) array */
    memset(temp, -1, temp_size*sizeof(int));
    for(a = 0; a < asize; a++) {
       start = (int)((arr_a[a].low+min_val)/min_diff);
       end = (int)((arr_a[a].high+min_val)/min_diff);
       while( start < end ) {
          temp[start] = a;
          start++;
       }
    }
    
    for(b = 0; b < bsize; b++) {
        remap[b] = 0;
        if( (start = (arr_b[b].low - min_val)/min_diff) < temp_size) {
            end = MIN(temp_size, (arr_b[b].high - min_val)/min_diff);
            for(i = start; i < end; i++) {
                if(temp[i] >= 0) {
                    remap[b] += original_values[temp[i]] * 1.0/(arr_a[temp[i]].high-arr_a[temp[i]].low);   //assume state variable is in original_value
                }
            }
        }
    }
    
    free(temp);

    //for (int ic = 0; ic < bsize; ic++){
    //   printf("%d: remap %lf\n",ic,remap[ic]);
    //}

    return remap;
}

real* remap2( real *arr_old_in, real *original_values, real *arr_new_in, int asize, int bsize, real max_a, real max_b, real min_val, real min_diff ) {
    double range, fraction;
    int a, b;

    real* remap = (real*)malloc(bsize*sizeof(real));
    for (int ic = 0; ic < bsize; ic++){
       remap[ic] = 0.0;
    }

    /* convert from cell format to array of reals */
    
    asize += 1; bsize += 1;
    real *sorted_values = (real*)malloc(asize*sizeof(real));

    real *arr_old = hashsort2((uint)asize, arr_old_in, original_values, sorted_values, min_diff, min_val, max_a);

    int *index = hashsort((uint)bsize, arr_new_in, min_diff, min_val, max_b);
    real *arr_new = (real*)malloc(bsize*sizeof(real));
    for (b = 0; b < bsize; b++){
       arr_new[b] = arr_new_in[index[b]];
    }

    //for(b = 0; b < bsize; b++) { //for bsize = asize
    //   printf("%d index %d\n",b,index[b]);
    //}
    
    //printf("\n\n");
    //for(a = 0; a < asize; a++) { //for bsize = asize
    //    printf("%lf  %lf\n", arr_old[a], arr_new[a]);
    //}
    //printf("\n\n");
    
    b = 1;
    for(a = 1; a < asize; a++) {
       while(b < bsize && arr_new[b] <= arr_old[a] ) {
          range = arr_new[b] - MAX(arr_old[a-1], arr_new[b-1]);
          fraction = range/(arr_old[a] - arr_old[a-1]);
          remap[index[b-1]] += fraction * sorted_values[a-1];
          b++;
       }
       range = arr_old[a] - MAX(arr_old[a-1], arr_new[b-1]);
       fraction = range/(arr_old[a]-arr_old[a-1]);
       remap[index[b-1]] += fraction * sorted_values[a-1];
    }

    free(arr_old);
    free(index);
    free(sorted_values);
    free(arr_new);

    //for (int ic = 0; ic < bsize-1; ic++){
    //   printf("%d: remap %lf\n",ic,remap[ic]);
    //}

    return remap;
}

void generateRealCells( int size, struct rcell *ptr, int mindx, int maxdx, real min, real *max ) {
    
    int i, index, front = 0;
    struct rcell swap;
    
    struct timeval tim;                //random seeding
    gettimeofday(&tim, NULL);
    //srand(tim.tv_sec*tim.tv_usec);
    
    srand(seed);
        seed++;
    
    ptr[0].low = min;        //start the array using the minimum value
    /* for each element, add a random value between mindx and maxdx to the previous element's value */
    for(i = 0; i < size-1; i++) {
        ptr[i].high = ptr[i].low + mindx + rand() % (maxdx - mindx + 1);
        ptr[i+1].low = ptr[i].high;
    }
    ptr[size-1].high = ptr[size-1].low + mindx + rand() % (maxdx - mindx + 1);
    
    *max = ptr[size-1].high;            //set the max value to the last element's value
    
    /* Mix up the array by selecting elements from shrinking front portion of array and placing them on back end of array */
    for(i = 0; (i < size) && (size - i != 0) ; i++) {
        index = rand() % (size - i - front) + front;
        swap = ptr[size-i-1];
        ptr[size-i-1] = ptr[index];
        ptr[index] = swap;
    }
}


cl_mem parallelRemap1( cl_mem a_buffer, cl_mem v_buffer, cl_mem b_buffer, uint asize, uint bsize, real max_a, real min_val, real min_diff, double *time ) {
    
    cl_int error = 0;
    
    uint temp_size = (uint)((max_a - min_val)/min_diff);
    
    cl_mem temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, temp_size*sizeof(int), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    size_t global_work_size[1];
    size_t local_work_size[1];
    
    local_work_size[0] = TILE_SIZE;
    global_work_size[0] = ((asize+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
    
    /******************
     * Hash Kernel
     ******************/
    
    error = clSetKernelArg(cHash_kernel, 0, sizeof(real), &min_val);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(cHash_kernel, 1, sizeof(real), &min_diff);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(cHash_kernel, 2, sizeof(cl_uint), &asize);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(cHash_kernel, 3, sizeof(cl_mem), (void*)&a_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(cHash_kernel, 4, sizeof(cl_mem), (void*)&temp_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    global_work_size[0] = ((asize+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];
    
    cl_event hash_kernel_event;
    
    error = clEnqueueNDRangeKernel(queue, cHash_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &hash_kernel_event);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    /*****************
     * Remap Kernel
     *****************/
    
    cl_mem remap_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bsize*sizeof(real), NULL, &error);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    error = clSetKernelArg(remap1_kernel, 0, sizeof(real), &min_val);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 1, sizeof(real), &min_diff);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 2, sizeof(cl_uint), &temp_size);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 3, sizeof(cl_uint), &bsize);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 4, sizeof(cl_mem), (void*)&a_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 5, sizeof(cl_mem), (void*)&v_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 6, sizeof(cl_mem), (void*)&b_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 7, sizeof(cl_mem), (void*)&temp_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    error = clSetKernelArg(remap1_kernel, 8, sizeof(cl_mem), (void*)&remap_buffer);
    if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);
    
    global_work_size[0] = ((bsize+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];

    cl_event remap_event;
    
    error = clEnqueueNDRangeKernel(queue, remap1_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &remap_event);
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
    
    clReleaseMemObject(temp_buffer);

    *time = gpu_time*1.0e-9;
    
    return remap_buffer;

}

