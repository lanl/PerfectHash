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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
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
#define EPS 1.0e-8
#else
typedef float real;
typedef cl_float cl_real;
typedef cl_float4 cl_real4;
#define EPS 1.0e-5
#endif

#define TILE_SIZE 256
#define dataval(x,y) data[(x)+((y)*xstride)]

cl_context context;
cl_command_queue queue;
cl_program program;
int is_nvidia=0;

double random_normal_dist(void);
double *interpolate_bruteforce(int isize, int xstride, double *density_axis, double *temp_axis,
      double *density_array, double *temp_array, double *data);
double *interpolate_bisection(int isize, int xstride, double *density_axis, double *temp_axis,
      double *density_array, double *temp_array, double *data);
int bisection(double *axis, int axis_size, double value);
double *interpolate_hashcpu(int isize, int xstride, double *density_axis, double *temp_axis,
      double *density_array, double *temp_array, double *data);
cl_mem interpolate_hashgpu(int isize, int xstride, cl_mem density_axis_buffer, cl_mem temp_axis_buffer,
      cl_mem density_array_buffer, cl_mem temp_array_buffer, cl_mem data_buffer, double *time);

int main(int argc, char *argv[])
{
   cl_int error;

   GPUInit(&context, &queue, &is_nvidia, &program, "table_kern.cl");

   interpolate_kernel = clCreateKernel(program, "interpolate_kernel", &error);
   if (error != CL_SUCCESS) printf("Error is %d at line %d\n",error,__LINE__);


#include "table.data"

   int i;

   double temp, density;

   double density_increment = (density_axis[50]-density_axis[0])/50.0;
   double temp_increment = (temp_axis[22]-temp_axis[0])/22.0;

   double density_avg = (density_axis[50]+density_axis[0])/2.0;
   double temp_avg = (temp_axis[22]+temp_axis[0])/2.0;

   double density_stddev = (density_axis[50]-density_axis[0])/6.0;
   double temp_stddev = (temp_axis[22]-temp_axis[0])/6.0;

   for (i=1; i<51; i++){
      density_axis[i] = density_axis[0]+(double)i*density_increment;
   }

   for (i=1; i<23; i++){
      temp_axis[i] = temp_axis[0]+(double)i*temp_increment;
   }

   cl_int ierror;
   int data_size = 1173;
   int density_axis_size = 51;
   int temp_axis_size = 23;

   real *data_real = (real *)malloc(data_size*sizeof(real));
   for (i=0; i<data_size; i++) { data_real[i] = (real)data[i]; }
   cl_mem data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size*sizeof(real), NULL, &ierror);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clEnqueueWriteBuffer(queue, data_buffer, CL_TRUE, 0, data_size*sizeof(real), data_real, 0, NULL, NULL);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   free(data_real);

   real *density_axis_real = (real *)malloc(density_axis_size*sizeof(real));
   for (i=0; i<density_axis_size; i++) { density_axis_real[i] = (real)density_axis[i]; }
   cl_mem density_axis_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, density_axis_size*sizeof(real), NULL, &ierror);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clEnqueueWriteBuffer(queue, density_axis_buffer, CL_TRUE, 0, density_axis_size*sizeof(real), density_axis_real, 0, NULL, NULL);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   free(density_axis_real);

   real *temp_axis_real = (real *)malloc(temp_axis_size*sizeof(real));
   for (i=0; i<temp_axis_size; i++) { temp_axis_real[i] = (real)temp_axis[i]; }
   cl_mem temp_axis_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, temp_axis_size*sizeof(real), NULL, &ierror);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clEnqueueWriteBuffer(queue, temp_axis_buffer, CL_TRUE, 0, temp_axis_size*sizeof(real), temp_axis_real, 0, NULL, NULL);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   free(temp_axis_real);

   printf("\n    Table Interpolate Performance Results\n\n");

   printf("Size,   \tBrute,    \tBisection \tHash CPU, \tHash GPU\n");

   double *value_gold, *value_test;

   int isize;
   for( isize = 64; isize <= 50000000; isize*=2 ) {
      printf("%d\t",isize);

      // Initialize look-up data
      double *temp_array=(double *)malloc(isize*sizeof(double));
      double *density_array=(double *)malloc(isize*sizeof(double));

      for (i = 0; i<isize; i++){
         temp_array[i]    = random_normal_dist()*temp_stddev    + temp_avg;
         density_array[i] = random_normal_dist()*density_stddev + density_avg;
      }

      int xstride = 51;
      struct timeval timer;
      double t1, t2;

      // call data table interpolation routine
      gettimeofday(&timer, NULL);
      t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
      value_gold = interpolate_bruteforce(isize, xstride, density_axis, temp_axis,
         density_array, temp_array, data);
      gettimeofday(&timer, NULL);
      t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
      printf("\t%.6lf,", t2 - t1);

      gettimeofday(&timer, NULL);
      t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
      value_test = interpolate_bisection(isize, xstride, density_axis, temp_axis,
         density_array, temp_array, data);
      gettimeofday(&timer, NULL);
      t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
      printf("\t%.6lf,", t2 - t1);

      for (i= 0; i<isize; i++){
         if (value_test[i] != value_gold[i]){
            printf("Warning %d does not match -- test %lf gold %lf\n",i,value_test[i],value_gold[i]);
         }
      }

      free(value_test);

      gettimeofday(&timer, NULL);
      t1 = timer.tv_sec+(timer.tv_usec/1000000.0);
      value_test = interpolate_hashcpu(isize, xstride, density_axis, temp_axis,
         density_array, temp_array, data);
      gettimeofday(&timer, NULL);
      t2 = timer.tv_sec+(timer.tv_usec/1000000.0);
      printf("\t%.6lf,", t2 - t1);

      for (i= 0; i<isize; i++){
         if (value_test[i] != value_gold[i]){
            printf("Warning %d does not match -- test %lf gold %lf\n",i,value_test[i],value_gold[i]);
         }
      }

      free(value_test);

      real *density_array_real = (real *)malloc(isize*sizeof(real));
      for (i=0; i<isize; i++) { density_array_real[i] = (real)density_array[i]; }
      cl_mem density_array_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, isize*sizeof(real), NULL, &ierror);
      if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
      ierror = clEnqueueWriteBuffer(queue, density_array_buffer, CL_TRUE, 0, isize*sizeof(real), density_array_real, 0, NULL, NULL);
      if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
      free(density_array_real);

      real *temp_array_real = (real *)malloc(isize*sizeof(real));
      for (i=0; i<isize; i++) { temp_array_real[i] = (real)temp_array[i]; }
      cl_mem temp_array_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, isize*sizeof(real), NULL, &ierror);
      if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
      ierror = clEnqueueWriteBuffer(queue, temp_array_buffer, CL_TRUE, 0, isize*sizeof(real), temp_array_real, 0, NULL, NULL);
      if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
      free(temp_array_real);

      cl_mem value_buffer = interpolate_hashgpu(isize, xstride, density_axis_buffer, temp_axis_buffer,
         density_array_buffer, temp_array_buffer, data_buffer, &t2);
      printf("\t%.6lf,", t2);

      clReleaseMemObject(density_array_buffer);
      clReleaseMemObject(temp_array_buffer);

      real *value_array_real = (real *)malloc(isize*sizeof(real));
      
      ierror = clEnqueueReadBuffer(queue, value_buffer, CL_TRUE, 0, isize*sizeof(real), value_array_real, 0, NULL, NULL);
      if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);

      clReleaseMemObject(value_buffer);
    
      value_test = (double *)malloc(isize*sizeof(double));
      for (i=0; i<isize; i++) { value_test[i] = (double)value_array_real[i]; }
    
      for (i= 0; i<isize; i++){
         if (fabs(value_test[i] - value_gold[i]) > EPS ){
            printf("Warning %d does not match -- test %lf gold %lf\n",i,value_test[i],value_gold[i]);
         }
      }

      free(value_test);

      printf("\n");

      free(value_gold);
   }

   clReleaseMemObject(data_buffer);
   clReleaseMemObject(density_axis_buffer);
   clReleaseMemObject(temp_axis_buffer);
}


double random_normal_dist(void)
{
    double x1, x2, x3, result;

    x1 = 2.0*drand48() - 1.0;
    x2 = 2.0*drand48() - 1.0;
    x3 = 2.0*drand48() - 1.0;
    result = x1 + x2 + x3;

    return(result);
}

double *interpolate_bruteforce(int isize, int xstride, double *density_axis, double *temp_axis,
      double *density_array, double *temp_array, double *data)
{
   int i;

   double *value_array=(double *)malloc(isize*sizeof(double));

   for (i = 0; i<isize; i++){
      int temp_slot, density_slot;

      for (temp_slot=0; temp_slot<21 && temp_array[i] > temp_axis[temp_slot+1]; temp_slot++);
      for (density_slot=0; density_slot<49 && density_array[i] > density_axis[density_slot+1]; density_slot++);

      double xfrac = (density_array[i]-density_axis[density_slot])/(density_axis[density_slot+1]-density_axis[density_slot]);
      double yfrac = (temp_array[i]-temp_axis[temp_slot])/(temp_axis[temp_slot+1]-temp_axis[temp_slot]);
      value_array[i] =      xfrac *     yfrac *dataval(density_slot+1,temp_slot+1) 
                     + (1.0-xfrac)*     yfrac *dataval(density_slot,  temp_slot+1)
                     +      xfrac *(1.0-yfrac)*dataval(density_slot+1,temp_slot)
                     + (1.0-xfrac)*(1.0-yfrac)*dataval(density_slot,  temp_slot);

   }

   return(value_array);
}

double *interpolate_bisection(int isize, int xstride, double *density_axis, double *temp_axis,
      double *density_array, double *temp_array, double *data)
{
   int i;

   double *value_array=(double *)malloc(isize*sizeof(double));

   for (i = 0; i<isize; i++){
      int temp_slot = bisection(temp_axis, 21, temp_array[i]);
      int density_slot = bisection(density_axis, 49, density_array[i]);

      double xfrac = (density_array[i]-density_axis[density_slot])/(density_axis[density_slot+1]-density_axis[density_slot]);
      double yfrac = (temp_array[i]-temp_axis[temp_slot])/(temp_axis[temp_slot+1]-temp_axis[temp_slot]);
      value_array[i] =      xfrac *     yfrac *dataval(density_slot+1,temp_slot+1) 
                     + (1.0-xfrac)*     yfrac *dataval(density_slot,  temp_slot+1)
                     +      xfrac *(1.0-yfrac)*dataval(density_slot+1,temp_slot)
                     + (1.0-xfrac)*(1.0-yfrac)*dataval(density_slot,  temp_slot);
   }

   return(value_array);
}

int bisection(double *axis, int axis_size, double value)
{
   int ibot = 0;
   int itop = axis_size+1;

   while (itop - ibot > 1){
      int imid = (itop + ibot) /2;
      if ( value >= axis[imid] ) 
         ibot = imid;
      else
         itop = imid;
   }
   return(ibot);
}

double *interpolate_hashcpu(int isize, int xstride, double *density_axis, double *temp_axis,
      double *density_array, double *temp_array, double *data)
{
   int i;
   // Computes a constant increment for each axis data look-up
   double density_increment = (density_axis[50]-density_axis[0])/50.0;
   double temp_increment = (temp_axis[22]-temp_axis[0])/22.0;

   double *value_array=(double *)malloc(isize*sizeof(double));

   for (i = 0; i<isize; i++){
      // Determine the interval for interpolation and the fraction in the interval
      int temp_slot = (temp_array[i]-temp_axis[0])/temp_increment;
      int density_slot = (density_array[i]-density_axis[0])/density_increment;

      double xfrac = (density_array[i]-density_axis[density_slot])/(density_axis[density_slot+1]-density_axis[density_slot]);
      double yfrac = (temp_array[i]-temp_axis[temp_slot])/(temp_axis[temp_slot+1]-temp_axis[temp_slot]);
      // Bi-linear interpolation
      value_array[i] =      xfrac *     yfrac *dataval(density_slot+1,temp_slot+1) 
                     + (1.0-xfrac)*     yfrac *dataval(density_slot,  temp_slot+1)
                     +      xfrac *(1.0-yfrac)*dataval(density_slot+1,temp_slot)
                     + (1.0-xfrac)*(1.0-yfrac)*dataval(density_slot,  temp_slot);
   }

   return(value_array);
}

cl_mem interpolate_hashgpu(int isize, int xstride, cl_mem density_axis_buffer, cl_mem temp_axis_buffer,
      cl_mem density_array_buffer, cl_mem temp_array_buffer, cl_mem data_buffer, double *time)
{
   int i;
   cl_int ierror;

   *time = 0.0;

   int data_size = 1173;
   int density_axis_size = 51;
   int temp_axis_size = 23;

   cl_mem value_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, isize*sizeof(real), NULL, &ierror);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);

   ierror = clSetKernelArg(interpolate_kernel, 0, sizeof(cl_uint), &isize);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 1, sizeof(cl_uint), &density_axis_size);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 2, sizeof(cl_uint), &temp_axis_size);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 3, sizeof(cl_uint), &data_size);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 4, sizeof(cl_mem), (void*)&density_axis_buffer);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 5, sizeof(cl_mem), (void*)&temp_axis_buffer);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 6, sizeof(cl_mem), (void*)&data_buffer);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 7, density_axis_size*sizeof(cl_real), NULL);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 8, temp_axis_size*sizeof(cl_real), NULL);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 9, data_size*sizeof(cl_real), NULL);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 10, sizeof(cl_mem), (void*)&density_array_buffer);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 11, sizeof(cl_mem), (void*)&temp_array_buffer);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);
   ierror = clSetKernelArg(interpolate_kernel, 12, sizeof(cl_mem), (void*)&value_buffer);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);

   size_t local_work_size[1];
   size_t global_work_size[1];

   local_work_size[0] = TILE_SIZE;
   global_work_size[0] = ((isize+local_work_size[0]-1)/local_work_size[0])*local_work_size[0];

   cl_event interpolate_event;

   ierror = clEnqueueNDRangeKernel(queue, interpolate_kernel, 1, 0, global_work_size, local_work_size, 0, NULL, &interpolate_event);
   if (ierror != CL_SUCCESS) printf("Error is %d at line %d\n",ierror,__LINE__);

   long gpu_time_start, gpu_time_end;

   clWaitForEvents(1,&interpolate_event);
   clGetEventProfilingInfo(interpolate_event, CL_PROFILING_COMMAND_START, sizeof(gpu_time_start), &gpu_time_start, NULL);
   clGetEventProfilingInfo(interpolate_event, CL_PROFILING_COMMAND_END, sizeof(gpu_time_end), &gpu_time_end, NULL);
   long gpu_time = gpu_time_end - gpu_time_start;
   clReleaseEvent(interpolate_event);

   *time = (double)gpu_time*1.0e-9;

   return(value_buffer);
}

