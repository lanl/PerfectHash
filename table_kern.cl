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

/* table_kern.cl */

#ifdef HAVE_CL_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real;
#else
typedef float   real;
#endif

#define dataval(x,y) data[(x)+((y)*xstride)]

__kernel void interpolate_kernel(
   const uint isize,
   const uint xaxis_size,
   const uint yaxis_size,
   const uint data_size,
   __global const real *xaxis_buffer,
   __global const real *yaxis_buffer,
   __global const real *data_buffer,
   __local        real *xaxis,
   __local        real *yaxis,
   __local        real *data,
   __global const real *x_array,
   __global const real *y_array,
   __global       real *value
   )
{
   const uint tid = get_local_id(0);
   const uint wgs = get_local_size(0);
   const uint gid = get_global_id(0);

   // Loads the axis data values
   if (tid < xaxis_size) xaxis[tid]=xaxis_buffer[tid];
   if (tid < yaxis_size) yaxis[tid]=yaxis_buffer[tid];

   // Loads the data table
   for (uint wid = tid; wid<data_size; wid+=wgs){
      data[wid] = data_buffer[wid];
   }
   // Need to synchronize before table queries
   barrier(CLK_LOCAL_MEM_FENCE);

   // Computes a constant increment for each axis data look-up
   real x_increment = (xaxis[50]-xaxis[0])/50.0;
   real y_increment = (yaxis[22]-yaxis[0])/22.0;

   int xstride = 51;

   if (gid < isize) {
      // Loads the next data value
      real xdata = x_array[gid];
      real ydata = y_array[gid];

      // Determine the interval for interpolation and the fraction in the interval
      int islot = (int)((xdata-xaxis[0])/x_increment);
      int jslot = (int)((ydata-yaxis[0])/y_increment);
      real xfrac = (xdata-xaxis[islot])/(xaxis[islot+1]-xaxis[islot]);
      real yfrac = (ydata-yaxis[jslot])/(yaxis[jslot+1]-yaxis[jslot]);

      // Bi-linear interpolation
      value[gid] =      xfrac *     yfrac *dataval(islot+1,jslot+1)
                 + (1.0-xfrac)*     yfrac *dataval(islot,  jslot+1)
                 +      xfrac *(1.0-yfrac)*dataval(islot+1,jslot)
                 + (1.0-xfrac)*(1.0-yfrac)*dataval(islot,  jslot);
   }
}

