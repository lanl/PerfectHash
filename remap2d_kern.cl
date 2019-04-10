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

/* remap_kern2d.cl */

#ifdef HAVE_CL_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real;
typedef struct {
    double *x;
    double *y;
    int *level;
} cell;
#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#else
typedef float   real;
typedef struct {
    float *x;
    float *y;
    int *level;
} cell;
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#endif

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif

// Cartesian Coordinate Indexing
#define two_to_the(ishift)       (1u <<(ishift) )
#define four_to_the(ishift)      (1u << ( (ishift)*2 ) )

#define HASHY (( two_to_the(levmx)*mesh_size ))
#define XY_TO_IJ(x,lev) (( (x-(ONE/(TWO*(real)mesh_size*(real)two_to_the(lev))))*(real)HASHY ))
#define SQ(x) ((x)*(x))
#define HASH_MAX (( SQ(HASHY) ))
#define HASH_KEY(x,y,lev) (( XY_TO_IJ(x,lev) + XY_TO_IJ(y,lev)*(real)HASHY ))


/* Remap Kernels */
__kernel void remap_hash_creation_kern(
   __global int* hash_table,
   __global const int* i,
   __global const int* j,
   __global const int* level,
   const int ncells_a,
   const int mesh_size,
   const int levmx) {

   const int ic = get_global_id(0);

   uint i_max = mesh_size*two_to_the(levmx);

   if(ic < ncells_a) {
       uint lev = level[ic];
       uint ii = i[ic];
       uint jj = j[ic];
       // If at the maximum level just set the one cell
       if (lev == levmx) {
           hash_table[(jj*i_max)+ii] = ic;
       } else {
           // Set the square block of cells at the finest level
           // to the index number
           uint lev_mod = two_to_the(levmx - lev);
           for (uint jjj = jj*lev_mod; jjj < (jj+1)*lev_mod; jjj++) {
              for (uint iii = ii*lev_mod; iii < (ii+1)*lev_mod; iii++) {
                  hash_table[(jjj*i_max)+iii] = ic;
              }
           }
       }
   }

}


__kernel void remap_hash_retrieval_kern(
   __global real* V_remap,
   __global const real* V_a,
   __global const int* hash_table,
   __global const real* mesh_a_x,
   __global const real* mesh_a_y,
   __global const int* mesh_a_level,
   __global const real* mesh_b_x,
   __global const real* mesh_b_y,
   __global const int* mesh_b_level,
   const int ncells_b,
   const int mesh_size,
   const int levmx) {

   const int ic = get_global_id(0);

   if(ic < ncells_b) {
      V_remap[ic] = ZERO;
      int yc, xc;
      int cell_remap;
      int hic = (int) HASH_KEY(mesh_b_x[ic], mesh_b_y[ic], mesh_b_level[ic]);
      int hwh = two_to_the(levmx - mesh_b_level[ic]);
      for(yc = 0; yc < hwh; yc++) {
         for(xc = 0; xc < hwh; xc++) {
            cell_remap = hash_table[hic];
            V_remap[ic] += (V_a[cell_remap] / (real)four_to_the(levmx-mesh_a_level[cell_remap]));
            hic++;
         }
         hic = hic - hwh + HASHY;
      }
   }

}

