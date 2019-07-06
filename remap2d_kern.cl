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

/* remap_kern2d.cl */

#ifdef HAVE_CL_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real;
#else
typedef float   real;
#endif

// Cartesian Coordinate Indexing
#define two_to_the(ishift)       (1u <<(ishift) )
#define four_to_the(ishift)      (1u << ( (ishift)*2 ) )

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
       int ii = i[ic];
       int jj = j[ic];
       int lev = level[ic];
       // If at the maximum level just set the one cell
       if (lev == levmx) {
           hash_table[(jj*i_max)+ii] = ic;
       } else {
           // Set the square block of cells at the finest level
           // to the index number
           int lev_mod = two_to_the(levmx - lev);
           for (int jjj = jj*lev_mod; jjj < (jj+1)*lev_mod; jjj++) {
              for (int iii = ii*lev_mod; iii < (ii+1)*lev_mod; iii++) {
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
   __global const int* mesh_a_i,
   __global const int* mesh_a_j,
   __global const int* mesh_a_level,
   __global const int* mesh_b_i,
   __global const int* mesh_b_j,
   __global const int* mesh_b_level,
   const int ncells_b,
   const int mesh_size,
   const int levmx) {

   const int jc = get_global_id(0);

   uint i_max = mesh_size*two_to_the(levmx);

   if(jc < ncells_b) {
      int ii = mesh_b_i[jc];
      int jj = mesh_b_j[jc];
      int lev = mesh_b_level[jc];
      int lev_mod = two_to_the(levmx - lev);
      real val_sum = 0.0;
      for(int jjj = jj*lev_mod; jjj < (jj+1)*lev_mod; jjj++) {
         for(int iii = ii*lev_mod; iii < (ii+1)*lev_mod; iii++) {
            int ic = hash_table[jjj*i_max+iii];
            val_sum += V_a[ic] / (real)four_to_the(levmx-mesh_a_level[ic]);
         }
      }
      V_remap[jc] += val_sum;
   }

}

