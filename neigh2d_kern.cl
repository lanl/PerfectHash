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

/* neigh2d_kern.cl */

#ifdef HAVE_CL_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real;
#else
typedef float   real;
#endif

struct neighbor2d {
   uint left;
   uint right;
   uint bottom;
   uint top;
};

__kernel void init_kern(
        const uint size,
	__global int *temp) {

	const uint idx = get_global_id(0);

        if (idx >= size) return;

	temp[idx] = -1;
}

__kernel void hash_kern(
	const real min_val,
	const real min_diff,
        const uint length,
	__global const real *arr,
	__global int *temp) {
	
	const uint idx = get_global_id(0);
	
        if(idx >= length) return;

        temp[(uint)((arr[idx]-min_val)/min_diff)] = idx;
}

#define hashval(j,i) hash[(j)*imaxsize+(i)]

__kernel void hash_setup_kern(
      const uint isize,
      const uint mesh_size,
      const uint levmx,
      __global const int  *levtable,
      __global const int  *i,
      __global const int  *j,
      __global const int  *level,
      __global int  *hash
      ) {

   const uint giX = get_global_id(0);

   if (giX >= isize) return;

   int imaxsize = mesh_size*levtable[levmx];

   int lev = level[giX];
   int ii = i[giX];
   int jj = j[giX];

   int levdiff = levmx - lev;

   int iimin =  ii   *levtable[levdiff];
   int iimax = (ii+1)*levtable[levdiff];
   int jjmin =  jj   *levtable[levdiff];
   int jjmax = (jj+1)*levtable[levdiff];

   for (   int jjj = jjmin; jjj < jjmax; jjj++) {
      for (int iii = iimin; iii < iimax; iii++) {
         hashval(jjj, iii) = giX;
      }
   }

}

__kernel void calc_neighbor2d_kern(
      const int isize,
      const uint mesh_size,
      const int levmx,
      __global const int *levtable,
      __global const int *i,
      __global const int *j,
      __global const int *level,
      __global const int *hash,
      __global struct neighbor2d *neigh2d
      ) {

   const uint giX  = get_global_id(0);

   if (giX >= isize) return;

   int imaxsize = mesh_size*levtable[levmx];
   int jmaxsize = mesh_size*levtable[levmx];

   int ii = i[giX];
   int jj = j[giX];
   int lev = level[giX];
   int levmult = levtable[levmx-lev];

   int nlftval = hashval(      jj   *levmult               , max(  ii   *levmult-1, 0         ));
   int nrhtval = hashval(      jj   *levmult               , min( (ii+1)*levmult,   imaxsize-1));
   int nbotval = hashval(max(  jj   *levmult-1, 0)         ,       ii   *levmult               );
   int ntopval = hashval(min( (jj+1)*levmult,   jmaxsize-1),       ii   *levmult               );

   neigh2d[giX].left = nlftval;
   neigh2d[giX].right = nrhtval;
   neigh2d[giX].bottom = nbotval;
   neigh2d[giX].top = ntopval;
}
