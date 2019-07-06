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

/* sort2d_kern.cl */

#ifdef HAVE_CL_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real;
#define ONE 1.0
#define TWO 2.0
typedef struct {
   long level;
   double xval;
   double yval;
} cell;
#else
typedef float   real;
#define ONE 1.0f
#define TWO 2.0f
typedef struct {
   long level;
   float xval;
   float yval;
} cell;
#endif

#define category cell

inline uint scan_warp_exclusive(__local volatile uint *input, const uint idx, const uint lane);
inline uint scan_warp_inclusive(__local volatile uint *input, const uint idx, const uint lane);
inline uint scan_workgroup_exclusive(
                                     __local uint* tile,
                                     const uint tiX,
                                     const uint lane,
                                     const uint warpID);

__kernel void hash_init_cl(
        const int hsize,
	__global int *hash_table) {

   const int ic = get_global_id(0);
   if(ic < hsize) hash_table[ic] = -1;

}


#define MESH_SIZE mesh_size
//#define MESH_SIZE 256

int powerOfTwo(int n) {
   int val = 1;
   int ic;
   for(ic = 0; ic < n; ic++) {val *= 2;}
   return val;
}

// XXX Make sure MESH_SIZE GETS SET XXX
// Cartesian Coordinate Indexing
#define HASHY_CL (( powerOfTwo(levmx)*MESH_SIZE ))
#define XY_TO_IJ_CL(x) (( (x-(ONE/(TWO*(real)MESH_SIZE*(real)powerOfTwo(mesh[ic].level))))*(real)HASHY_CL ))
#define HASH_MAX_CL (( SQR(HASHY_CL) ))
#define HASH_KEY_CL (( XY_TO_IJ_CL(mesh[ic].xval) + XY_TO_IJ_CL(mesh[ic].yval)*(real)HASHY_CL ))

__kernel void hash_build_cl(
        __global const cell* mesh,
        const int levmx,
        const int size,
	__global int *hash_table,
        const int mesh_size) {
	
	const int ic = get_global_id(0);
        if(ic < size) hash_table[(int)HASH_KEY_CL] = ic;
}

__kernel void scan1(
	const uint isize,
	__global uint *ioffset,
	__local volatile uint *itile,
	__global const int *temp) {
		
	const uint giX = get_global_id(0);
	const uint tiX = get_local_id(0);
	const uint ntX = get_local_size(0);
	const uint group_id = get_group_id(0);

        int temp_val = -1;
        if (giX < isize) temp_val = temp[giX];

        itile[tiX] = temp_val >= 0 ? 1 : 0;
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	for(uint offset = ntX >> 1; offset > 32; offset >>= 1) {
		if(tiX < offset) {
			itile[tiX] += itile[tiX+offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

        if(giX >= isize) return;

    //  Unroll the remainder of the loop as 32 threads must proceed in lockstep.
    if (tiX < 32)
    {  itile[tiX] += itile[tiX+32];
       itile[tiX] += itile[tiX+16];
       itile[tiX] += itile[tiX+8];
       itile[tiX] += itile[tiX+4];
       itile[tiX] += itile[tiX+2];
       itile[tiX] += itile[tiX+1]; }

    if(tiX == 0) {
        ioffset[group_id] = itile[0];
    }
}

inline uint scan_warp_exclusive(__local volatile uint *input, const uint idx, const uint lane) {
    if (lane > 0 ) input[idx] += input[idx - 1];
    if (lane > 1 ) input[idx] += input[idx - 2];
    if (lane > 3 ) input[idx] += input[idx - 4];
    if (lane > 7 ) input[idx] += input[idx - 8];
    if (lane > 15) input[idx] += input[idx - 16];
    
    return (lane > 0) ? input[idx-1] : 0;
}

inline uint scan_warp_inclusive(__local volatile uint *input, const uint idx, const uint lane) {
    if (1) {
       if (lane > 0 ) input[idx] += input[idx - 1];
       if (lane > 1 ) input[idx] += input[idx - 2];
       if (lane > 3 ) input[idx] += input[idx - 4];
       if (lane > 7 ) input[idx] += input[idx - 8];
       if (lane > 15) input[idx] += input[idx - 16];
    
       return input[idx];
    }
}

inline uint scan_workgroup_exclusive(
    __local uint* itile,
    const uint tiX,
    const uint lane,
    const uint warpID) {
    
    // Step 1: scan each warp
    uint val = scan_warp_exclusive(itile, tiX, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 2: Collect per-warp sums
    if (lane == 31) itile[warpID] = itile[tiX];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 3: Use 1st warp to scan per-warp sums
    if (warpID == 0) scan_warp_inclusive(itile, tiX, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 4: Accumulate results from Steps 1 and 3
    if (warpID > 0) val += itile[warpID-1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 6: Write and return the final result
    itile[tiX] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    return val;
}

__kernel void scan2(
    __local uint* itile,
    __global uint* ioffset,
    const uint size) {

    size_t tiX = get_local_id(0);
    const uint gID = get_group_id(0);
    const uint ntX = get_local_size(0);
    
    const uint lane = tiX & 31;
    const uint warpID = tiX >> 5;
    const uint EPT = (size+ntX-1)/ntX; //elements_per_thread;
    
    uint reduceValue = 0;
    
//  #pragma unroll 4
    for(uint i = 0; i < EPT; ++i)
    {
       uint offsetIdx = i * ntX + tiX;

#ifdef IS_NVIDIA
//     if (offsetIdx >= size) return;
#endif
        
       // Step 1: Read ntX elements from global (off-chip) memory to local memory (on-chip)
       uint input = 0;
       if (offsetIdx < size) input = ioffset[offsetIdx];           
       itile[tiX] = input;           
       barrier(CLK_LOCAL_MEM_FENCE);
        
       // Step 2: Perform scan on ntX elements
       uint val = scan_workgroup_exclusive(itile, tiX, lane, warpID);
        
       // Step 3: Propagate reduced result from previous block of ntX elements
       val += reduceValue;
        
       // Step 4: Write out data to global memory
       if (offsetIdx < size) ioffset[offsetIdx] = val;
     
       // Step 5: Choose reduced value for next iteration
       if (tiX == (ntX-1)) itile[tiX] = input + val;
       barrier(CLK_LOCAL_MEM_FENCE);
        
       reduceValue = itile[ntX-1];
       barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void scan3 (
    const int isize,
    __global const uint *ioffset,
    __local uint *itile,
    __global const int *temp,
    __global const cell *arr,
    __global cell *sorted) {
    
    const uint giX = get_global_id(0);
    const uint tiX = get_local_id(0);
    const uint group_id = get_group_id(0);

    const uint lane   = tiX & 31;
    const uint warpid = tiX >> 5;

    // Step 1: load global data into tile
    int temp_val = 0;
    if (giX < isize) temp_val = temp[giX];
    itile[tiX] = 0;
    if (temp_val >= 0) itile[tiX] = 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: scan each warp
    uint val = scan_warp_exclusive(itile, tiX, lane);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Collect per-warp sums
    if (lane == 31) itile[warpid] = itile[tiX];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 4: Use 1st warp to scan per-warp sums
    if (warpid == 0) scan_warp_inclusive(itile, tiX, lane);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 5: Accumulate results from Steps 2 and 4
    if (warpid > 0) val += itile[warpid-1];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (giX >= isize || temp_val < 0) return;

    // Step 6: Write and return the final result
    //itile[tiX] = val;
    //barrier(CLK_LOCAL_MEM_FENCE);

    val += ioffset[group_id];   //index to write to for each thread

    sorted[val]     = arr[temp_val];
}

