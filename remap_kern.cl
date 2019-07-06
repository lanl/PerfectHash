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

/* remap_kern.cl */

#ifdef HAVE_CL_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real;
#else
typedef float   real;
#endif

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif

struct rcell {
    real low;
    real high;
};

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


/* Remap Kernels */

__kernel void cellHash_kern(
    const real min_val,
    const real min_diff,
    const uint length,
    __global const struct rcell *arr,
    __global int *temp) {
    
    const uint idx = get_global_id(0);
    
    if( idx < length ) {
    
        uint start = (int)((arr[idx].low+min_val)/min_diff);
        uint end = (int)((arr[idx].high+min_val)/min_diff);
    
        while( start < end ) {
            temp[start] = idx;
            start++;
        }
    }

}

__kernel void remap1_kern(
    const real min_val,
    const real mindx,
    const uint hash_size,
    const uint bsize,
    __global struct rcell *arr_a,
    __global real *arr_v,
    __global struct rcell *arr_b,
    __global int *hash,
    __global real *remap) {
    
    const uint idx = get_global_id(0);
    if( idx < bsize ) {
    
        uint start = (arr_b[idx].low - min_val)/mindx;
        uint end = (arr_b[idx].high - min_val)/mindx;
    
        if(start > hash_size - 1) { remap[idx] = 0.0; return; }
        if(end > hash_size) end = hash_size;
    
        remap[idx] = 0.;
        for( uint i = start; i < end; i++ ) {
            if(hash[i] >= 0) {
                remap[idx] += arr_v[hash[i]] * 1./(arr_a[hash[i]].high - arr_a[hash[i]].low);   //assume state variable value of 1 in each original cell
            }
        }
    }
}

