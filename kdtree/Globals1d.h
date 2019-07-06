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

#ifndef _Globals1d_
#define _Globals1d_

#ifdef __cplusplus
extern "C"
{
#endif
   
//#define NDEBUG 1
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef ENTITY_COINCIDENCE_TOLERANCE
#define ENTITY_COINCIDENCE_TOLERANCE      ((double)1.0E-5)

#define KDTREE_ELEMENT_BLOCKING_SIZE      ((long)1024)
#endif

#ifndef POSITIVE_INFINITY
#define POSITIVE_INFINITY (+1.0E+64)
#define NEGATIVE_INFINITY (-1.0E+64)
#endif

#define XAXIS ((unsigned long)0)

typedef struct {
   double x;
} TVector1d;

#ifndef _BOOL
#define _BOOL
typedef unsigned char boolean;
#define true  ((boolean)1)
#define false ((boolean)0)
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef SWAP
#define SWAP(a,b,t) {t h; h = a; a = b; b = h; }
#endif

#ifndef MALLOC
#define MALLOC(n,t) ((t*)(malloc(n * sizeof(t))))
#define REALLOC(p,n,t) ((t*)(realloc((void*)p, n * sizeof(t))))
#define FREE(p) { if (p) free(p); }
#define MEMCPY(s,d,n,t) {memcpy((void*)d, (void*)s, n * sizeof(t)); }
#endif

#ifdef __cplusplus
}
#endif
   
#endif
