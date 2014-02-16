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
 * 
 *  Implements a 2-dimensional k-D tree. One begins to use the k-D tree by
 *  adding the bounding box of geometric "elements" to the tree structure
 *  through a call to "KDTreeAddElement". Every element should be of the same
 *  type, but could be a single point, a line segment, triangles, etc. Once
 *  all the element bounding boxes have been added, the user of the structure
 *  may make queries against the tree. The actual tree is constructed lazily
 *  when an actual query occurs on the structure.
 *
 *  This version only has one query -- intersection of a box with the elements
 *  and a set of "candidate" elements are returned. The candidates are identified
 *  by an index number (0, ...) signifying the order in which the element was
 *  added to the tree. It is up to the calling code to do additional processing
 *  based on the type of element being used to determine "real" intersections.
 *
 *  The process of actually building the tree takes "n log n" time. Queries 
 *  take "log n" time.
 *
 */

#ifndef _KDTree1d_
#define _KDTree1d_

#ifdef __cplusplus
extern "C"
{
#endif
  
#include "Globals1d.h"
#include "Bounds1d.h"
   
#define LEFT_HALF   0
#define RIGHT_HALF  1
#define BOTTOM_HALF 0
#define TOP_HALF    1   

typedef struct {
   TBounds1d extent;
   int elements_num, elements_allocated;
   TBounds1d* elements;
   boolean tree_built;
   int tree_size;
   TBounds1d* tree_safety_boxes;
   int * tree_link;
} TKDTree1d;

extern void KDTree_Initialize1d(TKDTree1d *t);
extern void KDTree_Destroy1d(TKDTree1d* t);
extern void KDTree_AddElement1d(TKDTree1d* t, TBounds1d* add);
extern void KDTree_CreateTree1d(TKDTree1d* t);
extern void KDTree_QueryBoxIntersect1d(TKDTree1d* t,
                                     int* result_num, int* result_indicies,
                                     TBounds1d* box);
   
#ifdef __cplusplus
}
#endif

#endif
