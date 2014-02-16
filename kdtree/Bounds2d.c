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

#include "Bounds2d.h"

void Bounds_Copy2d(TBounds2d* src, TBounds2d* dest) {
   assert(src && dest);
   MEMCPY(src, dest, 1, TBounds2d);
}

void Bounds_Infinite2d(TBounds2d* b){
   assert(b);
   b->min.x = POSITIVE_INFINITY;
   b->min.y = POSITIVE_INFINITY;
   b->max.x = NEGATIVE_INFINITY;
   b->max.y = NEGATIVE_INFINITY;
}

void Bounds_AddBounds2d(TBounds2d* b, TBounds2d* add) {
   assert(b && add);
   b->min.x = MIN(b->min.x, add->min.x);
   b->min.y = MIN(b->min.y, add->min.y);
   b->max.x = MAX(b->max.x, add->max.x);
   b->max.y = MAX(b->max.y, add->max.y);
}

void Bounds_AddEpsilon2d(TBounds2d* b, double add) {
   assert(b);
   b->min.x = b->min.x - add;
   b->min.y = b->min.y - add;
   b->max.x = b->max.x + add;
   b->max.y = b->max.y + add;
}

boolean Bounds_IsOverlappingBounds2d(TBounds2d* b, TBounds2d* tst) {
   assert(b && tst);
   if((tst->max.x < b->min.x) || (tst->min.x > b->max.x))
      return(false);
   if((tst->max.y < b->min.y) || (tst->min.y > b->max.y))
      return(false);
   return(true);
}

double Bounds_WidthAxis2d(TBounds2d* b, unsigned long axis)
{
   double width;
   
   assert(b);
   if(axis == XAXIS)
      width = b->max.x - b->min.x;
   else if(axis == YAXIS)
      width = b->max.y - b->min.y;
   else
      assert(NULL);
   return(width);
}

double Bounds_CenterAxis2d(TBounds2d* b, unsigned long axis)
{
   double center;
   
   assert(b);
   if(axis == XAXIS)
      center = (b->min.x + b->max.x) * 0.5;
   else if(axis == YAXIS)
      center = (b->min.y + b->max.y) * 0.5;
   else
      assert(NULL);
   return(center);
}
