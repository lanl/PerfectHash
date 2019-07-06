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

#include "Bounds1d.h"

void Bounds_Copy1d(TBounds1d* src, TBounds1d* dest) {
   assert(src && dest);
   MEMCPY(src, dest, 1, TBounds1d);
}

void Bounds_Infinite1d(TBounds1d* b){
   assert(b);
   b->min.x = POSITIVE_INFINITY;
   b->max.x = NEGATIVE_INFINITY;
}

void Bounds_AddBounds1d(TBounds1d* b, TBounds1d* add) {
   assert(b && add);
   b->min.x = MIN(b->min.x, add->min.x);
   b->max.x = MAX(b->max.x, add->max.x);
}

void Bounds_AddEpsilon1d(TBounds1d* b, double add) {
   assert(b);
   b->min.x = b->min.x - add;
   b->max.x = b->max.x + add;
}

boolean Bounds_IsOverlappingBounds1d(TBounds1d* b, TBounds1d* tst) {
   assert(b && tst);
   if((tst->max.x < b->min.x) || (tst->min.x > b->max.x))
      return(false);
   return(true);
}

double Bounds_WidthAxis1d(TBounds1d* b, unsigned long axis)
{
   double width;
   
   assert(b);
   if(axis == XAXIS)
      width = b->max.x - b->min.x;
   else
      assert(NULL);
   return(width);
}

double Bounds_CenterAxis1d(TBounds1d* b, unsigned long axis)
{
   double center;
   
   assert(b);
   if(axis == XAXIS)
      center = (b->min.x + b->max.x) * 0.5;
   else
      assert(NULL);
   return(center);
}
