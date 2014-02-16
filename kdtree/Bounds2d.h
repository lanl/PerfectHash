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

#ifndef _Bounds2d_
#define _Bounds2d_

#ifdef __cplusplus
extern "C"
{
#endif
   
#include "Globals2d.h"

typedef struct {
   TVector2d min, max;
} TBounds2d;

extern void Bounds_Copy2d(TBounds2d* src, TBounds2d* dest);
extern void Bounds_Infinite2d(TBounds2d* b);
extern void Bounds_AddBounds2d(TBounds2d* b, TBounds2d* add);
extern void Bounds_AddEpsilon2d(TBounds2d* b, double add);
extern boolean Bounds_IsOverlappingBounds2d(TBounds2d* b, TBounds2d* tst);
extern double Bounds_WidthAxis2d(TBounds2d* b, unsigned long axis);
extern double Bounds_CenterAxis2d(TBounds2d* b, unsigned long axis);

#ifdef __cplusplus
}
#endif

#endif
