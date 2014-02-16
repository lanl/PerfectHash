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

#include <math.h>
#include "KDTree1d.h"

static void median_sort1d(TKDTree1d* t,
                        int cut_direction, int k, int num, int* idx)
{
   int left, mid, right, a, i, j;
   
   for (left = 0, right = num - 1; (right - left) > 1;) {
      mid = (left + right) / 2;
      SWAP(idx[mid], idx[left + 1], int);
      if(Bounds_CenterAxis1d(&(t->elements[idx[left + 1]]), cut_direction) >
         Bounds_CenterAxis1d(&(t->elements[idx[right]]), cut_direction))
         SWAP(idx[left + 1], idx[right], int);
      if(Bounds_CenterAxis1d(&(t->elements[idx[left]]), cut_direction) >
         Bounds_CenterAxis1d(&(t->elements[idx[right]]), cut_direction))
         SWAP(idx[left], idx[right], int);
      if(Bounds_CenterAxis1d(&(t->elements[idx[left + 1]]), cut_direction) >
         Bounds_CenterAxis1d(&(t->elements[idx[left]]), cut_direction))
         SWAP(idx[left + 1], idx[left], int);
      a = idx[left];
      i = left + 1;
      j = right;
      while (1) {
         for (i++;
              Bounds_CenterAxis1d(&(t->elements[idx[i]]), cut_direction) <
                Bounds_CenterAxis1d(&(t->elements[a]), cut_direction);
              i++);
         for (j--;
              Bounds_CenterAxis1d(&(t->elements[idx[j]]), cut_direction) >
              Bounds_CenterAxis1d(&(t->elements[a]), cut_direction);
              j--);
         if(j < i)
            break;
         SWAP(idx[i], idx[j], int);
      }
      idx[left] = idx[j];
      idx[j] = a;
      if(j >= k)
         right = j - 1;
      if(j <= k)
         left = i;
   }
   if(((right - left) ==1) &&
      (Bounds_CenterAxis1d(&(t->elements[idx[right]]), cut_direction) <
       Bounds_CenterAxis1d(&(t->elements[idx[left]]), cut_direction)))
      SWAP(idx[right], idx[left], int);
}

void KDTree_Initialize1d(TKDTree1d* t)
{
   assert(t);
   /* Flush the overall tree extent */
   Bounds_Infinite1d(&(t->extent));
   /* Allocate the initial memory for tree elements */
   t->elements_num = 0;
   t->elements_allocated = KDTREE_ELEMENT_BLOCKING_SIZE;
   t->elements = MALLOC(t->elements_allocated, TBounds1d);
   assert(t->elements);
   /* Start without a built tree */
   t->tree_built = false;
   t->tree_size = 0;
   t->tree_safety_boxes = NULL;
   t->tree_link = NULL;
}

void KDTree_Destroy1d(TKDTree1d* t)
{
   assert(t);
   /* Flush the overall tree extent */
   Bounds_Infinite1d(&(t->extent));
   /* Destroy the element list */
   t->elements_num = 0;
   t->elements_allocated = 0;
   FREE(t->elements);
   t->elements = NULL;
   /* Destroy the actual tree */
   t->tree_built = false;
   t->tree_size = 0;
   FREE(t->tree_safety_boxes);
   t->tree_safety_boxes = NULL;
   FREE(t->tree_link);
   t->tree_link = NULL;
}



void KDTree_AddElement1d(TKDTree1d* t, TBounds1d* add)
{
   assert(t && add);
   /* Destroy the current tree if it is built */
   if(t->tree_built) {
      t->tree_built = false;
      t->tree_size = 0;
      FREE(t->tree_safety_boxes);
      t->tree_safety_boxes = NULL;
      FREE(t->tree_link);
      t->tree_link = NULL;
   }
   /* Expand the element array if necessary */
   if(t->elements_num == t->elements_allocated) {
      t->elements_allocated += KDTREE_ELEMENT_BLOCKING_SIZE;
      t->elements = REALLOC(t->elements, t->elements_allocated, TBounds1d);
      assert(t->elements);
   }
   /* Add the new element to the overall extent and the element list */
   Bounds_AddBounds1d(&(t->extent), add);
   Bounds_Copy1d(add, &(t->elements[t->elements_num]));
   t->elements_num++;
}

void KDTree_CreateTree1d(TKDTree1d* t)
{
   int i, next_node, stack_ptr, min, mid, max, parent, cut_direction;
   double width, max_width;
   int* stack;
   int* idx;
   
   assert(t);
   /* If the tree is already built, we don't have to do anything */
   if(t->tree_built)
      return;
   /* If there are no elements in the tree, we don't have to do anything */
   if(t->elements_num > 0) {
      /* Allocate the k-D tree memory */
      t->tree_size = 2 * t->elements_num;
      t->tree_safety_boxes = MALLOC(t->tree_size, TBounds1d);
      t->tree_link = MALLOC(t->tree_size, int);
      /* Create and initialize temporary arrays */
      next_node = 0;
      stack_ptr = 0;
      stack = MALLOC(3 * t->tree_size, int);
      idx = MALLOC(t->elements_num, int);
      for (i = 0; i <  t->elements_num; i++) {
         idx[i] = i;
      }
      /* Setup the root node of the tree and put it on the stack */
      stack[stack_ptr++] = 0;                   /* Node Number in the Tree */
      stack[stack_ptr++] = 0;                   /* Element Span Minumum */
      stack[stack_ptr++] = t->elements_num - 1; /* Element Span Maximum */
      Bounds_Copy1d(&(t->extent), &(t->tree_safety_boxes[0]));
      next_node++;
      /* Construct k-D tree by setting up each pair of child nodes */
      while (stack_ptr) {
         /* Pop the top entry off the stack */
         max = stack[--stack_ptr];
         min = stack[--stack_ptr];
         parent = stack[--stack_ptr];
         /* If the current node should be a leaf node, make it one */
         if ((max - min) == 0) {
            Bounds_Copy1d(&(t->elements[idx[min]]), &(t->tree_safety_boxes[parent]));
            t->tree_link[parent] = - idx[min];
            continue;
         }
         /* Select optimum cutting direction for the parent node's safety box */
         cut_direction = -1;
         max_width = NEGATIVE_INFINITY;
         for (i = 0; i < 1; i++) {
            width = Bounds_WidthAxis1d(&(t->tree_safety_boxes[parent]), i);
            if(width > max_width) {
               max_width = width;
               cut_direction = i;
            }
         }
         assert(cut_direction >= 0);
         /* Do a median sort of the elements under the parent node. The sort key
            is the center point of the element bounding boxes along the selected
            cutting direction. */
         mid = (min + max) /2;
         median_sort1d(t, cut_direction, mid - min, max - min + 1, &(idx[min]));
         /* Give the parent a reference to its two children */
         t->tree_link[parent] = next_node;
         /* Add the "left" child to the tree and the stack */
         stack[stack_ptr++] = next_node;  /* Node Number in the Tree */
         stack[stack_ptr++] = min;        /* Element Span Minimum */
         stack[stack_ptr++] = mid;        /* Element Span Maximum */
         Bounds_Infinite1d(&(t->tree_safety_boxes[next_node]));
         for (i = min; i <= mid; i++) {
            Bounds_AddBounds1d(&(t->tree_safety_boxes[next_node]),
                             &(t->elements[idx[i]]));
         }
         next_node++;
         /* Add the "right" child to the tree and the stack */
         stack[stack_ptr++] = next_node;  /* Node Number in the Tree */
         stack[stack_ptr++] = mid + 1;    /* Element Span Minimum */
         stack[stack_ptr++] = max;        /* Element Span Maximum */
         Bounds_Infinite1d(&(t->tree_safety_boxes[next_node]));
         for (i = min + 1; i <= max; i++) {
            Bounds_AddBounds1d(&(t->tree_safety_boxes[next_node]),
                             &(t->elements[idx[i]]));
         }
         next_node++;
      }
      /* Destroy the temporary arrays */
      FREE(stack);
      FREE(idx);
   }
   /* Mark the tree "built" */
   t->tree_built = true;
}

void KDTree_QueryBoxIntersect1d(TKDTree1d* t,
                              int* result_num, int* result_indicies,
                              TBounds1d* box)
{
   int stack_ptr, node;
   TBounds1d sb;
   int* stack;
   
   assert(t && result_num && result_indicies && box);
   /* Build the k-D tree if necessary */
   if(!t->tree_built){
      //printf("BUILDING TREE... \n");
      //fflush(stdout);
      KDTree_CreateTree1d(t);
   }
   /* Allocate the results array */
   *result_num = 0;
   /* Create the temporary stack array */
   stack_ptr = 0;
   stack = MALLOC(t->tree_size, int);
   
   /* Put the root node of the tree onto the stack */
   stack[stack_ptr++] = 0;
   /* Search the k-D tree until the stack is empty */
   
   while (stack_ptr) {
      /* Pop the top entry off the stack */
      node = stack[--stack_ptr];
      /* Check if the query box intersects an epsilon-expanded safety box for
         the current node. */
      Bounds_Copy1d(&(t->tree_safety_boxes[node]), &sb);
      //Bounds_AddEpsilon1d(&sb, ENTITY_COINCIDENCE_TOLERANCE);
      /* If the query box doesn't intersect this node's safety box, we are done
         visiting the node and should continue with the next node */
      if(!Bounds_IsOverlappingBounds1d(&sb, box))
         continue;
      /* If the current node is a leaf node, add it to the collision list. If
         the current node is an interior node, add its children to the stack. */
      if(t->tree_link[node] <= 0) {
         result_indicies[*result_num] = - t->tree_link[node];
         (*result_num)++;
      }
      else {
         stack[stack_ptr++] = t->tree_link[node];
         stack[stack_ptr++] = t->tree_link[node] + 1;
      }
   }
   /* Destroy the temporary stack array */
   FREE(stack);
}

