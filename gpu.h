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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __APPLE_CC__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

cl_kernel interpolate_kernel;

void GPUInit(cl_context *context, cl_command_queue *queue, int *is_nvidia, cl_program *program, char *filename);
int device_double_support(cl_device_id device);
void device_info(cl_device_id device);
