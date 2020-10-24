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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "gpu.h"

#ifdef HAVE_CL_DOUBLE
typedef double real;
#ifdef HAVE_OPENCL
typedef cl_double cl_real;
typedef cl_double4 cl_real4;
#endif
#else
typedef float real;
#ifdef HAVE_OPENCL
typedef cl_float cl_real;
typedef cl_float4 cl_real4;
#endif
#endif

#ifndef DEVICE_DETECT_DEBUG
#define DEVICE_DETECT_DEBUG 0
#endif

#ifdef HAVE_OPENCL
void GPUInit(cl_context *context, cl_command_queue *queue, int *is_nvidia, cl_program *program, char *filename) {
    
   cl_platform_id* platforms;
   cl_platform_id platform = NULL;
   cl_uint num_platforms;
   cl_uint num_devices;
   cl_device_id* devices;
   cl_uint nDevices_selected=0;
   int *device_appropriate;
   int device_selected = -99;
   cl_int platform_selected = -1;
   //cl_program program;
   cl_int ierr = 0;
  
   // Get the number of platforms first, then allocate and get the platform
   ierr = clGetPlatformIDs(0, NULL, &num_platforms);
   if (ierr != CL_SUCCESS){
      printf("GPU_INIT: Error with clGetPlatformIDs call in file %s at line %d\n", __FILE__, __LINE__);
      if (ierr == CL_INVALID_VALUE){
         printf("GPU_INIT: Invalid value in clGetPlatformID call\n");
      }
      exit(ierr);
   }
   if (num_platforms == 0) {
      printf("GPU_INIT: Error -- No opencl platforms detected in file %s at line %d\n", __FILE__, __LINE__);
      exit(-1);
   }
   if (DEVICE_DETECT_DEBUG){
      printf("\n\nGPU_INIT: %d opencl platform(s) detected\n",num_platforms);
   }

   platforms = (cl_platform_id *)malloc(num_platforms*sizeof(cl_platform_id));

   ierr = clGetPlatformIDs(num_platforms, platforms, NULL);
   if (ierr != CL_SUCCESS){
      printf("GPU_INIT: Error with clGetPlatformIDs call in file %s at line %d\n", __FILE__, __LINE__);
      if (ierr == CL_INVALID_VALUE){
         printf("Invalid value in clGetPlatformID call\n");
      }
   }

   if (DEVICE_DETECT_DEBUG){
      char info[1024];
      for (uint iplatform=0; iplatform<num_platforms; iplatform++){
         printf("  Platform %d:\n",iplatform+1);

         //clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_PROFILE,   1024L,info,0);
         //printf("    CL_PLATFORM_PROFILE    : %s\n",info);

         clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_VERSION,   1024L,info,0);
         printf("    CL_PLATFORM_VERSION    : %s\n",info);

         clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_NAME,      1024L,info,0);
         printf("    CL_PLATFORM_NAME       : %s\n",info);

         clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_VENDOR,    1024L,info,0);
         printf("    CL_PLATFORM_VENDOR     : %s\n",info);

         //clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_EXTENSIONS,1024L,info,0);
         //printf("    CL_PLATFORM_EXTENSIONS : %s\n",info);
      }
      printf("\n");
   }

   char info[1024];
   clGetPlatformInfo(platforms[0],CL_PLATFORM_VENDOR, 1024, info, 0);

   // Get the number of devices, allocate, and get the devices
   for (uint iplatform=0; iplatform<num_platforms; iplatform++){
      ierr = clGetDeviceIDs(platforms[iplatform],CL_DEVICE_TYPE_GPU,0,NULL,&num_devices);
      if (ierr == CL_DEVICE_NOT_FOUND) {
         if (DEVICE_DETECT_DEBUG) {
           printf("Warning: Device of requested type not found for platform %d in clGetDeviceID call\n",iplatform);
         }
         continue;
      }
      if (ierr != CL_SUCCESS) {
        /* Possible Errors
         *  CL_INVALID_PLATFORM:
         *  CL_INVALID_DEVICE_TYPE:
         *  CL_INVALID_VALUE:
         *  CL_DEVICE_NOT_FOUND:
         */
        printf("GPU_INIT clGetDeviceIDs ierr %d file %s line %d\n", ierr, __FILE__, __LINE__);
      }
      if (DEVICE_DETECT_DEBUG){
         printf("GPU_INIT: %d opencl devices(s) detected\n",num_devices);
      }
      platform_selected = iplatform;
      platform = platforms[iplatform];
      nDevices_selected = num_devices;
   }

   if (platform_selected == -1){
      printf("Warning: Device of requested type not found in clGetDeviceID call\n");
      exit(-1);
   }

   num_devices = nDevices_selected;

   devices = (cl_device_id *)malloc(num_devices*sizeof(cl_device_id));
   device_appropriate = malloc(num_devices*sizeof(int));
  
   ierr = clGetDeviceIDs(platforms[platform_selected], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
   if(ierr != CL_SUCCESS) {
     printf("Error getting device ids\n");
     exit(ierr);
   }
 
  int idevice_appropriate = 0;
  for (uint idevice=0; idevice<num_devices; idevice++){
     device_appropriate[idevice] = device_double_support(devices[idevice]);;
     if (device_appropriate[idevice] == 1){
        if (device_selected == -99) device_selected = idevice;
        devices[idevice_appropriate] = devices[idevice];
        idevice_appropriate++;
     }
     if (DEVICE_DETECT_DEBUG){
        printf(  "  Device %d:\n", idevice+1);
        device_info(devices[idevice]);
     }
  }
  num_devices = idevice_appropriate;

  if (DEVICE_DETECT_DEBUG) {
     printf("Device selected is %d number of appropriate devices %d\n",device_selected, num_devices);
  }

  cl_context_properties context_properties[3]=
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0 // 0 terminates list
  };   

  *context = clCreateContext(context_properties, num_devices, devices, NULL, NULL, &ierr);
  if(ierr != CL_SUCCESS) {
    printf("Error creating context\n");
    exit(ierr);
  }
  *queue = clCreateCommandQueue(*context, devices[0], CL_QUEUE_PROFILING_ENABLE, &ierr);
  if(ierr != CL_SUCCESS) {
    printf("Error creating command queue\n");
    exit(ierr);
  }
  
  // Load the kernel source code into the array source
  struct stat statbuf;
  FILE *fh;
  char *source;
  
  fh = fopen(filename, "r");
  if (!fh) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(-1);
  }
  stat(filename, &statbuf);
  source = (char*)malloc(statbuf.st_size + 1);
  if( fread(source, statbuf.st_size, 1, fh) != 1) {
      printf("Problem reading program source file\n");
  }
  source[statbuf.st_size] = '\0';
  fclose( fh );
  
  *program = clCreateProgramWithSource(*context, 1, (const char**) &source, NULL, &ierr);
  if (ierr != CL_SUCCESS){
      printf("clCreateProgramWithSource returned an ierr %d at line %d in file %s\n", ierr,__LINE__,__FILE__);
  }
  //printf("%d %s\n", (int)statbuf.st_size, source);
  
  size_t nReportSize;
  char* BuildReport;
  
#ifdef HAVE_CL_DOUBLE
  if (*is_nvidia) {
     ierr = clBuildProgram(*program, 0, NULL, "-DHAVE_CL_DOUBLE -DIS_NVIDIA", NULL, NULL);
  } else {
     ierr = clBuildProgram(*program, 0, NULL, "-DHAVE_CL_DOUBLE", NULL, NULL);
  }
#else
  if (*is_nvidia) {
     ierr = clBuildProgram(*program, 0, NULL, "-DNO_CL_DOUBLE -DIS_NVIDIA -cl-single-precision-constant", NULL, NULL);
  } else {
     ierr = clBuildProgram(*program, 0, NULL, "-DNO_CL_DOUBLE -cl-single-precision-constant", NULL, NULL);
  }
#endif
  if (ierr != CL_SUCCESS){
      printf("clBuildProgram returned an ierr %d at line %d in file %s\n", ierr,__LINE__,__FILE__);
      ierr = clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &nReportSize);
      if (ierr != CL_SUCCESS) {
          switch (ierr){
              case CL_INVALID_DEVICE:
                  printf("Invalid device in clProgramBuildInfo\n");
                  break;
              case CL_INVALID_VALUE:
                  printf("Invalid value in clProgramBuildInfo\n");
                  break;
              case CL_INVALID_PROGRAM:
                  printf("Invalid program in clProgramBuildInfo\n");
                  break;
          }
      }
      
      BuildReport = (char *)malloc(nReportSize);
      
      ierr = clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, nReportSize, BuildReport, NULL);
      if (ierr != CL_SUCCESS) {
          switch (ierr){
              case CL_INVALID_DEVICE:
                  printf("Invalid device in clProgramBuildInfo\n");
                  break;
              case CL_INVALID_VALUE:
                  printf("Invalid value in clProgramBuildInfo\n");
                  break;
              case CL_INVALID_PROGRAM:
                  printf("Invalid program in clProgramBuildInfo\n");
                  break;
          }
      }
      printf("%s\n", BuildReport);
  }
  
}

int device_double_support(cl_device_id device){
   int have_double = 0;
   char info[1024];

   clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(info), &info, NULL);

   if (!(strstr(info,"cl_khr_fp64") == NULL)){
     if (DEVICE_DETECT_DEBUG){
        printf(  "    Device has double : %s\n\n", strstr(info,"cl_khr_fp64"));
     }
     have_double = 1;
   }

   return(have_double);
}

void device_info(cl_device_id device){
   if (device == NULL) {
      printf(" Error with device in device_info\n");
   }
   char info[1024];
   cl_bool iflag;
   cl_uint inum;
   size_t isize;
   cl_ulong ilong;
   cl_device_type device_type;
   cl_command_queue_properties iprop;

   clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(device_type),&device_type,0);
   if( device_type & CL_DEVICE_TYPE_CPU )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_CPU");
   if( device_type & CL_DEVICE_TYPE_GPU )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_GPU");
   if( device_type & CL_DEVICE_TYPE_ACCELERATOR )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_ACCELERATOR");
   if( device_type & CL_DEVICE_TYPE_DEFAULT )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_DEFAULT");

   clGetDeviceInfo(device,CL_DEVICE_AVAILABLE,sizeof(iflag),&iflag,0);
   if (iflag == CL_TRUE) {
      printf(  "    CL_DEVICE_AVAILABLE                  : TRUE\n");
   } else {
      printf(  "    CL_DEVICE_AVAILABLE                  : FALSE\n");
   }

   clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_VENDOR                     : %s\n", info);

   clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_NAME                       : %s\n", info);

   clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(info), &info, NULL);
   printf(  "    CL_DRIVER_VERSION                    : %s\n", info);

   clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_VERSION                    : %s\n", info);

   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_COMPUTE_UNITS          : %d\n", inum);

   clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS   : %d\n", inum);

   size_t *item_sizes = (size_t *)malloc(inum*sizeof(size_t));
   clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(item_sizes),item_sizes,0);
   printf(  "    CL_DEVICE_MAX_WORK_ITEM_SIZES        : %ld %ld %ld\n",
         item_sizes[0], item_sizes[1], item_sizes[2]);
   free(item_sizes);

   clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(isize),&isize,0);
   printf(  "    CL_DEVICE_MAX_WORK_GROUP_SIZE        : %ld\n", isize);

   clGetDeviceInfo(device,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_CLOCK_FREQUENCY        : %d\n", inum);

   clGetDeviceInfo(device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_MEM_ALLOC_SIZE         : %d\n", inum);

#ifdef __APPLE_CC__
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_SIZE            : %llu\n", ilong);

   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE      : %llu\n", ilong);
#else
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_SIZE            : %lu\n", ilong);

   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE      : %lu\n", ilong);
#endif
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE  : %d\n", inum);

   clGetDeviceInfo(device,CL_DEVICE_MAX_CONSTANT_ARGS,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_GLOBAL_MAX_CONSTANT_ARGS   : %d\n", inum);

   clGetDeviceInfo(device,CL_DEVICE_ERROR_CORRECTION_SUPPORT,sizeof(iflag),&iflag,0);
   if (iflag == CL_TRUE) {
      printf(  "    CL_DEVICE_ERROR_CORRECTION_SUPPORT   : TRUE\n");
   } else {
      printf(  "    CL_DEVICE_ERROR_CORRECTION_SUPPORT   : FALSE\n");
   }

   clGetDeviceInfo(device,CL_DEVICE_PROFILING_TIMER_RESOLUTION,sizeof(isize),&isize,0);
   printf(  "    CL_DEVICE_PROFILING_TIMER_RESOLUTION : %ld nanosecs\n", isize);

   clGetDeviceInfo(device,CL_DEVICE_QUEUE_PROPERTIES,sizeof(iprop),&iprop,0);
   if (iprop & CL_QUEUE_PROFILING_ENABLE) {
      printf(  "    CL_DEVICE_QUEUE PROFILING            : AVAILABLE\n");
   } else {
      printf(  "    CL_DEVICE_QUEUE PROFILING            : NOT AVAILABLE\n");
   }

   clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_EXTENSIONS                 : %s\n\n", info);

}
#endif

