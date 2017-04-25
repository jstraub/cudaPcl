
/*
 * Copyright (c) 2010 - 2011 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/*
 * traversal.c -- Demonstrates the rtuTraversal API.  The input is a pair of files
 * composed of the rays to trace and the mesh to trace it against.  The output is an image
 * representing the normalized intersection depth, though the API supports additional
 * types of output.
 *
 */

#include <optix_world.h>
#include <sutil.h> 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* assumes that there is no context, just print to stderr */
#define RTU_TRAVERSE_CHECK_ERROR( func )                           \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS ) {                                     \
      const char* error_message;                                   \
      rtuTraversalGetErrorString( t, code, &error_message );       \
      sutilReportError(error_message);                             \
      exit(2);                                                     \
    }                                                              \
  } while(0)


void printUsageAndExit( const char* argv0 );

void readData( FILE* in, unsigned int* num_elems, void** data, size_t elem_size,
               const char* type, const char* filename);
void readRays( const char* filename, unsigned int* nrays, float** rays );
void readMesh( const char* filename,
               unsigned int* nverts, float** verts,
               unsigned int* ntris, unsigned int** indices );
void readSoup( const char* filename, unsigned int* ntris, float** tris );
void soupify( float** tris,
              unsigned int nverts, float* verts,
              unsigned int ntris, unsigned int* indices );
void outputDepth( RTUtraversalresult* results, unsigned int num_rays,
                  const char* filename,
                  unsigned int width, unsigned int height);

/* Reads in a 4 byte int for the number of elements then reads that much elem_size byte
 * sized data from the same file */
void readData( FILE* in, unsigned int* num_elems, void** data, size_t elem_size,
               const char* type, const char* filename)
{
  size_t bytes_read, buffer_size;
  
  bytes_read = fread( (void*)( num_elems ), sizeof( char ), sizeof(unsigned int), in );
  if (bytes_read != sizeof( unsigned int )) {
    fprintf(stderr, "ERROR: reading number of %s from %s\n", type, filename);
    exit(2);
  }
  fprintf( stdout, "Reading in %u %s ... \n", *num_elems, type );

  /* read in the data */
  buffer_size = *num_elems * elem_size;
  /* Allocate the buffer if it doesn't exist */
  if (*data == 0)
    *data = (float*)malloc( buffer_size );
  if (*data == 0) {
    fprintf( stderr, "ERROR: allocating buffer of size %llu for %s when reading %s\n",
             (unsigned long long)buffer_size,
             type, filename);
    exit(2);
  }
  bytes_read = fread( *data, sizeof( char ), buffer_size, in );
  if (bytes_read != buffer_size) {
    fprintf( stderr,
             "ERROR: when attempting to read %llu bytes for %s from %s, read only %llu bytes.\n",
             (unsigned long long)buffer_size,
             type, filename,
             (unsigned long long)bytes_read );
    exit(2);
  }
}

/* The rays file is assumed to have a 4 byte integer for the number of rays followed by
 * that many rays.
 *
 * nrays (4 byte int)
 * rays ( { float3 origin, float3 direction } * nrays )
 *
 */
void readRays( const char* filename, unsigned int* nrays, float** rays )
{
  FILE* in = fopen( filename, "rb" );
  if( !in ) {
    fprintf( stderr, "ERROR: Failed to open file '%s'\n", filename );
    exit(2);
  }

  readData(in, nrays, (void**)rays, 6 * sizeof(float), "rays", filename);
}


/* The triangles are represented as a mesh, vertices followed by indices.
 *
 * nverts (4 bypte int)
 * verts  ( 3 * float3 vertex * nverts )
 * ntris  (4 byte int)
 * indices ( 3 * int3 vindex * ntris )
 *
 */
void readMesh( const char* filename,
               unsigned int* nverts, float** verts,
               unsigned int* ntris, unsigned int** indices )
{
  FILE* in = fopen( filename, "rb" );
  if( !in ) {
    fprintf( stderr, "ERROR: Failed to open file '%s'\n", filename );
    exit(2);
  }

  /* read in the vertices */
  readData(in, nverts, (void**)verts, 3 * sizeof(float), "vertices", filename);

  /* read in the index data */
  readData(in, ntris, (void**)indices, 3 * sizeof( unsigned int ), "indices", filename);
}

/* The triangle soup file is assumed to have a 4 byte integer for the number of
 * triangles followed by ntris*3 vertices (float3).
 *
 * ntris (4 byte int)
 * triangle vertices ( { float3 v0, float3 v1, float3 v2 } * ntris )
 *
 */
void readSoup( const char* filename, unsigned int* ntris, float** tris )
{
  FILE* in = fopen( filename, "rb" );
  if( !in ) {
    fprintf( stderr, "ERROR: Failed to open file '%s'\n", filename );
    exit(2);
  }

  readData(in, ntris, (void**)tris, 3 * sizeof(float3), "tri soup", filename);
}


void soupify( float** tris,
              unsigned int nverts, float* verts,
              unsigned int ntris, unsigned int* indices )
{
  unsigned int tri_index;
  float3* soup;
  float3* vertices;

  size_t buffer_size = ntris*3 * sizeof(float3);
  *tris = (float*)malloc( buffer_size );
  if (*tris == 0) {
    fprintf( stderr, "ERROR: allocating buffer of size %llu for %s\n",
             (unsigned long long)(buffer_size),
             "tris");
    exit(2);
  }

  // For each triangle
  soup = (float3*)*tris;
  vertices = (float3*)verts;
  for(tri_index = 0; tri_index < ntris; ++tri_index)
  {
    // For each vertex
    unsigned int i;
    for(i = 0; i < 3; ++i)
    {
      unsigned int ind = tri_index*3+i;
      unsigned int vert_index = indices[ind];
      soup[ind].x = vertices[vert_index].x;
      soup[ind].y = vertices[vert_index].y;
      soup[ind].z = vertices[vert_index].z;
    }
  }
}

void outputDepth( RTUtraversalresult* results, unsigned int num_rays,
                  const char* filename,
                  unsigned int width, unsigned int height)
{
  FILE* outfile = fopen( filename, "wb" );
  float max_val = 0.0f;
  unsigned int i,j;
  int num_hits = 0;
  unsigned int written = 0u;
  unsigned int max = width * height;
  if( !outfile ) {
    fprintf( stderr, "Could not open file '%s' for writing. Skipping write.\n", filename );
    exit(0);
  }
  for( i = 0; i < num_rays; ++i ) {
    if( results[i].prim_id != -1 ) {
      num_hits++;
      max_val = results[i].t > max_val ? results[i].t : max_val;
    }
  }
  fprintf( stdout, "Got maxval of %f over %i hits\n", max_val, num_hits );
  fprintf( outfile, "P2\n%u %u\n255\n", width, height );
  for( j = height; j > 0 && written < max; j--) {
    for ( i = 0; i < width && written < max; i++) {
      unsigned int ind = (j-1) * width + i;
      if (ind >= num_rays)
        continue;
      if( results[ind].prim_id != -1 )
        fprintf( outfile, "%i\n", (int)(results[ind].t / max_val * 255.0f) );
      else
        fprintf( outfile, "%i\n", 0 );
      written++;
    }
  }
}

  
void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "App options:\n" );
  fprintf( stderr, "  --help | -h              Print this message\n");
  fprintf( stderr, "  --cpu NUM                Use cpu raytracer, specifying number of threads\n" );
  fprintf( stderr, "  --iter=NUM1xNUM2         Specify number of warmup and timed iterations\n" );
  fprintf( stderr, "  --print                  Print results to stdout\n" );
  fprintf( stderr, "  --external-context       Use an externally created OptiX context\n" );
  fprintf( stderr, "  --time-ray-data-transfer Re-upload ray data every iter and include in timing\n" );
  fprintf( stderr, "  --time-get-results-transfer  Copy results back after each iteration and include in timing\n");
  fprintf( stderr, "  --rays <FILENAME>        Filename specifying the rays to trace\n" );
  fprintf( stderr, "  --mesh <FILENAME>        Filename specifying the mesh to intersect the rays against\n" );
  fprintf( stderr, "  --soup <FILENAME>        Filename specifying the triangle soup to intersect the rays against.\n" );
  fprintf( stderr, "                           This argument take precedence over --mesh.\n" );
  fprintf( stderr, "  --soupify                Make a triangle soup from the mesh.\n" );
  fprintf( stderr, "  --compute-barycentric    Also compute the barycentric coordinates.\n" );
  fprintf( stderr, "  --compute-normals        Also compute the normals at the hit location.\n" );
  fprintf( stderr, "  --compute-backface       Also compute if the triangle was backfacing from the ray.\n" );
  fprintf( stderr, "  --dim=<width>x<height>   Set image dimensions.  Only meaningful if the rays\n" );
  fprintf( stderr, "                           provided represent a 2D image plane.\n" );
  exit(1);
}


int main( int argc, char** argv )
{
  RTUtraversal        t;
  RTUtraversalresult* results = 0;
  float*              barycentric_coords = 0u; // sizeof(float2)*num_rays;
  float*              normals = 0u;  // sizeof(float3)*num_rays;
  char*               backface = 0u; // sizeof(char)*num_rays;
  RTcontext           context;

  float*               rays_mapped_ptr = 0u;
  RTUtraversalresult*  results_mapped_ptr = 0u;
  void*                barycentric_coords_mapped_ptr = 0u;
  void*                normals_mapped_ptr = 0u;
  void*                backface_mapped_ptr = 0u;

  float*        rays = 0u;
  float*        verts = 0u;
  float*        tris = 0u;
  unsigned int* indices = 0u;
  double        t0, t1;
  unsigned int  num_warmup=50u, num_timed=100u;
  unsigned int  num_rays, num_tris, num_verts;
  unsigned int  cpu_threads = 0u;
  unsigned int  external_context = 0u;
  unsigned int  print = 0u;
  unsigned int  time_ray_data_transfer = 0u; 
  unsigned int  time_get_results = 0u;
  unsigned int  use_soup = 0u;      // Triangles are in soup format
  unsigned int  soupify_mesh = 0u;  // turns a triangle mesh into a triangle soup
  unsigned int  argcu = (unsigned int)argc;
  unsigned int  i;
  RTUoutput compute_barycentric = RTU_OUTPUT_NONE;
  RTUoutput compute_normals     = RTU_OUTPUT_NONE;
  RTUoutput compute_backface    = RTU_OUTPUT_NONE;
  char* rays_filename = 0;
  char* mesh_filename = 0;
  char* soup_filename = 0;
  unsigned int rays_filename_allocated = 0u;
  unsigned int mesh_filename_allocated = 0u;
  unsigned int width = 640u, height = 480u;
  char char_continue = 0;

  for( i = 1; i < argcu; ++i ) {
    const char* arg = argv[i];
    if( strcmp( arg, "--cpu" ) == 0 ) {
      if( i == argcu ) {
        printUsageAndExit( argv[0] );
      }
      cpu_threads = atoi( argv[++i] );
    } else if( strcmp( arg, "--print" ) == 0 ) {
      print = 1u;
    } else if( strcmp( arg, "--external-context" ) == 0 ) {
      external_context = 1u;
    } else if( strcmp( arg, "--time-ray-data-transfer" ) == 0 ) {
      time_ray_data_transfer = 1u;
    } else if( strcmp( arg, "--time-get-results-transfer" ) == 0 ) {
      time_get_results = 1u;
    } else if( strcmp( arg, "--compute-barycentric" ) == 0 ) {
      compute_barycentric = RTU_OUTPUT_BARYCENTRIC;
    } else if( strcmp( arg, "--compute-normals" ) == 0 ) {
      compute_normals = RTU_OUTPUT_NORMAL;
    } else if( strcmp( arg, "--compute-backface" ) == 0 ) {
      compute_backface = RTU_OUTPUT_BACKFACING;
    } else if( strcmp( arg, "--soupify" ) == 0 ) {
      soupify_mesh = 1u;
      use_soup = 1u;
    } else if( strcmp( arg, "--rays" ) == 0 ) {
      rays_filename = argv[++i];
    } else if( strcmp( arg, "--mesh" ) == 0 ) {
      mesh_filename = argv[++i];
    } else if( strcmp( arg, "--soup" ) == 0 ) {
      soup_filename = argv[++i];
      use_soup = 1u;
    } else if( strcmp( arg, "--wait" ) == 0 ) {
      char_continue = 1;
    } else if ( strncmp( arg, "--dim=", 6 ) == 0 ) {
      const char *dims_arg = &argv[i][6];
      if ( sutilParseImageDimensions( dims_arg, &width, &height ) != RT_SUCCESS ) {
        fprintf( stderr, "Invalid window dimensions: '%s'\n", dims_arg );
        printUsageAndExit( argv[0] );
      }
    } else if( (strcmp( arg, "--help" ) == 0) || (strcmp( arg, "-h") == 0) ) {
      printUsageAndExit( argv[0] );
    } else {
      char substring[32];
      strncpy( substring, arg, 7 );
      substring[7] = '\0';
      if( strcmp( substring, "--iter=" ) == 0 ) {
        strncpy( substring, &(arg[7]), 32 );
        if( sutilParseImageDimensions( substring, &num_warmup, &num_timed ) != RT_SUCCESS ) {
          printUsageAndExit( arg );
        }
      } else {
        fprintf( stderr, "Unknown option '%s'\n", arg ); printUsageAndExit( argv[0] ); }
    }
  }

  if (soup_filename && rays_filename == 0) {
    fprintf(stderr, "Must specify rays file name if specifying triangle soup file name.\n");
    printUsageAndExit( argv[0] );
  }
  if (mesh_filename && rays_filename == 0) {
    fprintf(stderr, "Must specify rays file name if specifying triangle mesh file name.\n");
    printUsageAndExit( argv[0] );
  }
  /* Initialize traversal state */
  
  context = 0;
  if( external_context ) {
    RT_CHECK_ERROR_NO_CONTEXT( rtContextCreate( &context ) );
  }

  fprintf( stdout, "Initializing traversal state ... " ); fflush( stdout );
  sutilCurrentTime( &t0 );
  RT_CHECK_ERROR_NO_CONTEXT( rtuTraversalCreate( &t, RTU_QUERY_TYPE_CLOSEST_HIT,
                                                     RTU_RAYFORMAT_ORIGIN_DIRECTION_INTERLEAVED,
                                                     use_soup ? RTU_TRIFORMAT_TRIANGLE_SOUP : RTU_TRIFORMAT_MESH,
                                                     compute_barycentric | compute_normals | compute_backface,
                                                     cpu_threads ? RTU_INITOPTION_CPU_ONLY : RTU_INITOPTION_NONE,
                                                     context ) );
  if( cpu_threads ) {
    fprintf( stdout, " numthreads: %i\n", cpu_threads );
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalSetOption( t, RTU_OPTION_INT_NUM_THREADS, (void*)(&cpu_threads) ) );
  }

  // Read geometry
  if ( soup_filename == 0 ) {

    if ( mesh_filename == 0 )
    {
      const char datafile[] = "/traversal/conference.mesh.binary";
      const char* samples_dir = sutilSamplesDir();
      char* filename = (char*)malloc(sizeof(datafile)+strlen(samples_dir)+1);
      sprintf( filename, "%s%s", samples_dir, datafile);
      mesh_filename = filename;
      mesh_filename_allocated = 1u;
    }

    readMesh( mesh_filename, &num_verts, &verts, &num_tris, &indices );
    if (mesh_filename_allocated)
      free ( mesh_filename );

    if ( soupify_mesh )
      soupify(&tris, num_verts, verts, num_tris, indices);

  } else {
    readSoup( soup_filename, &num_tris, &tris );
  }

  if ( rays_filename == 0 )
  {
//    const char datafile[] = "/traversal/conference.rays.binary";
//    const char* samples_dir = sutilSamplesDir();
//    char* filename = (char*)malloc(sizeof(datafile)+strlen(samples_dir)+1);
//    sprintf( filename, "%s%s", samples_dir, datafile);
//    rays_filename = filename;
//    rays_filename_allocated = 1u;
    rays = (float*)malloc(width*height*sizeof(float)*6);
    int i, j;
    for (i=0; i<height; ++i) 
      for (j=0; j<width; ++j) {
        rays[(i*width+j)*6+0] = 0.;
        rays[(i*width+j)*6+1] = 0.;
        rays[(i*width+j)*6+2] = 0.;
        rays[(i*width+j)*6+3] = j-(width-1.)*0.5;
        rays[(i*width+j)*6+4] = i-(height-1.)*0.5;
        rays[(i*width+j)*6+5] = 540.;
        float len = sqrtf(rays[(i*width+j)*6+3]*rays[(i*width+j)*6+3]+
          rays[(i*width+j)*6+4]*rays[(i*width+j)*6+4]+
          rays[(i*width+j)*6+5]*rays[(i*width+j)*6+5]);
        rays[(i*width+j)*6+3] /= len;
        rays[(i*width+j)*6+4] /= len;
        rays[(i*width+j)*6+5] /= len;
      }
    num_rays = width*height;
  } else {
    readRays( rays_filename, &num_rays, &rays );
  }
  RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapRays( t, num_rays, &rays_mapped_ptr ) );
  memcpy(rays_mapped_ptr, rays, num_rays*sizeof(float3)*2 );
  RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapRays( t ) );

  if (rays_filename_allocated)
    free ( rays_filename );

  if ( use_soup )
  {
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalSetTriangles( t, num_tris, tris ) );
  } else {

    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalSetMesh( t, num_verts, verts, num_tris, indices ) );
  }

  if( time_get_results ) {
    results = (RTUtraversalresult*)malloc(num_rays*sizeof(RTUtraversalresult));
    if ( compute_barycentric )
      barycentric_coords = (float*)malloc(num_rays*sizeof(float2));
    if ( compute_normals )
      normals = (float*)malloc(num_rays*sizeof(float3));
    if ( compute_backface )
      backface = (char*)malloc(num_rays*sizeof(char));
  }

  sutilCurrentTime( &t1 );
  fprintf( stdout, "Setup: %4.3lf sec.\n", t1 - t0 ); 
  
  fprintf( stdout, "Building accel and compiling ... " ); fflush( stdout );
  sutilCurrentTime( &t0 );
  RTU_TRAVERSE_CHECK_ERROR( rtuTraversalPreprocess( t ) );
  sutilCurrentTime( &t1 );
  fprintf( stdout, "%4.3lf sec.\n", t1 - t0 );
  if ( char_continue ) {
    fprintf( stdout, "Press enter to continue...\n");
    fgetc( stdin );
  }


  fprintf( stdout, "Traversing warmup rays (%u iterations) ... ", num_warmup ); fflush( stdout );
  sutilCurrentTime( &t0 );
  for( i = 0; i < num_warmup; ++i ) {
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalTraverse( t ) );
  }
  sutilCurrentTime( &t1 );
  fprintf( stdout, "%4.3lf sec.\n", t1 - t0 ); 

  fprintf( stdout, "Traversing timed rays (%u iterations) ... ", num_timed ); fflush( stdout );
  sutilCurrentTime( &t0 );
  for( i = 0; i < num_timed; ++i ) {
    if( time_ray_data_transfer ) {
      size_t ray_buf_size = num_rays*sizeof(float3)*2;
      RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapRays( t, num_rays, &rays_mapped_ptr ) );
      memcpy(rays_mapped_ptr, rays, ray_buf_size );
      RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapRays( t ) );
    }

    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalTraverse( t ) );

    if( time_get_results ) {
      RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapResults( t, &results_mapped_ptr ) );
      memcpy( results, results_mapped_ptr, num_rays * sizeof(RTUtraversalresult) );
      RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapResults( t ) );
      if ( compute_barycentric ) {
        RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapOutput( t, RTU_OUTPUT_BARYCENTRIC, &barycentric_coords_mapped_ptr ) );
        memcpy( barycentric_coords, barycentric_coords_mapped_ptr, num_rays * sizeof(float2) );
        RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapOutput( t, RTU_OUTPUT_BARYCENTRIC ) );
      }
      if ( compute_normals ) {
        RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapOutput( t, RTU_OUTPUT_NORMAL, &normals_mapped_ptr ) );
        memcpy( normals, normals_mapped_ptr, num_rays * sizeof(float3) );
        RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapOutput( t, RTU_OUTPUT_NORMAL ) );
      }
      if ( compute_backface ) {
        RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapOutput( t, RTU_OUTPUT_BACKFACING, &backface_mapped_ptr ) );
        memcpy( backface, backface_mapped_ptr, num_rays * sizeof(char) );
        RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapOutput( t, RTU_OUTPUT_BACKFACING) );
      }
    }
  }
  sutilCurrentTime( &t1 );
  fprintf( stdout, "%4.3lf sec.\n", t1 - t0 ); 

  fprintf( stdout, "%7.3lf ms per iteration\n", (t1-t0) / (float)num_timed * 1000.0 );
  fprintf( stdout, "%7.3lf MRay/sec\n", (float)num_timed*(float)num_rays / (float)(t1-t0) / 1e6 );

  /* Even though we aren't using the data, force the read with map/unmap */
  if ( compute_barycentric ) {
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapOutput( t, RTU_OUTPUT_BARYCENTRIC, &barycentric_coords_mapped_ptr ) );
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapOutput( t, RTU_OUTPUT_BARYCENTRIC ) );
  }
  if ( compute_normals ) {
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapOutput( t, RTU_OUTPUT_NORMAL, &normals_mapped_ptr ) );
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapOutput( t, RTU_OUTPUT_NORMAL ) );
  }
  if ( compute_backface ) {
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapOutput( t, RTU_OUTPUT_BACKFACING, &backface_mapped_ptr ) );
    RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapOutput( t, RTU_OUTPUT_BACKFACING ) );
  }

  RTU_TRAVERSE_CHECK_ERROR( rtuTraversalMapResults( t, &results_mapped_ptr ) );
  outputDepth(results_mapped_ptr, num_rays, "traversal-depth.pgm", width, height);
 
  if( print ) {
    fprintf( stdout, "%8s %10s : %s\n", "ray", "prim idx", "t value" );
    for( i = 0; i < num_rays; ++i ) {
      float t_val = results_mapped_ptr[i].prim_id == -1 ? 0.0f : results_mapped_ptr[i].t;
      fprintf( stdout, "%8u %10i : %12.6f\n", i, results_mapped_ptr[i].prim_id, t_val );
    }
  }
  RTU_TRAVERSE_CHECK_ERROR( rtuTraversalUnmapResults( t ) );

  RTU_TRAVERSE_CHECK_ERROR( rtuTraversalDestroy( t ) );
  if( external_context ) {
    RTU_TRAVERSE_CHECK_ERROR( rtContextDestroy( context ) );
  }
  
  free( verts );
  free( indices );
  free( results );
  free( rays );
  if ( barycentric_coords ) free( barycentric_coords );
  if ( normals ) free( normals );
  if ( backface ) free( backface );

  return 0;
}
