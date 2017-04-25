
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
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

#include "tutorial.h"

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );


//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;

RT_PROGRAM void pinhole_camera()
{
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  output_buffer[launch_index] = make_color( prd.result );
}


//
// Environment map background
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
  prd_radiance.result = make_float3( tex2D(envmap, u, v) );
}


//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);

  rtTerminateRay();
}
  

//
// (NEW)
// Procedural rusted metal surface shader
//

/*
 * Translated to CUDA C from Larry Gritz's LGRustyMetal.sl shader found at:
 * http://renderman.org/RMR/Shaders/LGShaders/LGRustyMetal.sl
 *
 * Used with permission from tal AT renderman DOT org.
 */

rtDeclareVariable(float,   metalKa, , ) = 1;
rtDeclareVariable(float,   metalKs, , ) = 1;
rtDeclareVariable(float,   metalroughness, , ) = .1;
rtDeclareVariable(float,   rustKa, , ) = 1;
rtDeclareVariable(float,   rustKd, , ) = 1;
rtDeclareVariable(float3,  rustcolor, , ) = {.437, .084, 0};
rtDeclareVariable(float3,  metalcolor, , ) = {.7, .7, .7};
rtDeclareVariable(float,   txtscale, , ) = .02;
rtDeclareVariable(float,   rusty, , ) = 0.2;
rtDeclareVariable(float,   rustbump, , ) = 0.85;
rtDeclareVariable(float3,  ambient_light_color, , );
rtBuffer<BasicLight>       lights;   
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float,   importance_cutoff, , );
rtDeclareVariable(int,     max_depth, , );
rtDeclareVariable(float3,  reflectivity_n, , );
#define MAXOCTAVES 6

rtTextureSampler<float, 3> noise_texture;
static __device__ __inline__ float snoise(float3 p)
{
  return tex3D(noise_texture, p.x, p.y, p.z) * 2 -1;
}


RT_PROGRAM void box_closest_hit_radiance()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 hit_point = ray.origin + t_hit * ray.direction;

  /* Sum several octaves of abs(snoise), i.e. turbulence.  Limit the
   * number of octaves by the estimated change in PP between adjacent
   * shading samples.
   */
  float3 PP = txtscale * hit_point;
  float a = 1;
  float sum = 0;
  for(int i = 0; i < MAXOCTAVES; i++ ){
    sum += a * fabs(snoise(PP));
    PP *= 2.0f;
    a *= 0.5f;
  }

  /* Scale the rust appropriately, modulate it by another noise 
   * computation, then sharpen it by squaring its value.
   */
  float rustiness = step (1-rusty, clamp (sum,0.0f,1.0f));
  rustiness *= clamp (abs(snoise(PP)), 0.0f, .08f) / 0.08f;
  rustiness *= rustiness;

  /* If we have any rust, calculate the color of the rust, taking into
   * account the perturbed normal and shading like matte.
   */
  float3 Nrust = ffnormal;
  if (rustiness > 0) {
    /* If it's rusty, also add a high frequency bumpiness to the normal */
    Nrust = normalize(ffnormal + rustbump * snoise(PP));
    Nrust = faceforward (Nrust, -ray.direction, world_geo_normal);
  }

  float3 color = mix(metalcolor * metalKa, rustcolor * rustKa, rustiness) * ambient_light_color;
  for(int i = 0; i < lights.size(); ++i) {
    BasicLight light = lights[i];
    float3 L = normalize(light.pos - hit_point);
    float nmDl = dot( ffnormal, L);
    float nrDl = dot( Nrust, L);

    if( nmDl > 0.0f || nrDl > 0.0f ){
      // cast shadow ray
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      float Ldist = length(light.pos - hit_point);
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      float3 light_attenuation = shadow_prd.attenuation;

      if( fmaxf(light_attenuation) > 0.0f ){
        float3 Lc = light.color * light_attenuation;
        nrDl = max(nrDl * rustiness, 0.0f);
        color += rustKd * rustcolor * nrDl * Lc;

        float r = nmDl * (1.0f-rustiness);
        if(nmDl > 0.0f){
          float3 H = normalize(L - ray.direction);
          float nmDh = dot( ffnormal, H );
          if(nmDh > 0)
            color += r * metalKs * Lc * pow(nmDh, 1.f/metalroughness);
        }
      }

    }
  }

  float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n * (1-rustiness));
  float importance = prd_radiance.importance * optix::luminance( r );

  // reflection ray
  if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
    PerRayData_radiance refl_prd;
    refl_prd.importance = importance;
    refl_prd.depth = prd_radiance.depth+1;
    float3 R = reflect( ray.direction, ffnormal );
    optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
    rtTrace(top_object, refl_ray, refl_prd);
    color += r * refl_prd.result;
  }

  prd_radiance.result = color;
}
  

//
// Phong surface shading with shadows and schlick-approximated fresnel reflections.
// Uses procedural texture to determine diffuse response.
//
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   reflectivity, , );
rtDeclareVariable(float3,   tile_v0, , );
rtDeclareVariable(float3,   tile_v1, , );
rtDeclareVariable(float3,   crack_color, , );
rtDeclareVariable(float,    crack_width, , );

RT_PROGRAM void floor_closest_hit_radiance()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;

  float v0 = dot(tile_v0, hit_point);
  float v1 = dot(tile_v1, hit_point);
  v0 = v0 - floor(v0);
  v1 = v1 - floor(v1);

  float3 local_Kd;
  if( v0 > crack_width && v1 > crack_width ){
    local_Kd = Kd;
  } else {
    local_Kd = crack_color;
  }

  for(int i = 0; i < lights.size(); ++i) {
    BasicLight light = lights[i];
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot( ffnormal, L);

    if( nDl > 0.0f ){
      // cast shadow ray
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      float Ldist = length(light.pos - hit_point);
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      float3 light_attenuation = shadow_prd.attenuation;

      if( fmaxf(light_attenuation) > 0.0f ){
        float3 Lc = light.color * light_attenuation;
        color += local_Kd * nDl * Lc;

        float3 H = normalize(L - ray.direction);
        float nDh = dot( ffnormal, H );
        if(nDh > 0)
          color += Ks * Lc * pow(nDh, phong_exp);
      }

    }
  }

  float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n);
  float importance = prd_radiance.importance * optix::luminance( r );

  // reflection ray
  if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
    PerRayData_radiance refl_prd;
    refl_prd.importance = importance;
    refl_prd.depth = prd_radiance.depth+1;
    float3 R = reflect( ray.direction, ffnormal );
    optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
    rtTrace(top_object, refl_ray, refl_prd);
    color += r * refl_prd.result;
  }

  prd_radiance.result = color;
}
  

//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
