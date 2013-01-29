#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "types.h"
#include <math.h>

#define NUM_SPHERES 6
#define NUM_LIGHTS 1
#define IMG_WIDTH 800
#define IMG_HEIGHT 800
#define BLOCK_DIM 32

void generateScene(sphere_t *spheres, point_light_t *lights)
{
}

void writeImage(color_t *image, const char *filename)
{
}

// returns a float with the t-value of the intersection between a ray
//  and a sphere
__device__ float sphereIntersectionTest(sphere_t s, ray_t r)
{
}

// Calculates the color of a ray which is known to intersect a sphere
__device__ color_t directIllumination(sphere_t s, ray_t r, float t, point_light_t *light)
{
}

// Finds the color of an arbitrary ray
__device__ color_t castRay(ray_t r, sphere_t *s, point_light_t *lights)
{
}

// Takes in a scene and outputs an image
__global__ rayTrace(ray_t *r, sphere_t *s, point_light_t *lights, color_t *image)
{
}

int main(void)
{
   sphere_t *spheres, *dev_spheres;
   point_light_t *lights, *dev_lights;
   ray_t *rays, *dev_rays;
   color_t *image, *dev_image;

   // malloc local arrays
   generateScene(spheres, lights);

   // cudaMalloc dev_ arrays
   // cudaMemcpy to device
   // invoke kernel
   dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
   dim3 dimGrid(IMG_WIDTH/BLOCK_DIM+1, IMG_HEIGHT/BLOCK_DIM+1);
   rayTrace<<<dimGrid, dimBlock>>>(dev_rays, dev_spheres, dev_lights, dev_image);
   // cudaMemcpy from device
   
   // write image to output file
   writeImage(image, "awesome.tga");
}