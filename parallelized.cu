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

// Set up rays based on camera position and image size
void initRays(ray_t *rays, int img_height, int img_width)
{
   ray_t ray;
   
   for (int y = 0; y < img_height; y++)
   {
      for (int x = 0; x < img_width; x++)
      {
         // TODO: math
         ray.pixel = y*img_width + x;
         rays[y*img_width + x] = ray;
      }
   }
}

void writeImage(char *filename, color_t *image, int width, int height)
{
   Image img(width, height);

   // Copy image to Image object
   // Image is weird: (0,0) is the lower left corner
   for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
         img.pixel(x, height-y-1, image[width*y + x]);
   
   img.WriteTga(filename, false);
}

// returns a float with the t-value of the intersection between a ray
//  and a sphere
__device__ float sphereIntersectionTest(sphere_t s, ray_t r)
{
   return -1;
}

// Calculates the color of a ray which is known to intersect a sphere
__device__ color_t directIllumination(sphere_t s, ray_t r, float t, point_light_t *light)
{
   color_t illum;
   illum.r = illum.g = illum.b = 1;
   
   return illum;
}

// Finds the color of an arbitrary ray
__device__ color_t castRay(ray_t r, sphere_t *s, point_light_t *lights)
{
   color_t color;
   // Set color to bright green for now for debugging purposes
   color.r = 0;
   color.g = 1;
   color.b = 0;
   color.f = 1;
   
   return color;
}

// Takes in a scene and outputs an image
__global__ void rayTrace(ray_t *rays, int num_rays,
                         sphere_t *spheres, int num_spheres,
                         point_light_t *lights, int num_lights,
                         color_t *pixels, int num_pixels)
{
   color_t bgColor;
   bgColor.r = bgColor.g = bgColor.b = 0;

   int index = blockDim.x * blockIdx.x + threadIdx.x;
   pixels[rays[index].pixel] = bgColor;
}

int main(void)
{
   sphere_t *spheres, *dev_spheres;
   point_light_t *lights, *dev_lights;
   ray_t *rays, *dev_rays;
   color_t *image, *dev_image;

   int spheres_size = NUM_SPHERES * sizeof(sphere_t);
   int lights_size  = NUM_LIGHTS * sizeof(point_light_t);
   int rays_size  = IMG_HEIGHT*IMG_WIDTH*sizeof(ray_t);
   int image_size = IMG_HEIGHT*IMG_WIDTH*sizeof(color_t);
   
   spheres = (sphere_t *)      malloc(spheres_size);
   lights  = (point_light_t *) malloc(lights_size);
   rays    = (ray_t *)         malloc(rays_size);
   image   = (color_t *)       malloc(image_size);
   
   generateScene(spheres, lights);
   initRays(rays, IMG_WIDTH, IMG_HEIGHT);

   // cudaMalloc dev_ arrays
   cudaMalloc(&dev_spheres, spheres_size);
   cudaMalloc(&dev_lights, lights_size);
   cudaMalloc(&dev_rays, rays_size);
   cudaMalloc(&dev_image, image_size);
   
   // cudaMemcpy the problem to device
   cudaMemcpy(dev_spheres, spheres, spheres_size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_lights, lights, lights_size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rays, rays, rays_size, cudaMemcpyHostToDevice);
   
   // Invoke kernel
   int dimBlock = BLOCK_DIM;
   int dimGrid = (IMG_HEIGHT*IMG_WIDTH) / BLOCK_DIM;
   rayTrace<<<dimGrid, dimBlock>>>(dev_rays, IMG_HEIGHT*IMG_WIDTH,
                                   dev_spheres, NUM_SPHERES,
                                   dev_lights, NUM_LIGHTS,
                                   dev_image, IMG_HEIGHT*IMG_WIDTH);

   // cudaMemcpy the result image from the device
   cudaMemcpy(image, dev_image, image_size, cudaMemcpyDeviceToHost);
   
   // write image to output file
   writeImage("awesome.tga", image, IMG_WIDTH, IMG_HEIGHT);

   // Free memory
   free(spheres);
   free(lights);
   free(rays);
   free(image);
   cudaFree(dev_spheres);
   cudaFree(dev_lights);
   cudaFree(dev_rays);
   cudaFree(dev_image);

   return 0;
}