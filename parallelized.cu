#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "types.h"
#include <math.h>

#define NUM_SPHERES 1
#define NUM_LIGHTS 1
#define IMG_WIDTH 800
#define IMG_HEIGHT 800
#define BLOCK_DIM 32

void generateScene(sphere_t *spheres, point_light_t *lights)
{
   spheres[0].x = 0;
   spheres[0].y = 0;
   spheres[0].z = 0;
   spheres[0].radius = 1;
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

__device__ double dotProduct(vector_t a, vector_t b)
{
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Returns the t-parameter value of the intersection between
//  a ray and a sphere.
// Returns a negative value if the ray misses the sphere.
__device__ double sphereIntersectionTest(sphere_t sphere, ray_t ray)
{
   // For explanation of algorithm, see http://tinyurl.com/yjoup3w
   
   // Transform ray into sphere space
   ray.start.x -= sphere.x;
   ray.start.y -= sphere.y;
   ray.start.z -= sphere.z;

   // We must solve the quadratic equation with A, B, C equal to:
   double A = dotProduct(ray.dir, ray.dir);
   double B = 2*dotProduct(ray.dir, ray.start);
   double C = dotProduct(ray.start, ray.start) -
               sphere.radius * sphere.radius;

   // If the discriminant is negative, the ray has missed the sphere
   double discriminant = B*B - 4*A*C;
   if (discriminant < 0)
      return discriminant;

   // q is an intermediary value in finding the solutions
   double q;
   if (B < 0)
      q = (-B - sqrtf(discriminant))/2.0;
   else
      q = (-B + sqrtf(discriminant))/2.0;

   // Compute the t-values of the intersections
   double t0 = q / A;
   double t1 = C / q;

   // Do a little branch just in case the camera is inside the sphere
   if (t0 > 0 && t1 > 0)
      return min(t0, t1);
   else
      return max(t0, t1);
}

// Calculates the color of a ray which is known to intersect a sphere
__device__ color_t directIllumination(sphere_t sphere, ray_t ray, double t,
                                      point_light_t *light, int num_lights)
{
   color_t illum;
   illum.r = illum.g = illum.b = 1;
   
   return illum;
}

// Finds the color of an arbitrary ray
__device__ color_t castRay(ray_t *ray,
                           sphere_t *spheres, int num_spheres,
                           point_light_t *lights, int num_lights)
{
   color_t bgColor;
   bgColor.r = bgColor.g = bgColor.b = 0;

   // Does this ray intersect with any spheres?
   double t = -1;
   for (int si = 0; si < num_spheres; si++)
      t = max(t, sphereIntersectionTest(spheres[si], *ray));

   if (t > 0) {
      // There was an intersection
      // Set color to bright green for now for debugging purposes
      color_t color;
      color.r = 0;
      color.g = 1;
      color.b = 0;
      return color;
   }
   else
      return bgColor;
}

// Takes in a scene and outputs an image
__global__ void rayTrace(ray_t *rays, int num_rays,
                         sphere_t *spheres, int num_spheres,
                         point_light_t *lights, int num_lights,
                         color_t *pixels, int num_pixels)
{

   int rayIdx = blockDim.x * blockIdx.x + threadIdx.x;
   pixels[rays[rayIdx].pixel] =
      castRay(&rays[rayIdx], spheres, num_spheres, lights, num_lights);
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