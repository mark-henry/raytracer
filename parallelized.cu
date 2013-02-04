#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "types.h"
#include <math.h>

#ifdef DEBUG
   #define NUM_SPHERES 1
   #define NUM_LIGHTS 1
   #define IMG_WIDTH 8
   #define IMG_HEIGHT 8
   #define BLOCK_DIM 1
   #define DRAW_DIST 500
#else
   #define NUM_SPHERES 1
   #define NUM_LIGHTS 1
   #define IMG_WIDTH 512
   #define IMG_HEIGHT 512
   #define BLOCK_DIM 32
   #define DRAW_DIST 500
#endif

void generateScene(sphere_t *spheres, point_light_t *lights)
{
   // At the moment, creates a single test sphere and light

   // Make sphere
   material_t mat;
   mat.diffuse  = (color_t){0.5, 0.1, 0.1};
   mat.specular = (color_t){.5, .4, .8};
   mat.ambient  = (color_t){0.11, 0.1, 0.1};
   mat.shininess = 100;
   
   spheres[0].position = (vector_t){0,0,0};
   spheres[0].radius = 1;
   spheres[0].material = mat;

   // Make light
   lights[0].position = (vector_t){-5, 5, -3};
   lights[0].color = (color_t){1,1,1};
}

// Set up rays based on camera position and image size
void initRays(ray_t *rays, int img_height, int img_width)
{
   ray_t ray;
   vector_t camPos = {0, 0, 10};
   vector_t camLook = {0, 0, -1};
   double camFOVx = tan(3.14159 / 4);
   double camFOVy = tan(camFOVx * img_height/img_width);

   // Iterate over all pixels
   for (int y = 0; y < img_height; y++)
   {
      for (int x = 0; x < img_width; x++)
      {
         // Calculate u,v coordinates of the look vector
         // Keep in mind that pixel (0,0) is in top left, while
         //  (u,v) = (0,0) is in the bottom left
         //double u = (double)x / img_width;
         //double v = (double)(img_height-y-1) / img_height;

         // Cast rays orthogonally along -z for now
         ray.start.x = camPos.x - 1 + 2 * (double)x / img_width;
         ray.start.y = camPos.y - 1 + 2 * (double)(img_height - y) / img_height;
         ray.start.z = camPos.z;
         ray.dir = camLook;
         
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

__device__ double dotProduct(vector_t *a, vector_t *b)
{
   return a->x * b->x + a->y * b->y + a->z * b->z;
}

__device__ double length(vector_t *v) {
   return sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
}

__device__ void normalize(vector_t *v)
{
   double len = length(v);
   v->x /= len;
   v->y /= len;
   v->z /= len;
}

__device__ vector_t reflection(vector_t *in, vector_t *across)
{
   vector_t R;
   double dotProd = dotProduct(in, across);

   R.x = -1 * in->x + 2 * dotProd * across->x;
   R.y = -1 * in->y + 2 * dotProd * across->y;
   R.z = -1 * in->z + 2 * dotProd * across->z;
   
   return R;
}

// Returns the t-parameter value of the intersection between
//  a ray and a sphere.
// Returns a negative value if the ray misses the sphere.
__device__ double sphereIntersectionTest(sphere_t *sphere, ray_t *in_ray)
{
   // For explanation of algorithm, see http://tinyurl.com/yjoup3w
   
   // Transform ray into sphere space
   ray_t ray = *in_ray;
   ray.start.x -= sphere->position.x;
   ray.start.y -= sphere->position.y;
   ray.start.z -= sphere->position.z;

   // We must solve the quadratic equation with A, B, C equal to:
   double A = dotProduct(&ray.dir, &ray.dir);
   double B = 2*dotProduct(&ray.dir, &ray.start);
   double C = dotProduct(&ray.start, &ray.start) -
               sphere->radius * sphere->radius;

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
__device__ color_t directIllumination(sphere_t *sphere, ray_t *ray, double t,
                                      point_light_t *lights, int num_lights)
{
   color_t illum = {0,0,0};

   // inter is the position of the intersection point
   vector_t inter = ray->start;
   inter.x += ray->dir.x * t;
   inter.y += ray->dir.y * t;
   inter.z += ray->dir.z * t;

   // normal is the surface normal at the point of intersection
   vector_t normal = inter;
   normal.x -= sphere->position.x;
   normal.y -= sphere->position.y;
   normal.z -= sphere->position.z;
   normalize(&normal);
   //printf("nomal at (%f,%f,%f) is (%f,%f,%f)\n", inter.x, inter.y, inter.z, normal.x, normal.y, normal.z);

   // V is the eye vector
   vector_t V = ray->dir;
   normalize(&V);

   // Add diffuse and specular for each point_light
   for (int li = 0; li < num_lights; li++) {
      // L is the incident light vector
      vector_t L = lights[li].position;
      L.x -= inter.x;
      L.y -= inter.y;
      L.z -= inter.z;
      normalize(&L);

      // Add ambient
      illum.r += sphere->material.ambient.r * lights[li].color.r;
      illum.g += sphere->material.ambient.g * lights[li].color.g;
      illum.b += sphere->material.ambient.b * lights[li].color.b;

      // Add diffuse
      double dotProd = max(0.0, dotProduct(&normal, &L));
      illum.r += dotProd * sphere->material.diffuse.r * lights[li].color.r;
      illum.g += dotProd * sphere->material.diffuse.g * lights[li].color.g;
      illum.b += dotProd * sphere->material.diffuse.b * lights[li].color.b;

      // Add specular
      vector_t R = reflection(&L, &normal);
      double specDotProd = pow(min(0.0, dotProduct(&V, &R)), sphere->material.shininess);
      illum.r += sphere->material.specular.r * specDotProd * lights[li].color.r;
      illum.g += sphere->material.specular.g * specDotProd * lights[li].color.g;
      illum.b += sphere->material.specular.b * specDotProd * lights[li].color.b;
   }
   
   return illum;
}

// Finds the color of an arbitrary ray
__device__ color_t castRay(ray_t *ray,
                           sphere_t *spheres, int num_spheres,
                           point_light_t *lights, int num_lights)
{
   color_t bgColor = {0,0,0};
   color_t rayColor = bgColor;

   // Does this ray intersect with any spheres?
   double closest = DRAW_DIST;
   double t;
   for (int sphere = 0; sphere < num_spheres; sphere++) {
      t = sphereIntersectionTest(&spheres[sphere], ray);
      if (t > 0 && t < closest) {
         closest = t;
         rayColor = directIllumination(&spheres[sphere], ray, t, lights, num_lights);
      }
   }

   return rayColor;
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