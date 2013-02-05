#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "types.h"
#include "vector_math.h"
#include <math.h>

#define NUM_SPHERES 100
#define SPHERE_RADIUS 1
#define NUM_LIGHTS 1
#define DRAW_DIST 50
#define SCENE_SEED 3

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define BLOCK_DIM 32

void generateScene(sphere_t *spheres, point_light_t *lights, camera_t *camera)
{
   // Generates NUM_SPHERES spheres, randomly placed in the frustum of the camera
   //  from DRAW_DIST/4 out to DRAW_DIST

   // The camera is at the origin looking along +Z
   camera->position = (vector_t){0,0,-2};
   camera->look = (vector_t){0,0,1};
   camera->up = (vector_t){0,1,0};
   
   srand(SCENE_SEED);

   vector_t pos;
   material_t mat;
   double spread = 1.5;
   int zval;
   for (int i = 0; i < NUM_SPHERES; i++)
   {
      zval = DRAW_DIST - pow(rand() % (int)sqrt(DRAW_DIST), 2);
      pos.z = zval;
      pos.x = (rand() % (zval+1) - (zval / 2)) * spread;
      pos.y = (rand() % (zval+1) - (zval / 2)) * spread;

      mat.diffuse.r = (rand() % 100) / 100.0;
      mat.diffuse.g = (rand() % 100) / 100.0;
      mat.diffuse.b = (rand() % 100) / 100.0;
      
      mat.specular.r = (rand() % 100) / 100.0;
      mat.specular.g = (rand() % 100) / 100.0;
      mat.specular.b = (rand() % 100) / 100.0;
      
      mat.ambient = (color_t){.05, .05, .05};
      
      mat.shininess = pow((rand() % 9) + 2, 2);

      spheres[i].position = pos;
      spheres[i].radius = 1;
      spheres[i].material = mat;
   }

   // Place a single light in the middle of the spheres
   lights[0].position = (vector_t){0,0,DRAW_DIST/3};
   lights[0].color = (color_t){1,1,1};
}

// Set up rays based on camera position and image size
void initRays(ray_t *rays, camera_t camera,
              int img_height, int img_width)
{
   ray_t ray;
   double aspectRatio = (double)img_height / img_width;
   vector_t rightShift, upShift;
   double u, v;
   vector_t right;

   normalize(&camera.look);
   normalize(&camera.up);
   right = cross(camera.look, camera.up);
   
   // Iterate over all pixels
   for (int y = 0; y < img_height; y++) {
      for (int x = 0; x < img_width; x++)
      {
         // Calculate ray direction
         u = aspectRatio * x / img_width * 2 - 1;
         v = -((double)y / img_height * 2 - 1);
         rightShift = multiply(right, u);
         upShift = multiply(camera.up, v);

         ray.start = camera.position;
         ray.dir = add(camera.look, add(rightShift, upShift));

         // Which pixel in the array does this ray correspond to?
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

// Returns the t-parameter value of the intersection between
//  a ray and a sphere.
// Returns a negative value if the ray misses the sphere.
__device__ double sphereIntersectionTest(sphere_t *sphere, ray_t *in_ray)
{
   // For explanation of algorithm, see http://tinyurl.com/yjoup3w
   
   // Transform ray into sphere space
   ray_t ray = *in_ray;
   ray.start = subtract(ray.start, sphere->position);

   // We must solve the quadratic equation with A, B, C equal to:
   double A = dotProduct(ray.dir, ray.dir);
   double B = 2*dotProduct(ray.dir, ray.start);
   double C = dotProduct(ray.start, ray.start) -
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

// Helper function for illumination calculations
inline __device__ void addLightingFactor(color_t *illum, color_t material, color_t light)
{
   illum->r += material.r * light.r;
   illum->g += material.g * light.g;
   illum->b += material.b * light.b;
}

// Calculates the color of a ray which is known to intersect a sphere
__device__ color_t directIllumination(sphere_t *sphere, ray_t *ray, double t,
                                      point_light_t *lights, int num_lights)
{
   color_t illum = {0,0,0};

   // inter is the position of the intersection point
   vector_t inter = add(ray->start, multiply(ray->dir, t));

   // normal is the surface normal at the point of intersection
   vector_t normal = subtract(inter, sphere->position);
   normalize(&normal);

   // V is the eye vector
   vector_t V = ray->dir;
   normalize(&V);

   // Add diffuse and specular for each point_light
   for (int li = 0; li < num_lights; li++) {
      // L is the incident light vector
      vector_t L = subtract(lights[li].position, inter);
      normalize(&L);

      // Add ambient
      addLightingFactor(&illum, sphere->material.ambient, lights[li].color);

      // Add diffuse
      double dotProd = max(0.0, dotProduct(normal, L));
      addLightingFactor(&illum, multiply(sphere->material.diffuse, dotProd),
                        lights[li].color);

      // Add specular
      vector_t R = reflection(L, normal);
      double specDotProd = pow(min(0.0, dotProduct(V, R)), sphere->material.shininess);
      addLightingFactor(&illum, multiply(sphere->material.specular, specDotProd),
                        lights[li].color);
   }

   illum.r = min(illum.r, 1.0);
   illum.g = min(illum.g, 1.0);
   illum.b = min(illum.b, 1.0);
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
   camera_t camera;

   int spheres_size = NUM_SPHERES * sizeof(sphere_t);
   int lights_size  = NUM_LIGHTS * sizeof(point_light_t);
   int rays_size  = IMG_HEIGHT*IMG_WIDTH*sizeof(ray_t);
   int image_size = IMG_HEIGHT*IMG_WIDTH*sizeof(color_t);
   
   spheres = (sphere_t *)      malloc(spheres_size);
   lights  = (point_light_t *) malloc(lights_size);
   rays    = (ray_t *)         malloc(rays_size);
   image   = (color_t *)       malloc(image_size);
   
   generateScene(spheres, lights, &camera);

   initRays(rays, camera, IMG_WIDTH, IMG_HEIGHT);

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