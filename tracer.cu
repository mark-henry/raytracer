#include <stdlib.h>
#include <stdio.h>
#include "types.h"
#include "vector_math.h"
#include <math.h>

#define NUM_SPHERES 100
#define SPHERE_RADIUS 1

#define NUM_LIGHTS 2
#define LIGHT_FALLOFF_BRIGHTNESS 4
#define LIGHT_FALLOFF_QUADRATIC 0
#define LIGHT_FALLOFF_LINEAR .5
#define SHADOWS

#define SCENE_SEED 10
#define SCENE_FRONT 7
#define SCENE_BACK 40

#define DRAW_DIST 200
#define BG_COLOR 0,0,0

#ifdef DEBUG
   #define IMG_WIDTH 8
   #define IMG_HEIGHT 4
   #define BLOCK_DIM 1
#else
   #define IMG_WIDTH 1024
   #define IMG_HEIGHT 1024
   #define BLOCK_DIM 32
#endif

void generateScene(sphere_t *spheres, point_light_t *lights)
{
   // Generates NUM_SPHERES spheres, randomly placed in the frustum of the camera
   //  from SCENE_FRONT to SCENE_BACK
   
   srand(SCENE_SEED);

   vector_t pos;
   material_t mat;
   double spread = 1.2;
   int zval;
   for (int i = 0; i < NUM_SPHERES; i++)
   {
      zval = (rand() % (SCENE_BACK-SCENE_FRONT)) + SCENE_FRONT;
      pos.z = zval;
      pos.x = (rand() % (zval+1) - (zval / 2)) * spread;
      pos.y = (rand() % (zval+1) - (zval / 2)) * spread;

      mat.diffuse.r = (rand() % 100) / 100.0;
      mat.diffuse.g = (rand() % 100) / 100.0;
      mat.diffuse.b = (rand() % 100) / 100.0;

      mat.specular.r = (rand() % 100) / 100.0;
      mat.specular.g = (rand() % 100) / 100.0;
      mat.specular.b = (rand() % 100) / 100.0;

      mat.ambient = (color_t){.07, .07, .07};

      mat.shininess = pow((rand() % 10) + 2, 2);

      spheres[i].position = pos;
      spheres[i].radius = 1;
      spheres[i].material = mat;
   }

   // Place a single light in the middle of the spheres
   lights[0].position.x = 0;
   lights[0].position.y = -4;
   lights[0].position.z = SCENE_FRONT;
   lights[0].color = (color_t){1,1,1};

   // Place another one for fill lighting
   #if NUM_LIGHTS == 2
      lights[1].position.x = -SCENE_BACK;
      lights[1].position.y = 2;
      lights[1].position.z = 2*SCENE_BACK;
      lights[1].color = (color_t){1,1,1};
   #endif
}

// Returns the t-parameter value of the intersection between
//  a ray and a sphere.
// Returns a negative value if the ray misses the sphere.
__device__ double sphereIntersectionTest(sphere_t *sphere, ray_t ray)
{
   // For explanation of algorithm, see http://tinyurl.com/yjoup3w
   
   // Transform ray into sphere space
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
inline __device__ void applyLightingFactor(color_t *illum, color_t material, color_t light)
{
   // Mults are commented out to optimize
   illum->r += material.r;// * light.r;
   illum->g += material.g;// * light.g;
   illum->b += material.b;// * light.b;
}

// Tests if any spheres lie along the line segment
//  from (start) to (start+ray)
__device__ bool isInShadow(vector_t start, vector_t ray, sphere_t *spheres,
                           int num_spheres, sphere_t *ignore_sphere)
{
   ray_t testray;
   testray.start = start;
   testray.dir = ray;
   
   double distance = length(ray);

   for (int si = 0; si < num_spheres; si++) {
      if (&spheres[si] == ignore_sphere)
         continue;
      double t = sphereIntersectionTest(&spheres[si], testray);
      if (t > 0 && t <= distance)
         return true;
   }

   return false;
}

// Calculates the color of a ray which is known to intersect a sphere
__device__ color_t directIllumination(sphere_t *sphere, sphere_t *spheres,
                                      int num_spheres,
                                      ray_t *ray, double t,
                                      point_light_t *lights, int num_lights)
{
   // Start with ambient light
   color_t illum = sphere->material.ambient;

   // inter is the position of the intersection point
   vector_t inter = add(ray->start, multiply(ray->dir, t));

   // normal is the surface normal at the point of intersection
   vector_t normal = subtract(inter, sphere->position);
   normalize(&normal);

   // V is the eye vector
   vector_t V = multiply(ray->dir, -1);
   normalize(&V);

   // For each point_light, add diffuse and specular
   for (int li = 0; li < num_lights; li++)
   {
      // L is the incident light vector
      vector_t L = subtract(lights[li].position, inter);

      #ifdef SHADOWS
         // Skip this light if something is in the way
         if (isInShadow(inter, L, spheres, num_spheres, sphere))
            continue;
      #endif

      // Calculate light falloff with distance
      double distance = length(L);
      double distanceFactor = LIGHT_FALLOFF_BRIGHTNESS /
         (LIGHT_FALLOFF_LINEAR * distance);

      normalize(&L);

      // Add diffuse
      double diffFactor = distanceFactor * max(0.0, dotProduct(normal, L));
      applyLightingFactor(&illum, multiply(sphere->material.diffuse, diffFactor),
                        lights[li].color);

      // Add specular
      vector_t R = reflection(L, normal);
      double specFactor = distanceFactor *
         pow(max(0.0, dotProduct(V, R)), sphere->material.shininess);
      applyLightingFactor(&illum, multiply(sphere->material.specular, specFactor),
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
   color_t bgColor = {BG_COLOR};
   color_t rayColor = bgColor;

   // Does this ray intersect with any spheres?
   double closest = DRAW_DIST;
   double t;
   for (int sphere = 0; sphere < num_spheres; sphere++) {
      t = sphereIntersectionTest(&spheres[sphere], *ray);
      if (t > 0 && t < closest) {
         closest = t;
         rayColor = directIllumination(&spheres[sphere], spheres, num_spheres, ray, t, 
               lights, num_lights);
      }
   }

   return rayColor;
}

// Takes in a scene and outputs an image
__global__ void rayTrace(camera_t camera,
                         uchar4 *pixels, int img_width, int img_height,
                         sphere_t *spheres, int num_spheres,
                         point_light_t *lights, int num_lights)
{

   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx >= img_width * img_height)
      return;
   
   // Initialize this thread's ray
   ray_t ray;
   int x = idx % img_width;
   int y = idx / img_height;
   double aspectRatio = (double)img_width / img_height;
   vector_t rightShift, upShift;
   vector_t right;
   double u, v;

   normalize(&camera.look);
   normalize(&camera.up);
   right = cross(camera.look, camera.up);

   u = aspectRatio * x / img_width * 2 - 1;
   v = -((double)y / img_height * 2 - 1);
   rightShift = multiply(right, u);
   upShift = multiply(camera.up, v);

   ray.start = camera.position;
   ray.dir = add(camera.look, add(rightShift, upShift));

   color_t pixel =
      castRay(&ray, spheres, num_spheres, lights, num_lights);
   pixels[idx].x = pixel.r * 255;
   pixels[idx].y = pixel.g * 255;
   pixels[idx].z = pixel.b * 255;
   pixels[idx].w = pixel.f * 255;
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }
} 

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* image, unsigned int img_width,
                  unsigned int img_height, float time)
{
   sphere_t *spheres, *dev_spheres;
   point_light_t *lights, *dev_lights;
   camera_t camera;

   int spheres_size = NUM_SPHERES * sizeof(sphere_t);
   int lights_size  = NUM_LIGHTS * sizeof(point_light_t);
   
   spheres = (sphere_t *)      malloc(spheres_size);
   lights  = (point_light_t *) malloc(lights_size);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   generateScene(spheres, lights);

   // Set up camera based on time param
   camera.position = (vector_t){2*cos(time/2), 0, 0.2*sin(time)};
   camera.look = (vector_t){0,0,1};
   camera.up = (vector_t){0,1,0};
   
   // cudaMalloc dev_ arrays
   cudaMalloc(&dev_spheres, spheres_size);
   cudaMalloc(&dev_lights, lights_size);

   // cudaMemcpy the problem to device
   cudaMemcpy(dev_spheres, spheres, spheres_size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_lights, lights, lights_size, cudaMemcpyHostToDevice);
   
   // Invoke kernel
   int dimBlock = BLOCK_DIM;
   int dimGrid = (img_height*img_width) / BLOCK_DIM;
   rayTrace<<<dimGrid, dimBlock>>>(camera, image,
                                   img_width, img_height,
                                   dev_spheres, NUM_SPHERES,
                                   dev_lights, NUM_LIGHTS);

   cudaThreadSynchronize();
   checkCUDAError("kernel failed!");
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   printf("Time in tracer: %.1f ms\n", elapsedTime);
   
   // Free memory
   free(spheres);
   free(lights);
   cudaFree(dev_spheres);
   cudaFree(dev_lights);
}
