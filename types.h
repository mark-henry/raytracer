#ifndef __TYPES_H__
#define __TYPES_H__

/* Color struct */
typedef struct color {
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"
} color_t;

typedef struct sphere {
  double x; 
  double y; 
  double z; 
  color_t clr; 
  double radius;
} sphere_t;
 
typedef struct vector{
  double x;
  double y; 
  double z; 
} vector_t;

typedef struct ray {
   vector_t start;
   vector_t dir;
   int pixel_index_x;
   int pixel_index_y;
} ray_t;

typedef struct point_light {
   vector_t location;
   color_t color;
} point_light_t;

#endif
