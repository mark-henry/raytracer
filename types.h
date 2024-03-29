#pragma once

/* Color struct */
typedef struct color {
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"
} color_t;

typedef struct material {
   color_t diffuse;
   color_t specular;
   color_t ambient;
   int shininess;
} material_t;

typedef struct vector{
  double x;
  double y; 
  double z; 
} vector_t;

typedef struct sphere {
  vector_t position;
  material_t material;
  double radius;
} sphere_t;

typedef struct ray {
   vector_t start;
   vector_t dir;
} ray_t;

typedef struct point_light {
   vector_t position;
   color_t color;
} point_light_t;

typedef struct camera {
   vector_t position;
   vector_t look;
   vector_t up;
} camera_t;
