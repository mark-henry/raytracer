#ifndef __TYPES_H__
#define __TYPES_H__

/* Color struct */
typedef struct color_struct {
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
} sphere;
 
typedef struct vector{
  double x;
  double y; 
  double z; 
} vector_t; 

#endif
