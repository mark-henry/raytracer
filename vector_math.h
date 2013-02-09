#include "types.h"

inline __device__ __host__ vector_t multiply(vector_t v, double coeff)
{
   v.x *= coeff;
   v.y *= coeff;
   v.z *= coeff;
   return v;
}

inline __device__ __host__ color_t multiply(color_t c, double coeff)
{
   c.r *= coeff;
   c.g *= coeff;
   c.b *= coeff;
   return c;
}

inline __device__ __host__ vector_t add(vector_t a, vector_t b)
{
   a.x += b.x;
   a.y += b.y;
   a.z += b.z;
   return a;
}

inline __device__ __host__ vector_t subtract(vector_t a, vector_t b)
{
   a.x -= b.x;
   a.y -= b.y;
   a.z -= b.z;
   return a;
}

inline __device__ __host__ double dotProduct(vector_t a, vector_t b)
{
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ __host__ vector_t cross(vector_t a, vector_t b)
{
   vector_t cross;
   cross.x = a.y*b.z - a.z*b.y;
   cross.y = a.z*b.x - a.x*b.z;
   cross.z = a.x*b.y - a.y*b.x;
   return cross;
}

inline __device__ __host__ double length(vector_t v)
{
   return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline __device__ __host__ void normalize(vector_t *v)
{
   double len =  1 / length(*v);
   v->x *= len;
   v->y *= len;
   v->z *= len;
}

inline __device__ __host__ vector_t reflection(vector_t in,
                                               vector_t across)
{
   vector_t R;
   double dotProd = dotProduct(in, across);

   R.x = 2 * dotProd * across.x - in.x;
   R.y = 2 * dotProd * across.y - in.y;
   R.z = 2 * dotProd * across.z - in.z;
   
   return R;
}