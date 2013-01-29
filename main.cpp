/*
  CPE 471 Lab 1 
  Base code for Rasterizer
  Example code using B. Somers' image code - writes out a sample tga
  CPE 570 Ray Tracer
  Cecilia Cadenas
*/
// f scalar about -1.25 d is the vector 
#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "types.h"
#include <math.h>
#define NS 6
#define F 1.25

vector_t normalize_V(vector_t vec); 
float dot_vector(vector_t v1, vector_t v2);
int closestSphere(sphere sA[], int index, vector_t D, vector_t e); 

int main(void) {
	
	

   float x,y,t1; //x and y values for the ray tracing and t1 is for the quadratic formula
   float disc,qA,qB,qC; //discriminant and the A,B,C values from the quadratic formula 
   vector_t D = { 0, 0, 0}; 
   vector_t e = {0,0, -1-F}; 
   int closest_I;
  
   color_t blk = {0,0,0,1}; 
   color_t ambK = {.2,.2,.2,1}; //ambient color
   color_t diffK; //diffuse light
   vector_t Ipoint; //point in sphere intersected
   vector_t normal; //normal of sphere at Ipoint

   vector_t reflect;  //reflected vector from the sphere and light
    
  
   vector_t lightp = {-4,4,-8};  //position of the light
   vector_t lightv; 
   color_t diff_light; //diffuse light
   color_t amb_light;  //ambient light
   color_t spec_light; //specular light
   sphere sA[NS]; //needs to be an array 
				
		
  
   sA[0].x = 2; 
   sA[0].y = 2; 
   sA[0].z = 10; 
   sA[0].clr.r = .5; 
   sA[0].clr.g = .5; 
   sA[0].clr.b = .9; 
   sA[0].clr.f = 1; 
   sA[0].radius = 2; 
   
    
   sA[1].x = -3; 
   sA[1].y = 3;  
   sA[1].z = 6; 
   sA[1].clr.r = .9; 
   sA[1].clr.g = .5; 
   sA[1].clr.b = .5; 
   sA[1].clr.f = 1; 
   sA[1].radius = .2;
  
   sA[2].x = 0; 
   sA[2].y = 0; 
   sA[2].z = 4; 
   sA[2].clr.r = .5; 
   sA[2].clr.g = .9; 
   sA[2].clr.b = .5; 
   sA[2].clr.f = 1; 
   sA[2].radius = .2;

   sA[3].x = -4; 
   sA[3].y = .1; 
   sA[3].z = 3; 
   sA[3].clr.r = .9; 
   sA[3].clr.g = .3; 
   sA[3].clr.b = .3; 
   sA[3].clr.f = 1; 
   sA[3].radius = .2; 
   
    
   sA[4].x = .3; 
   sA[4].y = -.5; 
   sA[4].z = 0; 
   sA[4].clr.r = .4; 
   sA[4].clr.g = .7; 
   sA[4].clr.b = .2; 
   sA[4].clr.f = 1; 
   sA[4].radius = .1;
  
   sA[5].x = 0; 
   sA[5].y = -.7; 
   sA[5].z = 0; 
   sA[5].clr.r = .5; 
   sA[5].clr.g = .3; 
   sA[5].clr.b = .8; 
   sA[5].clr.f = 1; 
   sA[5].radius = .045;
 

  

  // make a 800x800 image (allocates buffer on the heap)
  Image img(800,800);
    
	
	for(int i=0;i<800;i++)
    {
		for(int j=0;j<800;j++)
		{    
			img.pixel(i,j,blk); 
		}
	}
    
    for(int i=0;i<800;i++)
    {
       for(int j=0;j<800;j++)
       {    
           x = ((i/800.0)*2)-1;
           y = ((j/800.0)*2)-1;
           int k;
		    
		   D.x = x; 
		   D.y = y; 
		   D.z = F; 
		   
		//   D = normalize_V(D); 
		   
           for(k=0; k<NS; k++)
           {
               
              //go through all spheres and see if one intesects ray   
               qA = dot_vector(D,D);  
               qB = 2*(D.x*(e.x-sA[k].x)+(D.y*(e.y-sA[k].y))+(D.z*(e.z-sA[k].z)));
               qC = (((e.x-sA[k].x)*(e.x-sA[k].x)) + ((e.y-sA[k].y)*(e.y-sA[k].y)) + ((e.z-sA[k].z)*(e.z-sA[k].z))) - pow(sA[k].radius,2); 
         
               disc = (qB*qB) - (4*qA*qC);  
       
               if(disc >= 0)
               {   
				 //  printf("before closest\n");
				  closest_I = closestSphere(sA,k,D,e); 
			 
                  diffK.r = sA[closest_I].clr.r;
                  diffK.g = sA[closest_I].clr.g;
                  diffK.b = sA[closest_I].clr.b;
				//  printf("in main sphere %d r = %f, g = %f b = %f\n", closest_I, diffK.r,diffK.g,diffK.b); 
                  break; 
               }
                
            }
            if(disc >0)
            {
				qA = dot_vector(D,D);  
				qB = 2*(D.x*(e.x-sA[closest_I].x)+(D.y*(e.y-sA[closest_I].y))+(D.z*(e.z-sA[closest_I].z)));
				qC = (((e.x-sA[closest_I].x)*(e.x-sA[closest_I].x)) + ((e.y-sA[closest_I].y)*(e.y-sA[closest_I].y)) + ((e.z-sA[closest_I].z)*(e.z-sA[closest_I].z))) - pow(sA[closest_I].radius,2); 
				
				disc = (qB*qB) - (4*qA*qC);  
				
               //find intersecting point in sphere              
               t1 = (-qB - sqrt(disc))/(2*qA);
              
               Ipoint.x = e.x + t1*(D.x); 
               Ipoint.y = e.y + t1*(D.y);
               Ipoint.z = e.z + t1*(D.z);        
           
               //calculate normal and normalize
				//printf("main: closest_I %d\n", closest_I); 
				normal.x =  Ipoint.x-sA[closest_I].x; 
               normal.y =  Ipoint.y- sA[closest_I].y; 
               normal.z =  Ipoint.z - sA[closest_I].z; 
               normal= normalize_V(normal); 
          
               //calculate light vector and normalize 
               lightv.x = lightp.x - Ipoint.x; 
               lightv.y = lightp.y - Ipoint.y; 
               lightv.z = lightp.z - Ipoint.z;
               lightv = normalize_V(lightv); 
           
               //calculate reflective light
               float dot_reflect = dot_vector(normal, lightv);
			   
               reflect.x = -2*(dot_reflect)*normal.x+lightv.x;
               reflect.y = -2*(dot_reflect)*normal.y+lightv.y; 
               reflect.z = -2*(dot_reflect)*normal.z+lightv.z; 
			   reflect = normalize_V(reflect);  
               
			   if(dot_reflect < 0)
               {
                  dot_reflect = 0; 
               }
               if(dot_reflect > 1)
               {
                  dot_reflect = 1; 
               } 

               diff_light.r = dot_reflect*diffK.r;
               diff_light.g = dot_reflect*diffK.g;       
               diff_light.b = dot_reflect*diffK.b;
           
               //calculate specular lighting
               float dot_spec; 
			   D = normalize_V(D); 	
               dot_spec = dot_vector(D, reflect); 
				    
               if(dot_spec < 0)
               {
                 dot_spec = 0; 
               }
               if(dot_spec > 1)
               {
                  dot_spec = 1; 
               } 
            
               spec_light.r = diffK.r*pow(dot_spec,200); 
			   spec_light.g = diffK.g*pow(dot_spec,200);
               spec_light.b = diffK.b*pow(dot_spec,200);
               //add all lights into sphere color 
               color_t clr; 
               
               clr.r = spec_light.r +  diff_light.r + ambK.r; 
               clr.g = spec_light.g +  diff_light.g + ambK.g; 
               clr.b = spec_light.b +  diff_light.b + ambK.b;
             
         //  clr.r = normal.x*.5+.5; 
		 //  clr.g = normal.y*.5+.5;
        //    clr.b = normal.z*.5+.5;
 
         
               img.pixel(i,j,clr); 
           
          }
        
          else
          {
                             
               img.pixel(i,j,blk); 
           
          }
        }
      }

   

      

  // set a square to be the color above
//  for (int i=50; i < 100; i++) {
//    for (int j=50; j < 100; j++) {
//      img.pixel(i, j, clr);
//    }

  
 
   
  // write the targa file to disk
  img.WriteTga((char *)"awesome.tga", false); 
  // true to scale to max color, false to clamp to 1.0

}



float dot_vector(vector_t v1, vector_t v2)
{
   float dot_value = (v1.x *v2.x) + (v1.y * v2.y) + (v1.z * v2.z); 
   return dot_value; 
}

vector_t normalize_V(vector_t vec)
{
   float mag; 
  
   mag = sqrt((vec.x*vec.x)+(vec.y*vec.y)+(vec.z*vec.z)); 

   vec.x = vec.x/mag; 
   vec.y = vec.y/mag;
   vec.z = vec.z/mag; 

   return vec; 
} 


int closestSphere(sphere sA[], int index, vector_t D, vector_t e)
{
	int closest_I,i;
	float closestz;
	float qA,qB,qC, disc;
	closestz = sA[index].z; 
	closest_I = index;
	int intersects[NS]; 
	i=0;
	for(int k= 0; k<NS; k++)
	{
		
		    qA = dot_vector(D,D);  
		    qB = 2*(D.x*(e.x-sA[k].x)+(D.y*(e.y-sA[k].y))+(D.z*(e.z-sA[k].z)));
		    qC = (((e.x-sA[k].x)*(e.x-sA[k].x)) + ((e.y-sA[k].y)*(e.y-sA[k].y)) + ((e.z-sA[k].z)*(e.z-sA[k].z))) - pow(sA[k].radius,2); 
		
		   disc = (qB*qB) - (4*qA*qC);  
		   
		   if(disc >= 0)
		   {
			   intersects[i] = k; 
			   i++;
		   }
		
	}
	for(int k =0; k<i; k++)
	{
			   
		if(sA[intersects[k]].z <= closestz)
		{
			closestz = sA[intersects[k]].z; 
				
			closest_I = intersects[k]; 
			//printf("in function index %d\n",intersects[k]);
		}
	}
	
		
	
	return closest_I; 
}
	   

		
		
	

	
       









 
