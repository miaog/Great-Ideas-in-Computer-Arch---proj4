// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

#define MIN(a,b) (((a)<(b))?(a):(b)) // got from http://stackoverflow.com/questions/3437404/min-and-max-in-c
#define MAX(a,b) (((a)>(b))?(a):(b)) // got from http://stackoverflow.com/questions/3437404/min-and-max-in-c

float displacement(int dx, int dy)
{
    float squaredDisplacement = dx * dx + dy * dy;
    float displacement = sqrt(squaredDisplacement);
    return displacement;
}

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
    memset(depth, 0, imageHeight * imageWidth * sizeof(float));
    #pragma omp parallel 
    {
    /* The two outer for loops iterate through each pixel */
        #pragma omp for
        for (int y = featureHeight; y < imageHeight-featureHeight; y++) {
            for (int x = featureWidth; x < imageWidth-featureWidth; x++)  {
                /* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
                if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
                    {
                        depth[y * imageWidth + x] = 0;
                        continue;
                    }
                float minimumSquaredDifference = -1;
                float displace = 0;
                int minimumDy = 0;
                int odd = featureWidth % 2;
                int minimumDx = 0;
                float sum[4] = {0,0,0,0};
                int maxX = 2*featureWidth - imageWidth;
                int minX = 2*imageWidth - 2*featureWidth - 1;
                int maxY = 2*featureHeight - imageHeight;
                int minY = 2*imageHeight - 2*featureHeight - 1;
                /* Iterate through all feature boxes that fit inside the maximum displacement box. 
                   centered around the current pixel. */
                for (int dx = MAX(-maximumDisplacement, maxX); dx <= MIN(maximumDisplacement, minX); dx++) {
                    for (int dy = MAX(-maximumDisplacement, maxY); dy <= MIN(maximumDisplacement, minY); dy++) {                        
                    /* Skip feature boxes that dont fit in the displacement box. */
                        if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
                        {
                            continue;
                        }
                        float squaredDifference = 0;
                        int new = 0;
                        __m128 temp = _mm_setzero_ps(); //temporary vector to hold stored stuff
                        __m128 lefty;
                        __m128 righty;
                        __m128 a;
                        __m128 b;
                        if (odd == 0) { //even case
                            for (int boxX = -featureWidth; boxX <= featureWidth; boxX += 4) {
                                for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
                                    if (boxX + 4 > featureWidth) {
                                        new = boxX;
                                        break;
                                    }
                                    else {
                                        lefty = _mm_loadu_ps(((y + boxY)*imageWidth + x + boxX + left));
                                        righty = _mm_loadu_ps(((y + boxY + dy)*imageWidth + x + boxX + dx + right));
                                        a = _mm_sub_ps(lefty, righty);
                                        b = _mm_mul_ps(a, a);
                                        temp = _mm_add_ps(temp, b);
                                    }

                                }
                            }
                            _mm_storeu_ps(sum, temp);
                            squaredDifference += sum[0] + sum[1] + sum[2] + sum[3];
                            for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
                                int leftX = x + new;
                                int leftY = y + boxY;
                                int rightX = x + dx + new;
                                int rightY = y + dy + boxY;
                                float difference = left[leftY*imageWidth+leftX] - right[rightY*imageWidth+rightX];
                                squaredDifference += difference * difference;
                            }
                        }
                        else { //odd case
                            __m128 last = _mm_setzero_ps(); //tail case
                            for (int boxX = -featureWidth; boxX <= featureWidth; boxX += 4) {
                                for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
                                    if (boxX + 4 > featureWidth) {
                                        new = boxX;
                                    } 
                                    else {
                                        lefty = _mm_loadu_ps(((y + boxY)*imageWidth + x + boxX + left));
                                        righty = _mm_loadu_ps(((y + boxY + dy)*imageWidth + x + boxX + dx + right));
                                        a = _mm_sub_ps(lefty, righty);
                                        b = _mm_mul_ps(a, a);
                                        temp = _mm_add_ps(temp, b);
                                    }
                                }
                            }        
                            for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
                                lefty = _mm_loadu_ps(((y + boxY)*imageWidth + x + new + left));
                                righty = _mm_loadu_ps(((y + boxY + dy)*imageWidth + x + new + dx + right));
                                a = _mm_sub_ps(lefty, righty);
                                b = _mm_mul_ps(a, a);
                                last = _mm_add_ps(last, b);                
                            }
                            _mm_storeu_ps(sum, temp);
                            squaredDifference += sum[0] + sum[1] + sum[2] + sum[3];
                            _mm_storeu_ps(sum, last);
                            /*add first 3 elements for tail*/
                            squaredDifference += sum[0] + sum[1] + sum[2];
                        }
                        /* 
                        Check if you need to update minimum square difference. 
                        This is when either it has not been set yet, the current
                        squared displacement is equal to the min and but the new
                        displacement is less, or the current squared difference
                        is less than the min square difference.
                        */
                        if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacement(dx, dy) < displace)) || (minimumSquaredDifference > squaredDifference)) 
                        {
                            minimumSquaredDifference = squaredDifference;
                            minimumDx = dx;
                            minimumDy = dy;
                            displace = displacement(minimumDx, minimumDy);
                        }
                    }
                }
                /* 
                Set the value in the depth map. 
                If max displacement is equal to 0, the depth value is just 0.
                */
                if (minimumSquaredDifference != -1 && maximumDisplacement != 0)
                {
                    depth[y * imageWidth + x] = displace;
                }
                else
                {
                    depth[y * imageWidth + x] = 0;
                }
            }
        }
    }
}