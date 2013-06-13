#include <cstdio>
#include <cstdlib>
#define sqr(x)	((x)*(x))

float dboard(int darts) {
	int score = 0;
	
	/* "throw darts at board" */
	for (int n = 0; n < darts; n++)  {
		/* generate random numbers for x and y coordinates */
		float r = (float)rand() / (RAND_MAX+1.0);
		float x_coord = (2.0 * r) - 1.0;
		r = (float)rand() / (RAND_MAX+1.0);
		float y_coord = (2.0 * r) - 1.0;
		
		/* if dart lands in circle, increment score */
		if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)
			score++;
    }
	
	/* return our approximation of pi */
	return 4.0 * (float)score/(float)darts;
} 
