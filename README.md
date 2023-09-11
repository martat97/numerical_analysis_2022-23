# Numerical Analysis Project 2022/23 - Marta Tosolini

## Project description

In this project, we would like to compare the performance of some embarassingly simple algorithms to solve a classification problem based on the MNIST database.
The abstract aim of the program is to write a function:
result = classify(image)
that takes as input a small grey scale image of a hand-written digit (from the MNIST database), and returns the digit corresponding to the content of the image.

## Libraries

**numpy**, **math**, **time**, **matplotlib**, **scipy**, **collections**, **sklearn** libraries were used


## Assignment 1

Numpy operators (np.max, np.sum, np.sqrt, np.abs) were used to define the three distance functions

## Assignment 2

The function **compute_dm** computes the distance matrix of shape (N,N) between the first N entries of x_train.

An empty matrix is created with np.zeros((n,n)), then it is filled by computing a **dist** function (can be d_infty, d_one, d_two) between each pair of images. 

Pay attention that we need to perform the minimum number of operations, so:
- The matrix is symmetrical
- The element in the diagonal is always zero (distance between identical images)

## Assignment 3

**compute_dm** function is applied with n = 100 for each distance. 

Then the function plt.imshow is used to plot each distance matrix, plt.subplot is used to compare the three plots in a single big plot.

## Assignment 4

We need to count the number of failures in **count_failures** function, in the iteration we find the min distance with np.argmin, not considering the value in the diagonal (is zero), so it is temporarily set to np.inf.

If the prediction is wrong, error counter increments by one.
The average error is computed, dividing the number of errors by the length of the distance matrix

## Assignment 5

We need to run the algorithm implemented above for N=100,200,400,800,1600 on the three different distances, and plot the three error rate as a function of N.

So we create an empty matrix with np.zeros((5,3)) (5 different N and 3 distance functions) and we fill it calling count_failures() function at each iteration for each distance function. Elapsed time is computed as well for each n, we notice for n = 1600 that it is taking more than 1 minute, and it was a possibility to precompute a big distance matrix 1600x1600 for each distance, with the distance result in each cell. Then we could compute the error for each n iterating in the big matrix.

The time wasn't that high so it was not necessarily, but the distance matrix will be precomputed in the next assignment.

Then the plot shows the comparison between the average error of each distance, we immediately notice that d_infty is the worst one, and d_two is slightly better than d_one.

## Assignment 6

Each image is interpeted as a continuous function with values between zero and one, defined on a square domain \Omega=[0,27]x[0,27].

So we define the interpolation function for each of the first N = 1600 images, with **interpolate.interp2d** function (kind = 'linear')

H1 is defined, by this steps: 
- compute a,b
- compute (a - b)
- compute gradient for x and y with **gradient** function applied to (a - b)
- the result of the distance is the sum of: square of gradient x, square of gradient y, square of (a - b)
- we need to integrate the result with omega domain
- the final H1 distance is the sqrt of the integral of result.

Precomputing Phase:
- Images are interpolated in a new vector xi_train[1600]
- Integrals of each image are precomputed in a new vector _xi_integrals[1600] (we don't need to compute the integral of an image again in the iteration to build distance matrix)

Distance Matrix Bulding:
- We compute H1 distance of each pair of images, so we need to compute a new integral for each pair of images (remember that the matrix is symmetrical and diagonal is zero)

The problem is that with default epsrel in **integrate.nquad** function, each integral is taking too long, around 2-3 seconds.
With n = 1600, we need to compute more than 1 million integrals, so we're speaking about more than 3 million seconds, definitely too long.
To decrease time, it is a possibility is to increase the tolerance. The integral is less accurate, but the time is lower.

By doing comparisons with a lower n (= 100), the result of the error is the same even computing the integral with tolerance of 100%, so for this algorithm it is not a problem (of course in general it is better to pay attention to not increasing epsrel too much).

The time for each integral is decreased a lot, going from 2-3 seconds, to less than 0.02 seconds

Now each H1 distance is stored in diffH1 matrix of shape 1600x1600, with even higher tolerance of 150%
It took more than 1 hour to compute all the integrals even with higher tolerance, so in the future, parallelizing the operations in threads is a possibility to be considered (was not applied in this problem because it was not sustainable for my pc and 1 hour was still an acceptable time)

Finally, the results for distance H1 are plotted as well to do comparisons with the other distances. The error of H1 is the lower with n = 800, but is similar with d_one, d_two with n = 1600. Of course we need to consider the higher tolerance as a possible explanation of this, but even with that the error is still good.

## Assignment 7

Now n = [3200, 6400], we need to build a Balltree for each distance and for each n. 

From sklearn.neighbors we import BallTree, a package with all the functions we need.

We select the first 3200 images, we apply a reshape from 2D to 1D, then create a tree with BallTree function.

Notice that, precomputing all the first 6400 images reshaped was a possibility, was not applied because was not necessarily, the building of the trees is taking low time. Of course need to be considered if the time is too high.

At the end we obtain a list of 6 trees.

**test_knn** is the function that applies a query to the tree to find the nearest neighbour of each entry in the test dataset (10000 observations). An external library Counter was used to choose the most common category if you choose a k > 1. 

Then we check if the label is correct or if it is different from what we expected, finally we count the errors and compute the average.

We define a new matrix of shape (2,3) (for each distance function and for each n), it is called k1_errors, we have to apply test_knn function with a k = 1. Elapsed time is printed as well, it shows that it increases in respect to n. 

The difference between the errors is plotted for each distance function in respect to the n. d_two seems the best one, d_infty is the worst.

Errors are computed for k = 5 as well, to do comparison between k = 1. Again, the error is better with d_two, and the comparisons between the two different k makes it clear that the algorithm works better with an higher k.





