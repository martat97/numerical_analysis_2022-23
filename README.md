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
- The diagonal is always zero

## Assignment 3






