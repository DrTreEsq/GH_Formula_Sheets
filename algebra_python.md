<h2 align="center"> Course 1: Algebra✅ </h2>

#### Toolkit: Numpy Linear Algebra Packages

# Solving a System of Equations
###  Matrices & Vectors | Elimination and Subsitution
###### If there is no matrix in NumPy how do we make one? stacking np arrays!   

1. Matrix * Vector based off System of Equations based of Problem


Matrix_A * (times) Vector_B = Vector_X (the unknown)
|  |  |    | |  *  | |  =  | |
| --- | --- | --- | ---| --- | ---| --- | ---|
| 6  | 1  | 0  | 8  |   | 8  |   | x  |
| 2  | 3  | 4  | 1 |   | 4  |   | x  |
| 3  | 4  | 5 | 9 |   | 2  |   | x  |
| 5  | 6 | 8 | 3  |   | 1  |   | x  |


###### Construct matrix A and vector b corresponding to the system of linear equations

        A = np.array([     
            [2, -1, 1, 1],
            [1, 2, -1, -1],
            [-1, 2, 2, 2],
            [1, -1, 2, 1]    
        ], dtype=np.dtype(float)) 
        b = np.array([6, 3, 14, 8], dtype=np.dtype(float))
        


2. Determinant

| matrix | det formula (The Diagonal - the Antidiagonal) | singularity | rank |
| --- | --- |  --- | --- |
| 2x2  | (ad) - (bc) | x | x |
| 3x3 | (aei) + (bfg) + (cdh) - (afh) - (bdi) - (ceg) | x  | x |
| matrix x matrix (Product) | x | x | x |
| inverse of matrix| x | x  | x |

       d = np.linalg.det(A)
       print(f"Determinant of matrix A: {d:.2f}")




3. Solving the system of Equations
* Elimination / Substitution steps
1. Divide by the coefficient of a to isolate a
2. Plug the value of a into the equation for value b
  
* Systems w/ 3 + variables / 9 equations 🔴 - especially for solving inverses of 3x3 matrices
1. Divide by the coefficient of the first variable
2. xx
3. xxx

###### solution of the system of linear equations with the corresponding coefficients matrix A and free coefficients b

       x = np.linalg.solve(A, b)
       print(f"Solution vector: {x}")

## Performing other Operations

###### what is the difference between a matrix and a vector? A matrix is a combination of vectors which are just columns of scalars!

3. Matrix * Matrix

Matrix A * (times) Matrix B = Matrix C (the unknown)
| |  |    | | *  |  |    |  |  | = |    | |  | |
| --- | --- | --- | ---| --- | --- | --- | ---|  --- | --- | --- | ---| --- | ---|
| 6  | 1  | 0  | 8  |  | 6  | 1  | 0  | 8  | | a  | b  | c  | d  |
| 2  | 3  | 4  | 1 |  | 6  | 1  | 0  | 8  | | e  | f  | g  | h |
| 3  | 4  | 5 | 9 | | 6  | 1  | 0  | 8  |  | i  | j  | k | l |
| 5  | 6 | 8 | 3  | | 6  | 1  | 0  | 8  |  | m  | n | o | p  |


5. Inverse of Matrices

Matrix A (* times) Matrix B of Unknown Variables / The Inverse Matrix = The Correlating Identity Matrix
|  |  |    | | * |  |    | |  | = |    | |  |  |
| --- | --- | --- | ---| --- | --- | --- | ---| --- | --- | --- | ---| --- | --- |
| 6  | 1  | 0  | 8  |  |  a  | b  | c  | d  |  | 1  | 0  | 0  | 0  |
| 2  | 3  | 4  | 1 |  | e  | f  | g  | h |  | 0  | 1 | 0  | 0 |
| 3  | 4  | 5 | 9 | | i  | j  | k | l | | 0  | 0  | 1 | 0 |
| 5  | 6 | 8 | 3  | | m  | n | o | p  | | 0  | 0 | 0 | 1  |

5. Matrix Row Reduction 🔴
6. Row Echelon Form Rules:
* Each pivot is a 1
* Any Number above a pivot is a 0
* Rank of the Matrix is = to the # of pivots
* To get **Reduced Row-echelon form:
* Divide each row by the value of the pivot.
* Turn anything above a pivot to a 0.
7. Dot Product
###### to calculate in NumPy, use np.dot(param_1, param_2)!
8. Vector x Scalar
9. Vector x Vector: (a1 x b1) + (a2 x b2) + (a3 x b3)
10. Distance between vectors
11. Norm (L1 and L2)

## Plotting Linear Transformations with Eigenbasis 🔴
1. Basis rules:

2. Span: (how to determine if points span a certain basis on the plane)

3. Eigenbasis: Span of eigenvectors together

4. Eigenvalue: Solve Character Polynomial to get solutions which are eigenvalues
###### What is a Character Polynomial? It is the product of multiplying the original matrix * the Lambda identity matrix, in which the solutions will give the Eigenvalues.
5. Eigenvector: Solve Systems w/ no Scalars (after multiplying by Eigenvalues)
###### Use the Eigenvalues to multiply the original matrix and this * x/y and solve the system of equations
###### find the eigenvalues and eigenvectors of a defined matrix in NumPy by using np.linalg.eig(Matrix_np_array)

#### 7) Implementing the Neural Network
1. Neural networks have layers that need to be defined
2. The mathematic functions that were computed before need to be called in the algorithm to run for once optimization has run and it has converged with its best prediction.
###### how will you know the math is done properly? Well. 