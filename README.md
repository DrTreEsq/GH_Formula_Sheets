<h1 align="center"> Formula Sheet for Analytical and Computational 𝐌𝑎𝑇𝐇!</h1>

<h6 align="center"> The Code from Completion of Programming Assignments from courses in Mathematics for Machine Learning and Data Science by DeepLearning.AI</h6>

<h6 align="center">  the Formula sheet below is HEAVILY in progress  </h6>

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


<h2 align="center"> Course 2: Calculus✅</h2>

## Derivatives
### Derivative Notation
###### 𝚫dx  - change in x direction
##### _______
###### 𝚫dt - changes in the t direction
1. Lagrange: f'(x)
######  The function is f and at the point x the slope of the line is f'(x), that is like Lagrange's notation.
2. Leibenz: dy/dx or d/dx(f(x))
###### And the other one is the one you just saw, which is dy over dx, that can also be expressed as d over dx of f(x). In this case d over dx can be thought of as an operator, something that when you apply to f(x) you get something else, and this is Leibniz's notation. 

### Formulas for Derivatives🔴
1. Derivative of a Constant

###### Since the derivative of a function represents the slope of the function, the derivative of a constant function must be equal to its slope of zero. This gives you the first derivative rule – the Constant Rule. If f(x) = k, where k is any real number, then the derivative is equal to zero.

###### Learn more about this from: https://www.alamo.edu/contentassets/20691fef0c254307b473b980bb6648fb/differentiation/math1325-basic-rules-of-derivatives.pdf

2. Derivative of a Constant (number) * Variable

###### the derivative of the scalar * the derivative of the Variable will always be zero because the derivative of a constant (number) is zero.

3. Derivative of a Function x^n
##### To find the derivative of function x^n , take nx^n-1
##### for example: the derivative of x^4 = 4x^3

### Common Derivatives
1. Constants / Lines / Distance

###### slope = 𝚫x / 𝚫t
3. Linear Functions
4. Polynomial Quadratic Functions

###### slope = 𝚫f / 𝚫x = (x + 𝚫x)^2-(x)^2 / 𝚫x

5. Higher Degree Polynomials

###### slope = 𝚫f / 𝚫x = (x + 𝚫x)^3-(x)^3 / 𝚫x
7. Exponential and Logarithmic Functions


##### The Exponential e

7. Trigonometric Functions
8. The Inverse Function

### Rules of Derivative
1. Chain Rule: f(g(x)) =  g′(x)⋅f(g(x))

### Other Operations
1. Multiplication by Scalars /  Constants
### Plotting - Maxima and Minima

## Multivariable Calculus

##### check out some resources below that go a little more in-depth. Khan academy is a good secondary resource to Coursera! 
###### https://www.youtube.com/watch?v=JAf_aSIJryg

### Partial Derivatives

### Gradients

## Gradient Descent for Machine Learning Optimization

## Newton's Method
###### an alternative to Gradient Descent that finds the Zeros of the function which become the candidates for minimum and maximum points.

###### pick the best model for the training data

<h2 align="center"> Course 3: Probability & Statistics  - in Progress😰 </h2>

## Basics

1. Measures of Central Tendency
2. Discrete vs Continuous:
3. Independent vs Dependent:
4. Joint and Disjoint Events:
5. Conditional Probability vs Pure Probability:
6. Expected Value
7. Variance and Covariance


## Rules / Formulas
##### Complement Rule:
P(A’) = 1-P(A)


##### Product Rule (for Independent Events)
P(A  B) = P(A) * P(B)

##### Product Rule for Independent events – Conditional Probability
P(A  B) =  P(A) * P(B | A)

## Bayes Theorem

### Bayes Theorem Formula
P(A) * P(B | A) /
P(A) + P(B | A) +  P(A’)* P(B | A’)

## PROBABILITY DISTRIBUTIONS

#### Probability Mass Function:
###### Each bar represents the probability that the variable X3 takes each of the possible values (0,1,2,3,4,and 5). Now, for each x from 0 to 5, you have the probability that X3 is X. This is called the Probability Mass Function. All discrete random variables can be modeled by their probability mass function (PMF), since it contains all the necessary information to understand how the probability distributes among all the possible values of the variable 

1. Since it’s defined as the probability that the random variable takes a particular value, then it always has to be positive.
2. When you sum the PMF over all possible values, they all sum up to 1. This makes sense, since you’re considering the probability of all possible outcomes of the experiment.

#### Probability Density Function
##### Usually denoted as lowercase f, and it is the equivalent of the lowercase p in the discrete distribution. The equivalent of the mass function is now called Probability Density Function. For clarity, you can add a subscript of the variable it’s representing – the capital X. (fX).
###### PDF’s are a function defined only for continuous variables and it represents the rate at which you accumulate probability around each point. You can use the PDF to calculate probabilities by getting the area under the PDF curve between points A and B.
A function needs to satisfy in order for it to be considered a PDF:
1.	Defined for all number in the real line. That means that is can actually be 0 for many values, but it doesn’t need to b/c it could be positive for all the numbers and the area still being 1 if it gets really, really, tiny at the tails.
2.	Needs to be positive or 0 for all values. This is reasonable because otherwise it would get placed in negative probabilities, and probabilities cannot be negative.
3.	The area under the curve has to be 1. This restriction comes from the probabilities. The area under the curve is simply the probability of all possible outcomes of the variable and it always needs to add to 1.

#### Cumulative Distribution Function
##### The CDF is a Function that shows how much probability the variable has accumulated until a certain value. So the CDF is actually the probability that your random variable x is smaller than or equal to some value x. This is defined for each and every value on the real line from -∞ to + ∞. This function is normally denoted with a Capital F. Exactly as with PDF’s, we add a subscript with the random variable for extra clarity – in this case it’s Capital X.
###### Note: please be mindful not to use lowercase f because that’s the one reserved for the probability density functions of continuous distributions.

CDF needs to satisfy these properties:
1.	By definition, it’s a probability, all values have to be between 0 and 1.
2.	The left point of the CDF needs to be 0. In some cases it’s a fixed number, sometimes it’s n - ∞, but nonetheless it has to be 0.
3.	The right endpoint where all the probability has been accumulated, the CDF has to be 1.
4.	The CDF can never decrease since it’s accumulating probabilities and probabilities can never be negative.

#### Uniform Distribution Function
##### Uniform PDF
In general, a continuous variable is set to follow a continuous distribution if all possible values have the same frequency of occurrence, it has two parameters which are associated to the interval where the variable can take values a and b, and the PDF is defined notated with fx(x).
a is the beginning of the interval and b will be the end of it.
When the interval a to b is bigger, than the PDF is smaller / the height of the PDF is smaller and it quickly increases at the interval gets short.


##### Uniform CDF

^Consider some point X < 0. Therefore, for any value that is < 0, the CDF is 0.
^Now, take any x between 0 and 1. The probability of the variable being smaller than a particular x is the area here – the rectangle.

^You need to multiply the length of the base, which is x, by the height, which is 1. At that point, the CDF is x/1, which is the same as x. The same is valid for any x bigger than 1, you’ve already gathered all the area under the PDF, so the probability stays as 1.
^The CDF is a straight line, followed by a diagonal and then followed by a straight line. The one on the left is 0 and the one on the right is 1.

^For the general Uniform Distribution between a and b, for x less than a, you have 0 cumulative probability. Again, for bigger than b, you’ve already gathered all the probability, so therefore, the CDF is 1 after that point.


#### Normal (Gaussian) Distribution Function

	Mu is the center of the data / of the bell.
	Sigma is the spread of the bell / measure of the wideness.
	The curve is symmetrical
	The range is all the real numbers
•	Note: this is one example of a distribution which its probability density function is always positive, although it’s very small for bit anf for big negative numbers

1.	First thing we do instead of x, is consider x – 2, and that moves it all the way to the center.
2.	Divide by the standard deviation / sigma, which puts it at the right height and width and it makes it the standard normal with parameters 0.1
a.	we’re normally going to want parameters 0, 1 as the mean and one as the standard deviation
b.	So not looking at x, but instead looking at Z = X – Mu/ sigma

^Standard deviation is crucial in statistics because it helps us compare variables of different magnitudes. So if one variable, for example, moves in some range of values and another variable moves in a completely different range of values – then you can compare then by standardizing both of them. 

###### The CDF looks like usual, starting at 0 on the very left, which is the point - ∞ and ends at the point + ∞. But here’s a catch, this area is actually really hard to compute. It can be computed analytically.

#### Chi-Squared Distribution

The big question is now, what is the distribution of W? For simplicity, going to assume that Z follows a standard normal distribution with parameter 0 and 1. The probability is the area under the PDF curve of the Gaussian between these two numbers. You can get the CDF for W by finding these areas for each possible value W. Notice that for small values of W, you gain area at a much quicker rate. This is because the Gaussian distribution concentrates probability around 0. This is known as the Chi-squared distribution with 1 degree of Freedom.![image]

Since the CDF is the integral of the PDF, then you can easily find the PDF by taking the derivative of the CDF which is the slope of the CDF of each point. The rate at which the probability is accumulated it is big for small values of w, and gets smaller as w increases. Because for small values of w, it is a very steep curve that grows really quickly but does the opposite for larger values, growing slower and slower.


<h2 align="center"> END OF FORMULA SHEET </h2>

#### other resources
* Linear Algebra and Its Applications by Gilbert Strang
* Elementary Linear Algebra, 8th Edition by Ron Larson 
* JSX: https://jsxgraph.uni-bayreuth.de/wp/index.html
* Bootstrap: https://getbootstrap.com/docs/5.0/getting-started/introduction/
* Three js: https://threejs.org/
* CONED Mathbox: https://github.com/unconed/mathbox
* Khan Academy Algebra: https://www.khanacademy.org/math/algebra
* Deep Learning AI Community: https://community.deeplearning.ai/c/math-for-machine-learning/m4ml-course-2/m4ml-course-2-week-2/310
* derivative plotting: https://www.mathsisfun.com/calculus/derivative-plotter.html
* maximum and minimum value: https://study.com/learn/lesson/how-to-find-the-maximum-value-of-a-function.html#:~:text=We%20will%20set%20the%20first,will%20be%20a%20minimum%20value.
* max and min (quadratic) https://www.wikihow.com/Find-the-Maximum-or-Minimum-Value-of-a-Quadratic-Function-Easily
* solving linear equations: https://www.wikihow.com/Solve-Multivariable-Linear-Equations-in-Algebra
* JAX: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
* NumPy Constants: https://www.programiz.com/python-programming/numpy/constants
* ski-kit-learn: https://scikit-learn.org/stable/
* Gaussian Elimination & Row Echelon Form: https://www.youtube.com/watch?v=eDb6iugi6Uk
* Probability and Statistics (4th Edition) , Morris H. DeGroot, Mark J. Schervish,Pearson, 2011
* All of Statistics: A Concise Course in Statistical Inferenceby Larry WassermanSpringer, 2010 
* Probabilistic Machine Learning: An Introductionby Kevin Patrick Murphy. MIT Press, March 2022.
* https://jsxgraph.uni-bayreuth.de/wp/index.html
* https://getbootstrap.com/docs/5.0/getting-started/introduction/

#### troubleshooting tips - common errors when computing mathematic code
* unsupported operand type(s) for *: 'float' and 'function' : trying to carry out mathematic function on the wrong type, so use appropriate NumPy operations such as np.dot - https://stackoverflow.com/questions/46455943/unsupported-operand-types-for-float-and-builtin-function-or-method
