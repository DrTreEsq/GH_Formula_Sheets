<h2 align="center"> Course 3: Probability & Statistics✅ </h2>

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

