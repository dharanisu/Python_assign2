#Integers and floats work as you would expect from other languages:

def quicksort(eip):
    if len(eip) <= 1:
        return eip
    mlblr = eip[len(eip) // 2]
    eip_in= [mlblr_out for mlblr_out in eip if mlblr_out < mlblr]
    eip_out = [mlblr_out for mlblr_out in eip if mlblr_out == mlblr]
    mlblr_in = [mlblr_out for mlblr_out in eip if mlblr_out > mlblr]
    return quicksort(eip_in) + eip_out + quicksort(mlblr_in)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"

eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Exponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
eip = 2.5
print(type(eip)) # Prints "<class 'float'>"
print(eip, eip + 1, eip * 2, eip ** 2) # Prints "2.5 3.5 5.0 6.25"

#English words instead of symbols && ||

mlblr_in = True
mlblr_out = False
print(type(mlblr_in)) # Prints "<class 'bool'>"
print(mlblr_in and mlblr_out) # Logical AND; prints "False"
print(mlblr_in or mlblr_out)  # Logical OR; prints "True"
print(not mlblr_in)   # Logical NOT; prints "False"
print(mlblr_in != mlblr_out)  # Logical XOR; prints "True"

#Strings in Python

eip_in = 'hello'    # String literals can use single quotes
eip_out = "world"    # or double quotes; it does not matter.
print(eip_in)       # Prints "hello"
print(len(eip_in))  # String length; prints "5"
eip = eip_in + ' ' + eip_out  # String concatenation
print(eip)  # prints "hello world"
mlblr = '%s %s %d' % (eip_in, eip_out, 12)  # sprintf style string formatting
print(mlblr)  # prints "hello world 12"

#Useful methods

eip = "hello"
print(eip.capitalize())  # Capitalize a string; prints "Hello"
print(eip.upper())       # Convert a string to uppercase; prints "HELLO"
print(eip.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(eip.center(7))     # Center a string, padding with spaces; prints " hello "
print(eip.replace('l', '(ell)'))  # Replace all instances of one substring with another; # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"


#Python Containers - List

mlblr = [3, 1, 2]    # Create a list
print(mlblr, mlblr[2])  # Prints "[3, 1, 2] 2"
print(mlblr[-1])     # Negative indices count from the end of the list; prints "2"
mlblr[2] = 'foo'     # Lists can contain elements of different types
print(mlblr)         # Prints "[3, 1, 'foo']"
mlblr.append('bar')  # Add a new element to the end of the list
print(mlblr)         # Prints "[3, 1, 'foo', 'bar']"
eip = mlblr.pop()      # Remove and return the last element of the list
print(eip, mlblr)      # Prints "bar [3, 1, 'foo']"

#Slicing

mlblr = list(range(5))     # range is a built-in function that creates a list of integers
print(mlblr)               # Prints "[0, 1, 2, 3, 4]"
print(mlblr[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(mlblr[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(mlblr[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(mlblr[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(mlblr[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
mlblr[2:4] = [8, 9]        # Assign a new sublist to a slice
print(mlblr)               # Prints "[0, 1, 8, 9, 4]"

#Loops

eip = ['cat', 'dog', 'monkey']
for mlblr in eip:
    print(mlblr)
# Prints "cat", "dog", "monkey", each on its own line.

#Enumerate

eip = ['cat', 'dog', 'monkey']
for eip_in, eip_out in enumerate(eip):
    print('#%d: %s' % (eip_in + 1, eip))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line

#List Comprehension

mlblr = [0, 1, 2, 3, 4]
eip_list = []
for eip in mlblr:
    eip_list.append(eip ** 2)
print(eip_list)   # Prints [0, 1, 4, 9, 16]

mlblr = [0, 1, 2, 3, 4]
eip_list = [eip ** 2 for eip in mlblr]
print(eip_list)   # Prints [0, 1, 4, 9, 16]

# list comprehensions can also contain conditions

mlblr = [0, 1, 2, 3, 4]
eip_list = [eip ** 2 for eip in mlblr if eip % 2 == 0]
print(eip_list)  # Prints "[0, 4, 16]"


#Dictionaries

eip_dict = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(eip_dict['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in eip_dict)     # Check if a dictionary has a given key; prints "True"
eip_dict['fish'] = 'wet'     # Set an entry in a dictionary
print(eip_dict['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(eip_dict.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(eip_dict.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del eip_dict['fish']         # Remove an element from a dictionary
print(eip_dict.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


# It is easy to iterate over the keys in a dictionary

eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for eip_in in eip_dict:
    eip_out = eip_dict[eip_in]
    print('A %s has %d legs' % (eip_in, eip_out))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"


#Dictionary Comprehension

# If you want access to keys and their corresponding values, use the items method:

eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for eip_in, eip_out in eip_dict.items():
    print('A %s has %d legs' % (eip_in, eip_out))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

# Dictionary comprehension

mlblr = [0, 1, 2, 3, 4]
eip_dict = {eip: eip ** 2 for eip in mlblr if eip % 2 == 0}
print(eip_dict)  # Prints "{0: 0, 2: 4, 4: 16}"

#Sets

eip = {'cat', 'dog'}
print('cat' in eip)   # Check if an element is in a set; prints "True"
print('fish' in eip)  # prints "False"
eip.add('fish')       # Add an element to a set
print('fish' in eip)  # Prints "True"
print(len(eip))       # Number of elements in a set; prints "3"
eip.add('cat')        # Adding an element that is already in the set does nothing
print(len(eip))       # Prints "3"
eip.remove('cat')     # Remove an element from a set
print(len(eip))       # Prints "2"


#Tuples

eip_dict = {(eip, eip + 1): eip for eip in range(10)}  # Create a dictionary with tuple keys
mlblr = (5, 6)        # Create a tuple
print(type(mlblr))    # Prints "<class 'tuple'>"
print(eip_dict[mlblr])       # Prints "5"
print(eip_dict[(1, 2)])  # Prints "1"


#Functions

def sign(eip):
    if eip > 0:
        return 'positive'
    elif eip < 0:
        return 'negative'
    else:
        return 'zero'

for eip in [-1, 0, 1]:
    print(sign(eip))
# Prints "negative", "zero", "positive"
#Function Arguments

def hello(eip_in, eip_out=False):
    if eip_out:
        print('HELLO, %s!' % eip_in.upper())
    else:
        print('Hello, %s' % eip_in)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', eip_out=True)  # Prints "HELLO, FRED!"

#Classes

class Greeter(object):

    # Constructor
    def __init__(self, mlblr):
        self.mlblr = mlblr  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.mlblr.upper())
        else:
            print('Hello, %s' % self.mlblr)

eip = Greeter('Fred')  # Construct an instance of the Greeter class
eip.greet()            # Call an instance method; prints "Hello, Fred"
eip.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"

#NumPy

import numpy as np

mlblr_in = np.array([1, 2, 3])   # Create a rank 1 array
print(type(mlblr_in))            # Prints "<class 'numpy.ndarray'>"
print(mlblr_in.shape)            # Prints "(3,)" 3 columns
print(mlblr_in[0], mlblr_in[1], mlblr_in[2])   # Prints "1 2 3"
mlblr_in[0] = 5                  # Change an element of the array
print(mlblr_in)                  # Prints "[5, 2, 3]"

mlblr_out = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(mlblr_out.shape)                     # Prints "(2, 3)" 2 rows,3 columns
print(mlblr_out[0, 0], mlblr_out[0, 1], mlblr_out[1, 0])   # Prints "1 2 4"

#Some NumPy Functions

import numpy as np

mlblr_in = np.zeros((2,2))   # Create an array of all zeros
print(mlblr_in)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

mlblr_out = np.ones((1,2))    # Create an array of all ones
print(mlblr_out)              # Prints "[[ 1.  1.]]"

eip_in = np.full((2,2), 7)  # Create a constant array
print(eip_in)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

eip_out = np.random.random((2,2))  # Create an array filled with random values
print(eip_out)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
#Mixing integer indexing with slice indexing

import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
eip_in = eip[1, :]    # Rank 1 view of the second row of a
eip_out = eip[1:2, :]  # Rank 2 view of the second row of a
print(eip_in, eip_in.shape)  # Prints "[5 6 7 8] (4,)"
print(eip_out, eip_out.shape)  # Prints "[[5 6 7 8]] (1, 4)"

#Datatypes

import numpy as np

eip = np.array([1, 2])   # Let numpy choose the datatype
print(eip.dtype)         # Prints "int64"

eip = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(eip.dtype)             # Prints "float64"

eip = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(eip.dtype)                         # Prints "int64"

#Array Math

import numpy as np

eip = np.array([[1,2],[3,4]], dtype=np.float64)
mlblr = np.array([[5,6],[7,8]], dtype=np.float64)

print(np.add(eip, mlblr)) # or print(x + y)
# [[ 6.0  8.0]
#  [10.0 12.0]]

print(np.subtract(eip, mlblr)) or print(eip - mlblr)
# [[-4.0 -4.0]
#  [-4.0 -4.0]]

print(np.multiply(eip, mlblr)) or print(eip * mlblr)
# [[ 5.0 12.0]
#  [21.0 32.0]]

print(np.divide(eip, mlblr)) or print(eip / mlblr)
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]

print(np.sqrt(eip))
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]

#Array Indexing
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
mlblr = eip[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(eip[0, 1])   # Prints "2"
mlblr[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(eip[0, 1])   # Prints "77"


#Dot function

import numpy as np

eip_in = np.array([[1,2],[3,4]])
eip_out = np.array([[5,6],[7,8]])

mlblr_in = np.array([9,10])
mlblr_out = np.array([11, 12])

# Inner product of vectors; both produce 219
print(mlblr_in.dot(mlblr_out))
print(np.dot(mlblr_in, mlblr_out))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip_in.dot(mlblr_in))
print(np.dot(eip_in, mlblr_in))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip_in.dot(eip_out))
print(np.dot(eip_in, eip_out))

#Sum along an axis

import numpy as np

eip = np.array([[1,2],[3,4]])

print(np.sum(eip))  # Compute sum of all elements; prints "10"
print(np.sum(eip, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip, axis=1))  # Compute sum of each row; prints "[3 7]"

#MathPlotLib - Plotting Graphs

import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
eip = np.arange(0, 3 * np.pi, 0.1)
mlblr = np.sin(eip)

# Plot the points using matplotlib
plt.plot(eip, mlblr)
plt.show()  # You must call plt.show() to make graphics appear.

#Broadcasting
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlblr = np.array([1, 0, 1])
eip_in = np.empty_like(eip)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for eip_out in range(4):
    eip_in[eip_out, :] = eip[eip_out, :] + mlblr

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(eip_in)

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlblr = np.array([1, 0, 1])
mlblr_in = np.tile(mlblr, (4, 1))   # Stack 4 copies of v on top of each other
print(mlblr_in)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
eip_in = eip + mlblr_in  # Add x and vv elementwise
print(eip_in)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
         
            
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlblr = np.array([1, 0, 1])
eip_in = eip + mlblr  # Add v to each row of x using broadcasting
print(eip_in)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
          
          
import numpy as np

# Compute outer product of vectors
eip_list = np.array([1,2,3])  # v has shape (3,)
eip_dict = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(eip_list, (3, 1)) * eip_dict)

# Add a vector to each row of a matrix
eip = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(eip + eip_list)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((eip.T + eip_dict).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(eip + np.reshape(eip_dict, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(eip * 2)

#Images

import numpy as np
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt

# Uncomment the line below if you're on a notebook
# %matplotlib inline 
eip = imread('C:/Dharani/AI-ML-MeetUp/MLBLR/EIP/Assignments/cat.jpg')

print(eip.dtype, eip.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
mlblr = eip * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
mlblr = imresize(mlblr, (300, 300))

# Write the tinted image back to disk
imsave('C:/Dharani/AI-ML-MeetUp/MLBLR/EIP/Assignments/cat_tinted.jpg', mlblr)

mlblr = eip * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(eip)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(mlblr))
plt.show()

import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
eip = np.array([[0, 1], [1, 0], [2, 0]])
print(eip)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
mlblr = squareform(pdist(eip, 'euclidean'))
print(mlblr)

import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
eip = np.arange(0, 3 * np.pi, 0.1)
mlblr = np.sin(eip)

# Plot the points using matplotlib
plt.plot(eip, mlblr)
plt.show()  # You must call plt.show() to make graphics appear.

import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_in = np.sin(eip)
eip_out = np.cos(eip)

# Plot the points using matplotlib
plt.plot(eip, eip_in)
plt.plot(eip, eip_out)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_in = np.sin(eip)
eip_out = np.cos(eip)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(eip, eip_in)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(eip, eip_out)
plt.title('Cosine')

# Show the figure.
plt.show()
