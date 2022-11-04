import numpy
A = numpy.array([[1,4,-1],[2,0,1]])
B = numpy.array([[-1,0],[1,3],[-1,1]])
C = numpy.array([[1,4,-1],[2,0,1],[0,1,1]])

#a)
print("a)\nA + B^T = \n",A + numpy.transpose(B))    #transpose(B) = B.T
print("A - B^T = \n",A - numpy.transpose(B))

#b)
print("\nb)\nA*2 = \n", A*2)
print("A*B = \n", A @ B)        # A.dot(B)= A@B

#c)
print("\nc)\nC*C^-1 = \n", C @ numpy.linalg.inv(C))
