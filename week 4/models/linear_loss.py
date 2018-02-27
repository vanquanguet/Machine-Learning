import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(y.size) :
        tmps = 0
        for j in range(X[0].size) :
            tmps = tmps + X[i][j] * W[j]
        for j in range(X[0].size) :
            dW[j] = dW[j] + (tmps-y[i]) * X[i][j]
        loss = loss + np.power(tmps - y[i], 2)
        
    loss = loss / (2 * y.size) + reg
    dW = np.dot(X.transpose(), np.dot(X,W)-y)
    dW = dW / y.size
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = np.sum((np.dot(X,W)-y)**2)/(2*y.size) + reg
    dW = np.dot(X.transpose(), np.dot(X,W)-y)
    dW = dW / y.size
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW