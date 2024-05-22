"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None: # doesn't return any value

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights(
            (self.n_in, self.n_out)
        )  # pass in the shape: W = np.random.uniform(-a, a, size=shape)
        b = np.zeros((1, self.n_out))

        # cache for backprop
        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({
            "X": None,  # X: Placeholder for input mat
            "Z": None,  # Z: placeholder for pre-act mat
        })  

        # parameter gradients initialized to zero, MUST HAVE THE SAME KEYS AS `self.parameters`
        self.gradients = OrderedDict({"W": np.zeros(W.shape), "b": np.zeros(b.shape)})

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim) (Y in my case)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###

        # perform an affine transformation and activation Z = XW + b
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X @ W + b

        out = self.activation(Z)

        # store information necessary for backprop in `self.cache`
        self.cache["Z"] = Z
        self.cache["X"] = X 

        ### END YOUR CODE ##
        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###

        # unpack the cache
        Z = self.cache["Z"]
        X = self.cache["X"] 
        W = self.parameters["W"]
        b = self.parameters["b"]
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        # breakpoint()
        dZ = self.activation.backward(
            Z, dLdY
        )  # backward(self, Z: np.ndarray, dY: np.ndarray)
        dX = dZ @ W.T
        dW = X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients["b"] = db
        self.gradients["W"] = dW

        ### END YOUR CODE ###

        return dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
        # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        ### BEGIN YOUR CODE ###

        ## implement a convolutional forward pass
        # stride: how many rows/columns we skip each time we move our filter across the input image
        # pad: padding on top/bottom, left/right

        # Calculate the dimensions of the output feature map (number of times filter is applied)
        out_rows = 1 + int((in_rows - kernel_height + 2 * self.pad[0]) / self.stride) # 1 is the initial one 
        out_cols = 1 + int((in_cols - kernel_width + 2 * self.pad[1]) / self.stride) 

        # X(m, H, W, C), batch, height, width, channels
        # breakpoint()
        X_padded = np.pad(
            X,  # The array to be padded
            pad_width=(
                (0, 0),
                (self.pad[0], self.pad[0]),  # rows (height)
                (self.pad[1], self.pad[1]),  # cols (width)
                (0, 0),
            ),  # The amount of padding to apply on each dimension
            mode="constant",  # The padding mode
        )
        # init Z
        Z = np.empty((n_examples, out_rows, out_cols, out_channels), dtype=X.dtype)

        # Perform convolution operation across the spatial dimensions and for each output channel (d1, d2, channeks)
        for row in range(out_rows):
            for col in range(out_cols):
                for channel in range(out_channels):
                    # Apply the convolution by element-wise multiplication of the filter with the input,
                    # summing across the spatial and channel dimensions, then add the bias for the filter
                    Z[:, row, col, channel] = (
                        np.sum(
                            X_padded[:,
                                row * self.stride : row * self.stride + kernel_height, # top, bottom  
                                col * self.stride : col * self.stride + kernel_width,  # left, right
                                :]
                            * W[:, :, :, channel],
                            axis=(1, 2, 3), # summing over rows, cols, channels of Z 
                        )
                        + b[:, channel]  # Add the bias term for the current output channel
                    )

        # Apply the activation function to the convolution results to get the final output
        out = self.activation(Z)

        ## cache any values required for backprop
        self.cache["Z"] = Z
        self.cache["X"] = X

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###
        X = self.cache["X"]
        Z = self.cache["Z"]

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape

        out_rows = 1 + int((in_rows - kernel_height + 2 * self.pad[0]) / self.stride) # 1 is the initial one 
        out_cols = 1 + int((in_cols - kernel_width + 2 * self.pad[1]) / self.stride) 

        # perform a backward pass
        dZ = self.activation.backward(Z, dLdY)  # get dZ/dL
        X_padded = np.pad(
            X,  # The array to be padded
            pad_width=(
                (0, 0),
                (self.pad[0], self.pad[0]),  # rows (height)
                (self.pad[1], self.pad[1]),  # cols (width)
                (0, 0),
            ),  # The amount of padding to apply on each dimension
            mode="constant",  # The padding mode
        )

        # batch_size, in_rows, in_cols, in_channels dX/dL
        dX = np.zeros(X_padded.shape)

        # W[height, width, input_channels, output_channels]
        dW = np.zeros(W.shape)

        # dB sum over batch, row, col
        dB = dZ.sum(axis=(0, 1, 2)).reshape(1, -1) # reshape to 1 row -> 2 dim 

        for row in range(out_rows): 
            for col in range(out_cols): 
                for channel in range(out_channels): 
                    # breakpoint()
                    dW[:, :, :, channel] += np.sum(
                        X_padded[
                            :,
                            row * self.stride : row * self.stride + kernel_height,
                            col * self.stride : col * self.stride + kernel_width,
                            :,
                        ]
                        * dZ[:, row : row + 1, col : col + 1, np.newaxis, channel],
                        axis=0,
                    )

                    dX[
                        :,
                        row * self.stride : row * self.stride
                        + kernel_height,  # rows to index
                        col * self.stride : col * self.stride
                        + kernel_width,  # col to index
                        :,
                    ] += (
                        W[np.newaxis, :, :, :, channel]
                        * dZ[:, row : row + 1, col : col + 1, np.newaxis, channel]
                    )  # newaxis: add a batch dimension to W, making its shape [1, height, width, input_channels, output_channels]

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        # Crop out the padding from dX_pad to get the gradient of the original input (dX)
        # This step is necessary because the forward pass included padding, which we do not want
        # to include in the gradient passed back to the previous layer.
        dX = dX[:, self.pad[0]:in_rows + self.pad[0], self.pad[1]:in_cols + self.pad[1], :]

        # Return the gradient of the input with respect to the loss
        ### END YOUR CODE ###

        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # implement the forward pass

        # Extract the batch size and spatial dimensions from input shape
        n_examples, in_rows, in_cols, in_channels = X.shape

        # Extract the kernel shape
        kernel_height, kernel_width = self.kernel_shape

        # Compute the dimensions of the output feature map after pooling

        out_rows = 1 + int((in_rows - kernel_height + 2 * self.pad[0]) / self.stride)
        out_cols = 1 + int((in_cols - kernel_width + 2 * self.pad[1]) / self.stride)

        # Select the pooling function based on the mode set in the pooling layer
        if self.mode == "average":
            pool_fn = np.mean  
        elif self.mode == "max":
            pool_fn = np.max 

        # Pad the input array with zeros around the edges for proper dimensionality
        # during the pooling operation. Padding is only applied spatially (not on the batch or channel dimensions).
        X_padded = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')

        # Initialize the output array with zeros, with the calculated output dimensions
        X_pool = np.zeros((n_examples, out_rows, out_cols, in_channels))

        # Perform pooling operation over each spatial region of the input array
        for row in range(out_rows):
            for col in range(out_cols):
                # Determine the current slice of the padded input for pooling
                row_0, row_1 = row * self.stride, row * self.stride + kernel_height
                col_0, col_1 = col * self.stride, col * self.stride + kernel_width

                # Apply the pooling function to the current region and store in the output array
                X_pool[:, row, col, :] = pool_fn(
                    X_padded[:, row_0:row_1, col_0:col_1, :], axis=(1, 2)
                )

        # cache any values required for backprop
        self.cache["X"] = X
        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass
        # Retrieve spatial dimensions of output from cache
        out_rows = self.cache["out_rows"]
        out_cols = self.cache["out_cols"]
        X = self.cache["X"]

        # Get the batch size and dimensions of the input and the kernel
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height, kernel_width = self.kernel_shape

        out_rows = 1 + int((in_rows - kernel_height + 2 * self.pad[0]) / self.stride)
        out_cols = 1 + int((in_cols - kernel_width + 2 * self.pad[1]) / self.stride)

        # Pad the input array with zeros around the edges for dimensionality consistency during backpropagation
        X_padded = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')

        # Initialize the gradient tensor for the input with the same shape as the padded input
        dX = np.zeros_like(X_padded)

        # Iterate over each spatial location in the output feature map
        for row in range(out_rows):
            for col in range(out_cols):
                # Calculate the window boundaries for the current spatial location
                row_0, row_1 = row * self.stride, row * self.stride + kernel_height
                col_0, col_1 = col * self.stride, col * self.stride + kernel_width

                # Max pooling backward pass
                if self.mode == "max":
                    # Extract the current pooling region
                    x_curr_pool = X_padded[:, row_0:row_1, col_0:col_1, :].reshape(
                        n_examples, kernel_height * kernel_width, in_channels
                    )
                    # Find the indices of the max values within the pooling region
                    idxs = np.argmax(x_curr_pool, axis=1)
                    # Create a mask with the same shape as the pooling region and set the max value positions to 1
                    mask = np.zeros(x_curr_pool.shape)
                    n_idx, c_idx = np.indices((n_examples, in_channels))
                    mask[n_idx, idxs, c_idx] = 1
                    # Reshape the mask to match the original shape of the pooling region
                    mask = mask.reshape(n_examples, kernel_height, kernel_width, in_channels)                    
                    # Multiply the mask by the upstream gradient and add the result to the gradient of the input
                    dX[:, row_0:row_1, col_0:col_1, :] += mask * dLdY[:, row:row+1, col:col+1, :]

                # Average pooling backward pass
                elif self.mode == "average":
                    # Distribute the upstream gradient evenly to the pooling region
                    dX[:, row_0:row_1, col_0:col_1, :] += dLdY[:, row:row+1, col:col+1, :] / (kernel_height * kernel_width)

        # Trim the padding off the input gradient to match the original input dimensions
        dX = dX[:, self.pad[0]:in_rows + self.pad[0], self.pad[1]:in_cols + self.pad[1], :]

        ### END YOUR CODE ###

        return dX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
