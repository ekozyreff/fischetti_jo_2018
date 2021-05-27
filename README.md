# A connection between Integer Programming and Neural Networks

This code is an implementation of some of the ideas described in the paper "Deep neural networks and mixed integer linear optimization", by M. Fischetti and J. Jo (https://doi.org/10.1007/s10601-018-9285-6). It requires the solver **Gurobi** (https://www.gurobi.com/) installed with a valid license.

In summary, the code does two things:

- First, it trains a neural network for digit recognition (MNIST set) with ReLU activations using Tensorflow. The archictecure of the network is simple (784-8-8-8-10 neurons) and the accuracy is about 92%.

- Then, a mixed integer programming (MIP) formulation is used to model the network using binary variables to indicate the activation of each neuron. This MIP formulation allows us to modify an image that is correctly classified (say, as the digit 7) and turn it into an image that is classified as another digit (say a 3). The mathematical model assures that the modifications on the original image are minimal, so the digit can still be recognized by human eyes as the original one (in this example, a 7).

The images below show the original image (classified as a 7) and the modified image (classified as a 3).

![ogirinal and modified images](73.png "Original image (7) and modified image (3)")

This project was build with the intent of investigating the relationship of integer programming and deep learning with an applied example. A post describing this project in more details was published [here](https://www.linkedin.com/pulse/how-fool-neural-network-ern%25C3%25A9e-kozyreff-filho/).
