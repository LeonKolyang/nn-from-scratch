# Basic Neural Network & Backpropagation
This is a simple neural network with one weight. It is used to explain the basic concepts of neural networks and backpropagation. 

This chapter is split into the following sections:
- [Intuition](#intuition)
- [Neural Network](#neural-network)
- [Learning Process](#learning-process)
  - [Error](#error)
  - [Gradient & Backpropagation](#gradient-&-backpropagation)
  - [Update](#update)
- [Executing the Learning Process](#executing-the-learning-process)
- [Implementation](#implementation)
  - [Early Stopping](#early-stopping)
  - [Implementation as a Class](#implementation-as-a-class)
- [Sources](#sources)

## Intuition
We start with a mapping of one value to another. Our extremely simple neural network will learn this mapping. The mapping could be, for example:  
`input_value: 2.0 -> output_value: 1.0`  

Intuitively we know that we can reach that mapping by multiplying our input value by 0.5:  
```
input_value * 0.5 = output_value or
        2.0 * 0.5 =         1.0
```

The value we multiply our input with is what our neural network has to learn. We call this value the **weight**. For now, our network has only one weight. Our network's task, or the learning process, is to find the correct weight. This learning process is also called **training**.

## Neural Network
The neural network is a simple **feedforward neural network**. It takes an input, multiplies that input by a value, and gives back an output. This process is done in a **hidden layer** and is called a **forward pass**. Our hidden layer consists of one neuron with the following components we already know:  
- **input_value**: The input to the network and the neuron. _2.0 in our example._
- **weight**: The weight of the neuron. The value the input gets multiplied with. _We want this value to become 0.5_
- **output_value**: The output of the neuron. The result of the multiplication of the input and the weight. _1.0 in our example._  

Neurons also consider bias, but we will ignore that for now and set it to 0 in the implementation.

The following function thereby defines our simple neural network:  
`output = input_value * weight + bias`  
We can already give it an input, which will return an output.
> Implementation: [neural_net.py](neural_net.py#L17)
## Learning Process
Let's start with a different input and output combination:  
`input_value: 2.5 -> output_value: 1.675`  
Finding the right weight to multiply our input becomes more challenging and less intuitive. Our simple neural network is still defined by the same function, with the values this time being:
```
input_value * weight = output_value or
        2.5 * weight =        1.675
```
Using trial and error, we could start with a `weight = 0.5` and then increase or decrease the weight until we reach our desired output just by using our gut feeling and some mathematical intuition:
```
2.5 * 0.5   = 1.25      # 1.25   <  1.675 -> increase weight to 0.75
2.5 * 0.75  = 1.875     # 1.875  >  1.675 -> decrease weight to 0.6
2.5 * 0.6   = 1.5       # 1.5    <  1.675 -> increase weight to 0.675
2.5 * 0.675 = 1.6875    # 1.6875 >  1.675 -> decrease weight to 0.67
2.5 * 0.67  = 1.675     # 1.675  == 1.675 -> we found our weight
```

This is a highly tedious process and might work for one or two examples, but it is not feasible if we want to solve more complex or just more mappings in general.

So let's give this task to our neural network and let it find the correct weight by itself. To enable our network to do this, we need three additional components:
- **loss function**, expressing the **error** between prediction and expected value
- **gradient** of the error to find the direction of the weight adjustment. This process is called **backpropagation**.
- **update function** and **learning_rate** to adjust the weight

Given these components, our neural network can perform a similar process to our handwritten calculation. Let's go over them step by step and find out how they help us to find the right weight.

### Error
> Implementation: [cost_function](neural_net.py#L2)  

We'll stick with our example from earlier and start again with `weight = 0.5`. Running through our network again results in the following output:  
`2.5 * 0.5 = 1.25`  

Instead of intuition, our network needs to rely on a more mathematical approach to determine how good the generated output is: a **loss function**, sometimes called **cost function**.  
The loss function calculates the error between the generated and expected outputs. The error is the difference between the two values. For our simple network, we take the **squared error** as our loss function, which eliminates negative error values and punishes more significant errors more than smaller ones:
```
error = (generated_output - expected_output)²
      = (1.25 - 1.675)²
      = 0.18062500000000004
```
The error is a value that we want to minimize. The smaller the error, the better our network is performing. 

### Gradient & Backpropagation
> Implementation: [gradient_calculation](neural_net.py#L3)

To minimize the error, we need to know in _which direction to adjust our weight_, again replacing our intuition from the earlier example with a mathematical approach. Our first guess with `weight = 0.5` resulted in `2.5 * 0.5   = 1.25`, with the output of `1.25` being smaller than the expected output of `1.675`. We concluded that we need to increase the weight based on intuition. Let's find out how we can enable our network to reach the same conclusion.

The tool our network can use for that is called **gradient descent**. By looking at _the gradient of our error_, the network knows in which direction it has to adjust the weight to minimize the error.

We need to take the **derivative** of our loss function to calculate the gradient. To consider the weight of our network, we need to take the **partial derivative** of our loss function with respect to the weight. 

In more simpler terms, we are using the output calculated during our forward pass and look at the resulting error. We now calculate how much our weight influenced this error. We do this by passing the error back through the network, adjusting the weight, and then calculating the error again. This process is called **backpropagation**.

This does not seem very impressive in our simple example with one weight. In neural networks with more weights and layers, this process allows us to adjust each weight by looking at the error calculated by the network.

Let's look at our loss function again and then bring the weight into the equation:  

```
error            = (generated_output - expected_output)²
generated_output = input_value * weight
```

We can use the chain rule to define the derivative of the loss function with respect to the weight. Our goal is to solve for d(error)/d(weight):
```
d(error)          d(error)         d(generated_output)    
_________  = ___________________ * __________________
d(weight)    d(generated_output)        d(weight)
```
We can solve this equation step by step. Let's start with the first part of the equation:
```
     d(error)           
___________________ = 2 * generated_output - expected_output
d(generated_output)   

                    = 2 * (input_value * weight) - expected_output
```
We solve the second part with:
```
d(generated_output)    
___________________ = input_value
     d(weight)
```
Leaving us with the following equation:
```
d(error)    
_________ = (2 * (input_value * weight) - expected_output) * input_value
d(weight) 
```

We can now calculate the gradient of our error with respect to the weight. Let's plug in our values:
```
# input_value = 2.5
# expected_output = 1.675
# weight = 0.5

d(error)    
_________ = 2 * (input_value * weight - expected_output) * input_value
d(weight)

          = 2 * (2.5 * 0.5 - 1.675) * 2.5
          = -2.125
```
### Update Function
> Implementation: [update_weight](neural_net.py#L6)

In our early example, we increased and decreased the weight by hand until we reached the desired output. Our network can do better with the help of the gradient and the **learning rate**. Thanks to the gradient, the network already knows which direction to adjust the weight. The learning rate defines how big the adjustment should be. The learning rate is a value between 0 and 1. 

We get the weight adjustment by multiplying the gradient with the learning rate. We can then add the weight adjustment to the current weight to get the new weight. 
```
adjusted_weight = weight - (gradient * learning_rate)
```

Let's plug in our values:
```
# weight = 0.5
# gradient = -2.125
# learning_rate = 0.1
adjusted_weight = 0.5 - (-2.125 * 0.1)
                = 0.7125
```
## Executing the Learning Process
We use the adjusted weight to calculate the **output, error, gradient**, and, if needed, the **adjusted weight**. We repeat this process until the error is small enough. In the following example, we start again with `weight = `0.5` and kick off the learning process with the steps described above:
```
# input_value = 2.5
# expected_output = 1.675
# weight = 0.5
# learning_rate = 0.1

# 1st iteration
# neural network defined by: generated_output = input_value * weight
generated_output = 2.5 * 0.5                            # -> 1.25

# loss function: error = (generated_output - expected_output)²
error            = (1.25 - 1.675)²                      # -> 0.18062500000000004

# gradient = 2 * (input_value * weight - expected_output) * input_value
gradient         = 2 * (2.5 * 0.5 - 1.675) * 2.5        # -> -2.125

# adjusted_weight = weight - (gradient * learning_rate)
adjusted_weight  = 0.5 - (-2.125 * 0.1)                 # -> 0.7125

# 2nd iteration, using the adjusted weight
generated_output = 2.5 * 0.7125                         # -> 1.78125
error            = (1.78125 - 1.675)**2                 # -> 0.0112890625
gradient         = 2 * (2.5 * 0.7125 - 1.675) * 2.5     # -> 0.53125
adjusted_weight  = 0.7125 - (0.53125 * 0.1)             # -> 0.659375

# 3rd iteration, using the adjusted weight again
generated_output = 2.5 * 0.659375                       # -> 1.6484375
error            = (1.6484375 - 1.675)**2               # -> 0.0007055664062500023
gradient         = 2 * (2.5 * 0.659375 - 1.675) * 2.5   # -> -0.1328125
adjusted_weight  = 0.659375 - (-0.1328125 * 0.1)        # -> 0.67265625
```
We can see our error getting smaller with each iteration and the weight getting closer to the value we came up with earlier. The approximation we reached within three iterations already looks good, but we can continue the process until the error is small enough. After 30 more iterations, we eventually reached the desired output of `1.675` with a weight of `0.67`. After hitting the desired output multiple times, our error no longer improves, and our model knows it is done learning.
```
# 26th iteration
generated_output = 2.5 * 0.67                          # -> 1.675
error            = (1.675 - 1.675)**2                  # -> 0.0
gradient         = 2 * (2.5 * 0.67 - 1.675) * 2.5      # -> 0.0
adjusted_weight  = 0.67 - (0.0 * 0.1)                  # -> 0.67
.
.
.
# 32nd iteration
generated_output = 2.5 * 0.67                          # -> 1.675
error            = (1.675 - 1.675)**2                  # -> 0.0
gradient         = 2 * (2.5 * 0.67 - 1.675) * 2.5      # -> 0.0
adjusted_weight  = 0.67 - (0.0 * 0.1)                  # -> 0.67
```
## Implementation
The [neural_net.py](neural_net.py) script implements the learning process in a simple for-loop.  
We define all the functions to [calculate the network's output](neural_net.py#L17), [calculate the error with the cost function](neural_net.py#L2), [calculate the gradient](neural_net.py#L3-L5), and [update the weight](neural_net.py#L6).

We build the process defined in this chapter by executing these functions step by step in the for-loop. To use that learning loop, we wrap it into a [train function](neural_net.py#L9-L34) and take the parameters for the learning process (_input_value_, _expected_output_) and the network (_weight_, _learning_rate_, _iterations_) as arguments.
Additionally, we store the [_outputs_, _errors_, and _weights_](neural_net.py#L11-L13) in lists to track the learning process and visualize it later on.

We can now call our training function with the [parameters we used in the example above](neural_net.py#L43-L52) and [execute a learning run](neural_net.py#L54).

After a successful learning run, we can use the weight and bias to [run a test prediction](neural_net.py#L57).

### Early Stopping
The script [neural_net_early_stopping.py](neural_net_early_stopping.py) implements the same learning process as [neural_net.py](neural_net.py), but it stops the learning process early if the error does not improve over a certain number of iterations. 

We use the function [check_cost_improvement](neural_net_early_stopping.py#L14-L19) to look at the last few errors and check if the error is still improving. If not, we stop the learning process.

### Implementation as a Class
To make our network usable, we can wrap it into a class. The script [neural_net_early_stopping_class.py](neural_net_early_stopping_class.py) implements the same logic as above but wraps it in a class. We could now easily use it within a project or do things like storing it as an artifact and load it into different projects or environments.

## Sources
- [Basic Backpropagation](https://www.youtube.com/watch?v=8d6jf7s6_Qs&t=0s)
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
