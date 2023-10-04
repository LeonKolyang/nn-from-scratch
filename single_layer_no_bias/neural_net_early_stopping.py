# %%
# Logs
OUTPUTS = []
WEIGHTS = []
ERRORS = []

cost_function = lambda output, expected_output: (output - expected_output) ** 2
gradient_calculation = (
    lambda input_value, expected_output, weight: 2 * (input_value * weight - expected_output) * input_value
)
update_weight = lambda weight, gradient, learning_rate: weight - learning_rate * gradient


def check_cost_improvement(errors, early_stopping_steps):
    # Go over the last n errors, if the last error is smaller than all the previous ones, return True
    for lookback in errors[-(early_stopping_steps + 1) : -1]:
        if errors[-1] < lookback:
            return False
        return True


def train(input_value, expected_output, bias, learning_rate, training_steps, weight=0.8, early_stopping_steps=5):
    """
    Train the network.
    The network will stop training if
    - the cost function does not improve for early_stopping_steps steps
    - or the training_steps is reached.
    """
    # Network initialization
    early_stop = False

    # Training loop
    for i in range(training_steps):
        # Calculate output: Out network for now
        output = input_value * weight + bias

        # Calculate cost: (output - expected_output) ** 2
        cost = cost_function(output, expected_output)
        # Calculate gradient: input_value * 2 * (input_value * weight)
        gradient = gradient_calculation(input_value, expected_output, weight)

        # Update weight: weight - learning_rate * gradient
        weight = update_weight(weight, gradient, learning_rate)

        OUTPUTS.append(output)
        ERRORS.append(cost)
        WEIGHTS.append(weight)

        early_stop = check_cost_improvement(ERRORS, early_stopping_steps)

        if early_stop:
            output = OUTPUTS[-(early_stopping_steps)]
            cost = ERRORS[-(early_stopping_steps + 1)]
            weight = WEIGHTS[-(early_stopping_steps + 1)]
            break
    return weight, bias


def predict(input_value, weight, bias):
    return input_value * weight + bias


# %%
# Training data
input_value = 2.5
expected_output = 1.675

# Hyperparameters
learning_rate = 0.001
training_steps = 10000
early_stopping_steps = 5
## Network initialization
initial_weight = 0.8
bias = 0

weight, bias = train(
    input_value, expected_output, bias, learning_rate, training_steps, initial_weight, early_stopping_steps
)
print(f"Weight: {weight}, Bias: {bias}")

prediction = predict(input_value, weight, bias)
print(f"Prediction: {prediction}")

# %%
