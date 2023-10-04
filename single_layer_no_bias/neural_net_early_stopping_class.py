# %%
class FeedForwardModel:
    # Logs
    OUTPUTS = []
    WEIGHTS = []
    ERRORS = []

    def __init__(self, weight=0.8, bias=0):
        self.weight = weight
        self.bias = bias

    def cost_function(self, output, expected_output):
        return (output - expected_output) ** 2

    def gradient_calculation(self, input_value, expected_output, weight):
        return 2 * (input_value * weight - expected_output) * input_value

    def update_weight(self, weight, gradient, learning_rate):
        return weight - learning_rate * gradient

    def check_cost_improvement(self, errors, early_stopping_steps):
        # Go over the last n errors, if the last error is smaller than all the previous ones, return True
        for lookback in errors[-(early_stopping_steps + 1) : -1]:
            if errors[-1] < lookback:
                return False
            return True

    def train(self, input_value, expected_output, learning_rate, training_steps, early_stopping_steps=5):
        """
        Train the network. Our network consists for now only of one neuron, consisting of a weight and a bias.
        We ignore the bias for now, so we only train our one weight.

        The network will stop training if
        - the cost function does not improve for early_stopping_steps steps or
        - the training_steps is reached.
        """
        # Network initialization
        early_stop = False

        # Reset logs
        self.OUTPUTS = []
        self.WEIGHTS = []
        self.ERRORS = []

        # Training loop
        for i in range(training_steps):
            # Calculate output: Out network for now
            output = input_value * self.weight + self.bias

            # Calculate cost: (output - expected_output) ** 2
            cost = self.cost_function(output, expected_output)
            # Calculate gradient: (2 * (input_value * weight) - expected_output) * input_value
            gradient = self.gradient_calculation(input_value, expected_output, self.weight)

            # Update weight: weight - learning_rate * gradient
            self.weight = self.update_weight(self.weight, gradient, learning_rate)

            self.OUTPUTS.append(output)
            self.ERRORS.append(cost)
            self.WEIGHTS.append(self.weight)

            early_stop = self.check_cost_improvement(self.ERRORS, early_stopping_steps)

            if early_stop:
                output = self.OUTPUTS[-(early_stopping_steps)]
                cost = self.ERRORS[-(early_stopping_steps + 1)]
                self.weight = self.WEIGHTS[-(early_stopping_steps + 1)]
                break

    def predict(self, input_value):
        return input_value * self.weight + self.bias


# %%
# Training data
input_value = 2.5
expected_output = 1.675

# Hyperparameters
learning_rate = 0.1
training_steps = 10000
early_stopping_steps = 5

# Network initialization
initial_weight = 0.5
bias = 0

model = FeedForwardModel(initial_weight, bias)
model.train(input_value, expected_output, learning_rate, training_steps, early_stopping_steps)
print(f"Weight: {model.weight}, Bias: {model.bias}")

prediction = model.predict(input_value)
print(f"Prediction: {prediction}")

# %%
