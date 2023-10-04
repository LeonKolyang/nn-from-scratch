# %%
cost_function = lambda output, expected_output: (output - expected_output) ** 2
gradient_calculation = (
    lambda input_value, expected_output, weight: 2 * (input_value * weight - expected_output) * input_value
)
update_weight = lambda weight, gradient, learning_rate: weight - learning_rate * gradient


def train(input_value, expected_output, bias, learning_rate, iterations, weight=0.8):
    # Logs
    OUTPUTS = []
    ERRORS = []
    WEIGHTS = []

    for i in range(iterations):
        # Calculate output: input_value * weight + bias
        output = input_value * weight + bias

        # Calculate cost: (output - expected_output) ** 2
        cost = cost_function(output, expected_output)
        # Calculate gradient: 2 * (input_value * weight - expected_output) * input_value
        gradient = gradient_calculation(input_value, expected_output, weight)

        # Update weight: weight - learning_rate * gradient
        weight = update_weight(weight, gradient, learning_rate)

        OUTPUTS.append(output)
        ERRORS.append(cost)
        WEIGHTS.append(weight)

        # Check if cost function improved, stop if not
        if len(ERRORS) > 1 and ERRORS[-2] < cost:
            break
    return weight, bias


def predict(input_value, weight, bias):
    return input_value * weight + bias


if __name__ == "__main__":
    # %%
    # Training data
    input_value = 2.5
    expected_output = 1.675

    # Hyperparameters
    learning_rate = 0.1
    iterations = 10000
    ## Network initialization
    initial_weight = 0.5
    bias = 0

    weight, bias = train(input_value, expected_output, bias, learning_rate, iterations, initial_weight)
    print(f"Weight: {weight},\nBias: {bias}\nIterations: {len(WEIGHTS)}")

    prediction = predict(input_value, weight, bias)
    print(f"Prediction: {prediction}")
    # %%
