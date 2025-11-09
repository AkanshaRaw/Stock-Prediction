import matplotlib.pyplot as plt

def plot_results(train_data, test_data, title, *predictions):
    """
    Plots the training data, actual test data, and one or more prediction sets.
    
    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Test data (for 'y_test' actuals).
        title (str): The title for the plot.
        *predictions (tuple): Tuples of (label, prediction_array).
    """
    plt.figure(figsize=(16, 8))
    plt.title(title, fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($)', fontsize=14)
    
    # Plot training data
    plt.plot(train_data.index, train_data['Close'], label='Train History')
    
    # Plot actual test data
    plt.plot(test_data.index, test_data['Close'], label='Actual Price (Test)', color='orange')
    
    # Plot each set of predictions
    colors = ['green', 'red', 'purple', 'brown']
    for i, (label, pred_values) in enumerate(predictions):
        color = colors[i % len(colors)]
        
        # Ensure prediction length matches test data length for plotting
        if len(pred_values) != len(test_data):
            print(f"Warning: Prediction '{label}' length ({len(pred_values)}) does not match test data length ({len(test_data)}).")
            # This can happen if predictions start at a different point.
            # We'll plot against the *last* N indices of the test data.
            plot_indices = test_data.index[-len(pred_values):]
            plt.plot(plot_indices, pred_values, label=label, color=color, linestyle='--')
        else:
            plt.plot(test_data.index, pred_values, label=label, color=color, linestyle='--')

    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
