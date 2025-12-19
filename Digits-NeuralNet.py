import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_and_split_data(train_size=1400, seed=0):
    """Load the digits dataset and split into train/test sets."""
    digits = load_digits()
    X, y = digits.data, digits.target
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess_data(X_train, X_test):
    """Standardize the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def create_neural_network(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0):
    network = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        max_iter=max_iter,
        random_state=random_state,
    )
    return network

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

def visualize_sample(X_test, y_test, sample_index=0, output_path="input-sample.png"):
    image = X_test[sample_index].reshape(8, 8)

    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(f"True label: {y_test[sample_index]}")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()


def print_class_probabilities(model, X_sample, y_true):
    probabilities = model.predict_proba([X_sample])[0]
    predicted_class = np.argmax(probabilities)

    print(f"True label: {y_true}")
    print("Class probabilities:")
    for digit, prob in enumerate(probabilities):
        print(f"  {digit}: {prob:.4f}")

    print(f"\nPredicted class: {predicted_class}")


def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]} (8x8 images), Classes: 10 (digits 0-9)\n")

    # Preprocess
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # Create, train, and evaluate
    model = create_neural_network(hidden_layer_sizes=(64, 32))
    accuracy = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Neural network accuracy: {accuracy:.3f}\n")


    sample_index = 0
    visualize_sample(X_test, y_test, sample_index, "input-sample.png")
    print("Sample image saved to input-sample.png\n")
    print_class_probabilities(model, X_test_scaled[sample_index], y_test[sample_index])


if __name__ == "__main__":
    main()
