import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, valid_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_passed = len(train_losses)
    ax.plot(range(1, epochs_passed + 1), train_losses, label="Train", linewidth=2, marker='o', markersize=2)
    ax.plot(range(1, epochs_passed + 1), valid_losses, label="Validation", linewidth=2, marker='o', markersize=2)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, len(train_losses))
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    fig.tight_layout()
    fig.show()