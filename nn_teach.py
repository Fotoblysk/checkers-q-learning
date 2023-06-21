from src.Qlearning_model import run_learning


def main():
    models = [
        ['model_big_neurons', 32 * 32 * 4], #4096
        ['model_small_neurons', 32 * 32 // 4], #256
        ['model_mid_neurons', 32 * 32], #1024
    ]
    for i in models:
        model_name, hidden_size = i
        run_learning(model_name, hidden_size, num_games=2000)


if __name__ == "__main__":
    main()
