from src.Qlearning_model import run_learning


def main():
    models = [
        ['model_big_neurons', 32],#32 * 32 * 4],
        ['model_small_neurons', 8],#32 * 32 // 4],
        ['model_mid_neurons', 16]#32 * 32],
    ]
    for i in models:
        model_name, hidden_size = i
        run_learning(model_name, hidden_size, num_games=200)#00000)


if __name__ == "__main__":
    main()
