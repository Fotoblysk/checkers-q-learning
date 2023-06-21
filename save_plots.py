import os
import csv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


if __name__ == '__main__':
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    CSV_PREFIX = 'models/'
    files = find_csv_filenames(f'./{CSV_PREFIX}')

    for file_name in files:
        with open(f"{CSV_PREFIX}{file_name}", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            data = [row for row in spamreader if len(row) > 0]

        data_series_labels = ['Loss', 'Combos', 'Len', 'Reward']
        operations = [[100, 500, 1000] for i in data_series_labels]

        data_series_labels.append('Wins')
        data.append([max(0, i) for i in data[3]])
        operations.append([500000])

        for i, data_series in enumerate(data):
            plt.figure()
            plt.xlabel(f"{data_series_labels[i]}")
            plt.ylabel("Game")
            if not data_series_labels[i] == 'Wins':
                plt.plot(data[i], label="value")
                plt.savefig(f'{"./plots/" + file_name}_{data_series_labels[i]}_raw.png')

            for op in operations[i]:
                smoothed = uniform_filter1d(data[i], size=op)
                plt.plot(smoothed, label=f"smoothed x{op}")

            plt.title(data_series_labels[i] + ' ' + file_name.split('_')[1])

            plt.legend()
            plt.savefig(f'{"./plots/" + file_name}_{data_series_labels[i]}_legend.png')
    # plt.show()
