import csv


def get_file_length(filename):
    with open(filename, newline='', encoding='utf-16') as csvfile:
            reader = csv.reader(csvfile)
            length = sum(1 for row in reader)
    return length


def print_backtest_status(counter, 
                          backtest_length, freq=100):
    if backtest_length is not None:
        if counter % freq == 0:
            print("Backtest: {:.3f}%".format(100 * counter / backtest_length))
        
    