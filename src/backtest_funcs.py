import csv

def get_file_length(filename):
    with open(filename, newline='', encoding='utf-16') as csvfile:
            reader = csv.reader(csvfile)
            length = sum(1 for row in reader)
    return length