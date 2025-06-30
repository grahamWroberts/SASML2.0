import numpy as np

infile = open("performance3.csv", 'r')
lines = infile.readlines()
best_perf = 0
for line in lines:
    tokens = line.split()
    perf = float(tokens[-1])
    if perf > best_perf:
        best_perf = perf
        best_line = line
print(best_line)
