import os

# os.system('python3 automate.py > out.txt')

f = open('out.txt')
counter = 0 
for line in f:
    if '.json' in line:
        print(' \\\\ ') # New line
        counter += 1
        print('File '+str(counter), end = ' ')
    if 'Peaks' in line:
        print('&', end = ' ')
        peak = line.strip().split()[-1]
        print(peak, end = ' ')
    if 'Beats' in line:
        print('&', end = ' ')
        beats = float(line.strip().split()[-1])
        print(round(beats,2), end = ' ')
f.close()
print()
