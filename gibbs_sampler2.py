# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 03:46:05 2014

@author: mit
"""

"""
This script implement gibbs sampling with information in the form of alpha parameter
used for the dirichlet prior of the cathegorical distribution. Therefore, we can 
derive the full conditional probability distribution and sample from there.
"""

from numpy.random import randint
from sampling import sample
import math
import matplotlib.pyplot as plt

# Read data from file
filename = 'a.seq'
f = open(filename, 'r')

K = int(f.readline())
N = int(f.readline())
w = int(f.readline())
alphabet = list(f.readline()[:-1])
alpha_b = list(map(float, f.readline()[:-1].split(',') ))
alpha_w = list(map(float, f.readline()[:-1].split(',') ))

sequences = []
for i in range(K):
    seq = f.readline()[:-1].split(',')
    sequences += [seq]

position = list(map(int, f.readline()[:-1].split(',') ))
f.close()

def full_conditional(sequences, pos, seqth):
    
    # Count the background
    q = {x:[1]*w for x in alphabet}
    p = {x: 1 for x in alphabet}
    
    for i in range(len(sequences)):
        if i == seqth:
            continue
        for j in range(len(sequences[i])):
            if j < pos[i] or j > pos[i]+w:
                c = sequences[i][j]
                p[c] = p[c]+1
    
    for i in range(len(sequences)):
        if i== seqth:
            continue
        for j in range(w):
            start_pos = pos[i]
            c = sequences[i][start_pos+j]
            q[c][j] = q[c][j]+1
    
    A = [0]*(N-w)
    
    for i in range(N-w):
        pback = math.gamma(sum(alpha_b)) / math.gamma(K*(N-w) + sum(alpha_b))
        
        extra = {'A':0, 'C':0, 'T':0, 'G':0}
        for j in range(N):
            if j < i or j > i+w:
                c = sequences[seqth][j]
                extra[c] += 1
        for j in range(len(alphabet)):
            a = alphabet[j]
            pback *= math.gamma(p[a]+extra[a] + alpha_b[j]) / math.gamma(alpha_b[j])
            
        pcol = 1
        
        for j in range(w):
            pm = math.gamma(sum(alpha_w)) / math.gamma(K + sum(alpha_w))
            
            for k in range(len(alphabet)):
                a = alphabet[k]
                extra = 0
                if sequences[seqth][i+j] == a:
                    extra = 1
                pm *= math.gamma(q[a][j]+extra) / math.gamma(alpha_w[k])
            
            pcol = pcol*pm
        
        A[i] = pback * pcol
    
    return A
        

# First, initialize the start position randomly
pos = [randint(0, N-w+1) for x in range(K)]
orig_pos = pos[:]

MAX_ITER = 100
p = [0]*(N-w)
b = [0]*MAX_ITER
for it in range(MAX_ITER):
    
    for i in range(K):
        p = full_conditional(sequences, pos, i)
        
        total = sum(p)
        p = [x/total for x in p]
        # Sample new position
        #print 'p', p
        pos[i] = sample(list(range(N-w)), p)
    b[it] = max(p)


# Happy printing
print('start pos', orig_pos)
print('last pos', pos)
print('true pos', position)
plt.plot(b)