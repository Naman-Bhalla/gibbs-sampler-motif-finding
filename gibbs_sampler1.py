# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:17:08 2014

@author: mit
"""

"""
This script implements gibbs sampling to find motif in DNA sequence. The sampler
works with no information about probability distribution of character in the sequence.
"""

from numpy.random import randint
from sampling import sample


# Read data from file
filename = 'a.seq'
f = open(filename, 'r')

K = int(f.readline())
N = int(f.readline())
w = int(f.readline())
alphabet = list(f.readline()[:-1])
alpha_b = f.readline()              # Not too important
alpha_w = f.readline()              # Not too important

sequences = []
for i in range(K):
    seq = f.readline()[:-1].split(',')
    sequences += [seq]

position = list(map(int, f.readline()[:-1].split(',') ))
f.close()


def compute_model(sequences, pos, alphabet, w):
    """
    This method compute the probability model of background and word based on data in 
    the sequences.
    """
    q = {x:[1]*w for x in alphabet}
    p = {x: 1 for x in alphabet}
    
    # Count the number of character occurrence in the particular position of word
    for i in range(len(sequences)):
        start_pos = pos[i]        
        for j in range(w):
            c = sequences[i][start_pos+j]
            q[c][j] += 1
    # Normalize the count
    for c in alphabet:
        for j in range(w):
            q[c][j] = q[c][j] / float( K+len(alphabet) )
    
    # Count the number of character occurrence in background position
    # which mean everywhere except in the word position
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            if j < pos[i] or j > pos[i]+w:
                c = sequences[i][j]
                p[c] += 1
    # Normalize the count
    total = sum(p.values())
    for c in alphabet:
        p[c] = p[c] / float(total)
    
    return q, p
            

# First, initialize the state (in this case position) randomly
pos = [randint(0,N-w+1) for x in range(K)]

# Loop until converge (the burn-in phase)
MAX_ITER = 10
for it in range(MAX_ITER):
    # We pick the sequence, well, in sequence starting from index 0
    for i in range(K):
        # We sample the next position of magic word in this sequence
        # Therefore, we exclude this sequence from model calculation
        seq_minus = sequences[:]; del seq_minus[i]
        pos_minus = pos[:]; del pos_minus[i]
        q, p = compute_model(seq_minus, pos_minus, alphabet, w)
        
        # We try for every possible position of magic word in sequence i and
        # calculate the probability of it being as background or magic word
        # The probability for magic word is calculated by multiplying the probability
        # for each character in each position
        qx = [1]*(N-w)
        px = [1]*(N-w)
        for j in range(N-w):
            for k in range(w):
                c = sequences[i][j+k]
                qx[j] = qx[j] * q[c][k]
                px[j] = px[j] * p[c]
        
        # Compute the ratio between word and background, the pythonic way
        Aj = [x/y for (x,y) in zip(qx, px)]
        norm_c = sum(Aj)
        Aj = [x/norm_c for x in Aj]
        
        # Sampling new position with regards to probability distribution Aj
        pos[i] = sample(list(range(N-w)), Aj)

# Happy printing
print('new pos', pos)
