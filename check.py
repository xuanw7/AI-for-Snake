import numpy as np
import utils


true_Q = utils.load('data/checkpoint1.npy')
true_N = utils.load('data/checkpoint1_N.npy')


Q = utils.load('checkpoint.npy')
N = utils.load('checkpoint_N.npy')


for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
                for e in range(2):
                    for f in range(2):
                        for g in range(2):
                            for h in range(2):
                                for i in range(4):
          
                                    x = N[a][b][c][d][e][f][g][h][i]
                                    y = true_N[a][b][c][d][e][f][g][h][i]
                                    if (x != y):
                                        print(x,y)
                                        print(a,b,c,d,e,f,g,h,i)
            
    

print(f'Your Q matrix is correct: {np.array_equal(true_Q, Q)}')
print(f'Your N matrix is correct: {np.array_equal(true_N, N)}')