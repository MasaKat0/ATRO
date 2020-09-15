import numpy as np
import argparse
import sys
from sklearn.datasets import load_svmlight_file

sys.path.append('../')

from algorithms import *



def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='ATRO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='staimage',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=200,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['australian', 'cod', 'pima', 'skin'],
                        help="Presets of configuration")
    args = parser.parse_args(arguments)

    if args.preset == 'australian':
        args.sample_size = 500
        args.dataset = 'australian'
        args.num_trials = 10
    elif args.preset == 'cod':
        args.sample_size = 500
        args.dataset = 'cod-rna'
        args.num_trials = 10
    elif args.preset == 'pima':
        args.sample_size = 500
        args.dataset = 'diabetes'
        args.num_trials = 10
    elif args.preset == "skin":
        args.sample_size = 500
        args.dataset = 'skin_nonskin'
        args.num_trials = 10
    return args

def data_generation(data_name, N):
    X, Y = load_svmlight_file('data/%s'%data_name)
    X = X.toarray()
    maxX = X.max(axis=0)
    maxX[maxX == 0] = 1
    X = X/maxX
    Y = np.array(Y, np.int64)

    N_train = np.int(N*0.7)

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'skin_nonskin':
        Y[Y==2] = -1
    if data_name == 'liver-disorders':
        Y[Y==0] = -1
        
    X_train, X_test = X[:N_train], X[N_train:]
    Y_train, Y_test = Y[:N_train], Y[N_train:]
    
    return X_train, Y_train, X_test, Y_test

def main(arguments):
    args = process_args(arguments)

    data_name = args.dataset
    num_trials = args.num_trials
    sample_size = args.sample_size

    epsilons = [0, 0.001, 0.01, 0.1]
    costs = [0.2, 0.3, 0.4]

    X_train, Y_train, X_test, Y_test = data_generation(data_name, sample_size)
    
    res_SVM = np.zeros((num_trials, len(epsilons),  len(costs), len(epsilons)))
    res_AT = np.zeros((num_trials, len(epsilons),  len(costs), len(epsilons)))
    res_MH = np.zeros((num_trials, len(epsilons),  len(costs), len(epsilons)+1))
    res_ATRO = np.zeros((num_trials, len(epsilons),  len(costs), len(epsilons)+1))
    num_reject_MH = np.zeros((num_trials, len(epsilons),  len(costs), len(epsilons)+1))
    num_reject_ATRO = np.zeros((num_trials, len(epsilons),  len(costs), len(epsilons)+1))
    
    for trial in range(num_trials):
        for i in range(len(epsilons)):
            for j in range(len(costs)):
                epsilon = epsilons[i]
                cost = costs[j]
                
                alpha = 1
                beta = 1/(1-2*cost)
                
                clf_SVM = SVM(None)
                x_train, x_test, lda_chosen = clf_SVM.model_selection(X_train, Y_train, X_test, folds=2, num_basis=100)
                
                clf_SVM = SVM(lda_chosen)
                clf_SVM.fit(x_train, Y_train)
                
                for k in range(len(epsilons)):
                    res_SVM[trial, i, j, k] = clf_SVM.error(x_test+epsilons[k]*np.sign(clf_SVM.gamma), Y_test)

                clf_AT = AT(lda_chosen, epsilon)
                clf_AT.fit(x_train, Y_train)
                
                for k in range(len(epsilons)):
                    res_AT[trial, i, j, k] = clf_AT.error(x_test+epsilons[k]*np.sign(clf_AT.gamma), Y_test)
                
                clf_MH = MH(None, None, cost, alpha, beta)
                x_train, x_test, lda0_chosen, lda1_chosen = clf_MH.model_selection(X_train, Y_train, X_test, folds=2, num_basis=100)
                
                clf_MH = MH(lda0_chosen, lda1_chosen, cost, alpha, beta)
                clf_MH.fit(x_train, Y_train)
                
                for k in range(len(epsilons)):
                    res1, res2 = clf_MH.error(x_test+epsilons[k]*np.sign(clf_MH.gamma), Y_test,  x_test+epsilons[k]*np.sign(clf_MH.theta), show_rate=True)
                    res_MH[trial, i, j, k] = res1
                    num_reject_MH[trial, i, j, k] = res2

                res1, res2 = clf_MH.error(x_test+0.01*np.sign(clf_MH.gamma), Y_test,  x_test, show_rate=True)
                res_MH[trial, i, j, -1] = res1
                num_reject_MH[trial, i, j, -1] = res2
                
                clf_ATRO = ATRO(None, None, cost, alpha, beta, epsilon)
                x_train, x_test, lda0_chosen, lda1_chosen = clf_ATRO.model_selection(X_train, Y_train, X_test, folds=2, num_basis=100)
                
                clf_ATRO = ATRO(lda0_chosen, lda1_chosen, cost, alpha, beta, epsilon)
                clf_ATRO.fit(x_train, Y_train)
                
                for k in range(len(epsilons)):
                    res1, res2 = clf_ATRO.error(x_test+epsilons[k]*np.sign(clf_ATRO.gamma), Y_test,  x_test+epsilons[k]*np.sign(clf_ATRO.theta), show_rate=True)
                    res_ATRO[trial, i, j, k] = res1
                    num_reject_ATRO[trial, i, j, k] = res2
                    
                res1, res2 = clf_ATRO.error(x_test+0.01*np.sign(clf_ATRO.gamma), Y_test,  x_test, show_rate=True)
                res_ATRO[trial, i, j, -1] = res1
                num_reject_ATRO[trial, i, j, -1] = res2
                
                np.save('results/res_svm_%s'%data_name, res_SVM)
                np.save('results/res_at_%s'%data_name, res_AT)
                np.save('results/res_mh_%s'%data_name, res_MH)
                np.save('results/res_atro_%s'%data_name, res_ATRO)
                np.save('results/num_reject_mh_%s'%data_name, num_reject_MH)
                np.save('results/num_reject_atro_%s'%data_name, num_reject_ATRO)

if __name__ == '__main__':
    main(sys.argv[1:])

    
