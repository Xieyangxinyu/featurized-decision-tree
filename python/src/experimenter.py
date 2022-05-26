# standard library imports
import argparse
import os
import sys
import importlib
import time
import csv

# package imports
import pandas as pd
__version__ = '0.1.0'
__author__ = u'Wenying Deng'

'''
Script for re-running a script ("subprocess") for different command line argument

Arguments:
--args: csv file containing arguments to run (see required syntax below)
--subproc: python script to run (e.g., './experiment.py')
--dir_out: directory of where to store results

Notes:
- if subprocess returns a dictionary, results will be stored as a dataframe in "dir_out/aggregate_results.pkl",
  where each row corresponds to a single run 

'''


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args', type=str, default='./args.txt')
    parser.add_argument('--subproc', type=str, default='./sub.py')
    parser.add_argument('--dir_out', type=str, default='./output/')
    parser.add_argument('--rowid', type=int, default=1)
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # import subprocess
    sys.path.append(os.path.dirname(args.subproc))
    module = os.path.split(os.path.splitext(args.subproc)[0])[-1]
    sub = importlib.import_module(module)

    # output directory
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    argsdf = pd.read_csv(args.args, index_col=0)
    index = args.rowid - 1

    print('Beginning experiment [%d/%d]' % (index + 1, argsdf.shape[0]))

    row = list(argsdf.iterrows())[index][1]
    args_list = [x for y in zip(argsdf.columns.tolist(), row.tolist()) for x in y]
    args_list = list(filter(lambda z: z != '', args_list))
    try:
        args_list = [str(item) for item in args_list]
    except:
        try:
            args_list = [str(args_list)]
        except:
            args_list = [args_list.lstrip().rstrip()]

    start_time = time.time()
    results = sub.main(args_list)
    runtime = time.time() - start_time

    print('Completed experiment [%d/%d] (time: %.3f seconds)' % (index + 1, argsdf.shape[0], runtime))

    if results is not None:
        # results['runtime_experiment'] = runtime

        # aggregate and save current results
        # resultsdf = pd.concat([pd.DataFrame(row.tolist()).T, pd.DataFrame(results, index=[0])], axis=1)
        resline = row.tolist() + list(results.values())
        csvFile = open(os.path.join(args.dir_out, 'results.csv'), "a")
        writer = csv.writer(csvFile)
        writer.writerow(resline)
        csvFile.close()


if __name__ == '__main__':
    main()