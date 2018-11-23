from experiments.learning_rate.alexnetmini.experiment_validation import schedules
from experiments.runner import Runner

import itertools
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nruns', type=int, default=5, help='Number of runs per config')
    parser.add_argument('-j', '--jobs', type=int, default=3, help='Number of processes to use')
    args = parser.parse_args()

    lrs = [0.05, 0.1, 0.2, 0.5]
    batch_sizes = [128, 256, 1024]
    runs = list(itertools.chain.from_iterable(itertools.product(schedules, lrs, batch_sizes)
                                              for _ in range(args.nruns)))

    experiment_scriptname = os.path.join(os.path.dirname(__file__), 'experiment_validation.py')

    runner = Runner(args.jobs)
    for (schedule, lr, bs) in runs:
        runner.submit(experiment_scriptname, schedule=schedule, base_lr=lr, batch_size=bs)

    runner.run()


if __name__ == '__main__':
    main()
