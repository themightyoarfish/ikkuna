from experiments.learning_rate.alexnetmini.experiment_validation import ex, schedules

from subprocess import Popen
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('-n', '--nruns', type=int, default=5, help='Number of runs per schedule')
    args = parser.parse_args()
    runs = [schedule for schedule in schedules for _ in range(args.nruns)]
    experiment_scriptname = os.path.join(os.path.dirname(__file__), 'experiment_validation.py')

    if args.parallel:
        print('Running parallel')
        processes = []
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        for schedule in runs:
            processes.add(Popen([experiment_scriptname,
                                 'with', f'schedule={schedule}']))
        for process in processes:
            print('Waiting on all processes')
            process.wait()
    else:
        print('Running serial')
        for schedule in runs:
            ex.run(config_updates={'schedule': schedule})

if __name__ == '__main__':
    main()
