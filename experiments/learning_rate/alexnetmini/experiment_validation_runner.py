from experiments.learning_rate.alexnetmini.experiment_validation import ex, schedules

import itertools
import os
from subprocess import Popen, DEVNULL
from argparse import ArgumentParser
import time


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--parallel', action='store_true', help='Run experiments to run')
    parser.add_argument('-n', '--nruns', type=int, default=5, help='Number of runs per config')
    parser.add_argument('-j', '--jobs', type=int, default=3, help='Number of processes to use')
    args = parser.parse_args()

    lrs = [0.05, 0.1, 0.2, 0.5]
    batch_sizes = [128, 256, 1024]
    runs = list(itertools.chain.from_iterable(itertools.product(schedules, lrs, batch_sizes)
                                              for _ in range(args.nruns)))

    experiment_scriptname = os.path.join(os.path.dirname(__file__), 'experiment_validation.py')

    if args.parallel:
        print('Running parallel')
        running = set()

        # shortcut to pop a run config from the list and start a process with it
        def start_job():
            schedule, lr, bs = runs.pop()
            running.add(Popen(['python', experiment_scriptname, 'with', f'schedule={schedule}',
                               f'base_lr={lr}', f'batch_size={bs}'],
                              stdout=DEVNULL))

        # do not start more procs than configs
        procs_to_start = min(len(runs), args.jobs)

        try:
            # run initial batch
            print(f'Starting {procs_to_start} jobs')
            for i in range(procs_to_start):
                start_job()

            # keep polling the list of running processes. if one has terminated, remove it and start
            # a new one, unless the list of run configs is empty. In that case, if there are no more
            # running processes, exit the loop, otherwise simply remove the terminated process.
            while True:
                # all done?
                if not runs and not running:
                    break

                for proc in running:
                    if proc.poll() is not None:
                        if runs:
                            print('Found! Starting new job.')
                            start_job()
                            running.remove(proc)
                            print(f'{len(runs)} left.')
                        else:
                            running.remove(proc)
                time.sleep(10)
        except Exception as e:
            print(e)
            print('Attempting to cancel all processes...')
            for proc in running:
                proc.terminate()
    else:
        print('Running serial')
        for schedule in runs:
            ex.run(config_updates={'schedule': schedule})


if __name__ == '__main__':
    main()
