from experiments.learning_rate.alexnetmini.experiment_validation import ex, schedules

import os
from subprocess import Popen, DEVNULL
from argparse import ArgumentParser
import time


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--parallel', action='store_true', help='Run experiments to run')
    parser.add_argument('-n', '--nruns', type=int, default=5, help='Number of runs per schedule')
    parser.add_argument('-r', '--jobs', type=int, default=3, help='Number of processes to use')
    args = parser.parse_args()
    runs = [schedule for schedule in schedules for _ in range(args.nruns)]
    experiment_scriptname = os.path.join(os.path.dirname(__file__), 'experiment_validation.py')

    # start 6 jobs
    # while True: poll all jobs and sleep in between
    #   if any one is empty, start new one

    if args.parallel:
        print('Running parallel')
        running = set()

        # shortcut to pop a run config from the list and start a process with it
        def start_job():
            running.add(Popen(['python', experiment_scriptname, 'with', f'schedule={runs.pop()}'],
                              stdout=DEVNULL))

        # do not start more procs than configs
        procs_to_start = min(len(runs), args.jobs)

        # run initial batch
        print(f'Starting {procs_to_start} jobs')
        for i in range(procs_to_start):
            start_job()

        # keep polling the list of running processes. if one has terminated, remove it and start a
        # new one, unless the list of run configs is empty. In that case, if there are no more
        # running processes, exit the loop, otherwise simply remove the terminated process.
        while True:
            # all done?
            if not runs and not running:
                break

            print(f'Checking for finished jobs...')
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
    else:
        print('Running serial')
        for schedule in runs:
            ex.run(config_updates={'schedule': schedule})


if __name__ == '__main__':
    main()
