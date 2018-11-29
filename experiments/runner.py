from subprocess import Popen, DEVNULL
import time
import torch
import itertools
import os


class Job(object):
    gpu_cycler = itertools.cycle(range(torch.cuda.device_count()))

    def __init__(self, scriptname, updates):
        self._scriptname = scriptname
        self._updates    = [f'{name}={value}' for name, value in updates.items()]
        self._process    = None

    def create(self):
        gpu_index = next(Job.gpu_cycler)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = f'{gpu_index}'
        self._process = Popen(['python', self._scriptname, '-l', 'ERROR', 'with'] + self._updates,
                              stdout=DEVNULL, env=env)

    def poll(self):
        return self._process.poll()

    def terminate(self):
        return self._process.terminate()


class Runner(object):

    def __init__(self, n_parallel):
        self._n_parallel = n_parallel
        self._jobs       = set()
        self._running    = set()

    def submit(self, scriptname, **conf_updates):
        self._jobs.add(Job(scriptname, conf_updates))

    def run(self):
        # do not start more procs than configs
        procs_to_start = min(len(self._jobs), self._n_parallel)

        def start_job():
            job = self._jobs.pop()
            job.create()
            self._running.add(job)

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
                if len(self._jobs) == 0 and not self._running:
                    break

                for proc in self._running:
                    if proc.poll() is not None:
                        if len(self._jobs) > 0:
                            print('Starting new job.')
                            start_job()
                            self._running.remove(proc)
                            print(f'{len(self._jobs)} left.')
                        else:
                            self._running.remove(proc)
                time.sleep(5)
        except Exception as e:
            print(e)
            print('Attempting to cancel all processes...')
            for proc in self._running:
                proc.terminate()
