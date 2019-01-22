from subprocess import Popen, DEVNULL
import time
import torch
import itertools
import os

gpu_cycler = itertools.cycle(range(torch.cuda.device_count()))


class Job(object):

    def __init__(self, scriptname, updates):
        self._scriptname = scriptname
        self._updates    = [f'{name}={value}' for name, value in updates.items()]
        self._process    = None
        self.gpu_index   = None

    def create(self, gpu_index):
        self.gpu_index = gpu_index
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = f'{gpu_index}'
        self._process = Popen(['python', self._scriptname, '-l', 'ERROR', 'with'] + self._updates, env=env)

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

        def get_job(gpu_index):
            job = self._jobs.pop()
            job.create(gpu_index)
            return job

        try:
            # run initial batch
            print(f'Starting {procs_to_start} jobs')
            for i in range(procs_to_start):
                self._running.add(get_job(next(gpu_cycler)))

            # keep polling the list of running processes. if one has terminated, remove it and start
            # a new one, unless the list of run configs is empty. In that case, if there are no more
            # running processes, exit the loop, otherwise simply remove the terminated process.
            while True:
                # all done?
                if len(self._jobs) == 0 and not self._running:
                    break

                # we can't change the _running set during iteration, so we keep track of what to do
                # during the loop and do it after
                to_remove = set()
                to_start = set()
                for proc in self._running:
                    if proc.poll() is not None:
                        if len(self._jobs) > 0:
                            print('Starting new job.')
                            to_start.add(get_job(proc.gpu_index))
                            to_remove.add(proc)
                            print(f'{len(self._jobs)} left.')
                        else:
                            to_remove.add(proc)
                for p in to_remove:
                    self._running.remove(p)
                for p in to_start:
                    self._running.add(p)
                time.sleep(5)
        except Exception as e:
            print(e)
            print('Attempting to cancel all processes...')
            for proc in self._running:
                proc.terminate()
