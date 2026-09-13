"""Keep each MuJoCo/OpenGL context on one process and one thread.

Gradio can resume a generator on a different worker thread, and cancellation
can close it on another thread. Native renderer lifetimes must not follow that
thread scheduling. A spawned process owns all trial state and renderer cleanup.
"""
import multiprocessing as mp
from queue import Empty, Full
import time


def _worker(inputs, options, messages, stop):
    import torch
    from .public_trial import run_trial
    torch.set_num_threads(2)
    deadline = time.perf_counter() + options['max_wall_seconds'] + 30
    iterator = run_trial(*inputs, **options, cancel_event=stop)
    try:
        for result in iterator:
            while not stop.is_set():
                try:
                    messages.put(('frame', result), timeout=.2)
                    break
                except Full:
                    if time.perf_counter() > deadline:
                        stop.set()
                    continue
            if stop.is_set():
                break
    except Exception:
        # The public output does not contain server paths or stack traces.
        import traceback
        traceback.print_exc()
        try:
            messages.put(('frame', (None, 'The host could not complete this trial. Try again.',
                                   {'status':'failed','message':'Host runtime error.'})), timeout=.5)
        except Full:
            pass
    finally:
        iterator.close()
        try:
            messages.put(('end', None), timeout=.5)
        except Full:
            pass
        messages.close()
        # Do not let an abandoned image queue keep a cancelled child alive.
        if stop.is_set():
            messages.cancel_join_thread()


class TrialWorker:
    def __init__(self, inputs, *, suite=None, cache_dir=None, max_wall_seconds=300):
        ctx = mp.get_context('spawn')
        self.stop_event = ctx.Event()
        self.messages = ctx.Queue(maxsize=2)
        self.process = ctx.Process(target=_worker, args=(inputs, {
            'suite':suite,'cache_dir':cache_dir,'max_wall_seconds':max_wall_seconds}, self.messages, self.stop_event))
        self.process.daemon = True
        self.max_wall_seconds = max_wall_seconds
        self.started = time.perf_counter()
        self.process.start()

    def frames(self):
        last = None
        while not self.stop_event.is_set():
            try:
                kind, value = self.messages.get(timeout=.25)
            except Empty:
                if not self.process.is_alive():
                    break
                if time.perf_counter() - self.started > self.max_wall_seconds + 30:
                    self.stop_event.set()
                    break
                continue
            if kind == 'end':
                return
            last = value
            yield value
        if not self.stop_event.is_set() and (last is None or last[2].get('status') in ('running','preparing')):
            yield None, 'The simulation worker stopped. Start a fresh trial.', {'status':'failed','message':'Simulation worker stopped.'}

    def stop(self):
        self.stop_event.set()

    def close(self):
        self.stop()
        self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        self.messages.close()
