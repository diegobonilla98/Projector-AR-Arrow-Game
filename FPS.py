import time


class CountsPerSec:
    def __init__(self):
        self._start_time = None
        self.num_occurrences = 0

    def start(self):
        self._start_time = time.time()
        return self

    def increment(self):
        self.num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = time.time() - self._start_time
        return self.num_occurrences / elapsed_time
