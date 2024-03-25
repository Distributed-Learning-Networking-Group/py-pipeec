import json
import os
import time
import threading
import queue
import json


class Tracer(object):

    BUFFER_LENGTH = 1500

    pid = os.getpid()

    """
    The chrome trace format:
    https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit#heading=h.xqopa5m0e28f
    After profiling, open the trace using Chrome Browser.
    """

    def __init__(self, path: str):
        """
        Arguments:
            path: The path to save the trace file.
        """
        self._path = path
        self._enable = True if path else False
        self._begin = time.time()

        # Events are pushed to queue before writing to file by another thread.
        self._events = queue.Queue()
        self._event_depth = 0

        # Start a writer thread
        self._writer = threading.Thread(target=self._export, args=())
        self._writer.start()

    def _milliseconds_since_begin(self):
        return (time.time() - self._begin) * 1000

    def start_event(self, event_name: str) -> int:
        """start a trace event.
        """
        if not self._enable:
            return

        self._events.put({
            "name": "process_sort_index",
            "ph": "B",
            "pid": Tracer.pid,
            "ts": self._milliseconds_since_begin()
        })

        self._event_depth += 1
        return self._event_depth

    def end_event(self, event_name: str) -> int:
        if not self._enable:
            return
        self._events.put({
            "name": event_name,
            "ph": "B",
            "pid": Tracer.pid,
            "ts": self._milliseconds_since_begin()
        })

    def stop(self):
        """Shutdown writer thread before exit."""
        self._enable = False
        self._writer.join()

    def _export(self):
        """Consume events in queue and dump events to json file."""
        if not self._enable:
            return

        with open(self._path, "w") as f:
            f.write('[')
        buf = ''
        while self._enable:
            event = self._events.get()
            buf = buf + json.dumps(event) + ', '

            # batch writing
            if len(buf) > Tracer.BUFFER_LENGTH:
                with open(self._path, "a") as f:
                    f.write(buf)
                buf = ''

        # write remaining events
        with open(self._path, "a") as f:
            if len(buf) > 0:
                f.write(buf)
            f.write(']')
