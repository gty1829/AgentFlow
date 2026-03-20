import io
import queue
import traceback
import contextlib
import multiprocessing as mp


def _initial_env():
    return {
        "__name__": "__main__",
        "__builtins__": __builtins__,
    }


def _worker_loop(task_queue: mp.Queue, result_queue: mp.Queue):
    env = _initial_env()

    while True:
        task = task_queue.get()
        cmd = task.get("cmd")

        if cmd == "shutdown":
            result_queue.put({
                "ok": True,
                "stdout": "",
                "stderr": "",
                "error": None,
                "msg": "shutdown",
            })
            break

        elif cmd == "reset":
            env = _initial_env()
            result_queue.put({
                "ok": True,
                "stdout": "",
                "stderr": "",
                "error": None,
                "msg": "reset",
            })

        elif cmd == "run":
            code = task["code"]
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    exec(code, env, env)

                result_queue.put({
                    "ok": True,
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                    "error": None,
                })
            except Exception:
                result_queue.put({
                    "ok": False,
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                    "error": traceback.format_exc(),
                })

        elif cmd == "get_vars":
            visible_keys = sorted(
                k for k in env.keys()
                if not k.startswith("__")
            )
            result_queue.put({
                "ok": True,
                "stdout": "",
                "stderr": "",
                "error": None,
                "variables": visible_keys,
            })

        else:
            result_queue.put({
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": f"Unknown command: {cmd}",
            })


class PersistentPythonExecutor:
    def __init__(self):
        self.task_queue = None
        self.result_queue = None
        self.process = None
        # self._start_worker()

    def _start_worker(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process = mp.Process(
            target=_worker_loop,
            args=(self.task_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()

    def _restart_worker(self):
        self.close()
        self._start_worker()

    def run(self, code: str, timeout: float = 5.0):
        if not self.process or not self.process.is_alive():
            self._start_worker()

        self.task_queue.put({"cmd": "run", "code": code})

        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            self._restart_worker()
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": f"Execution timed out after {timeout} seconds. Worker restarted.",
            }

    def reset(self, timeout: float = 2.0):
        if not self.process or not self.process.is_alive():
            self._start_worker()
            return {
                "ok": True,
                "stdout": "",
                "stderr": "",
                "error": None,
                "msg": "worker recreated and context reset",
            }

        self.task_queue.put({"cmd": "reset"})
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            self._restart_worker()
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": "Reset timed out. Worker restarted with a fresh context.",
            }

    def get_variables(self, timeout: float = 2.0):
        if not self.process or not self.process.is_alive():
            self._start_worker()

        self.task_queue.put({"cmd": "get_vars"})
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            self._restart_worker()
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": "get_variables timed out. Worker restarted.",
            }

    def close(self):
        if self.process and self.process.is_alive():
            try:
                self.task_queue.put({"cmd": "shutdown"})
                self.result_queue.get(timeout=1.0)
            except Exception:
                pass

            self.process.kill()
            self.process.join(timeout=1.0)

        self.process = None
        self.task_queue = None
        self.result_queue = None
