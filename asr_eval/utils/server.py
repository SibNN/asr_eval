import os
import signal
import subprocess
import threading


__all__ = [
    'ServerAsSubprocess',
]


class ServerAsSubprocess:
    """The class constructor runs a given command as a suprocess and
    waits until a :code:`ready_message` appears in the stdout output.
    After this, you can use :code:`.stop()` to send SIGINT to the
    process.
    
    Example:
        >>> vllm_proc = ServerAsSubprocess([  #doctest: +SKIP
        ...     'vllm', 'serve', 'mistralai/Voxtral-Mini-3B-2507', '--port', '8001', ...
        ... ], ready_message='Application startup complete', verbose=False)
        >>> # here you can make API calls to the VLLM server http://localhost:8001/v1
        >>> vllm_proc.stop()  #doctest: +SKIP
    """
    
    def __init__(
        self,
        cmd: list[str],
        ready_message: str | None = 'Application startup complete',
        verbose: bool = True
    ):
        self.verbose = verbose
        
        if not verbose and (value := os.environ.get('SERVER_VERBOSE')):
            print(
                'ServerAsSubprocess: verbose set to'
                f' true due to SERVER_VERBOSE={value}'
            )
            self.verbose = True
        
        print('Starting the server')
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        
        # Waiting until the server is ready
        if ready_message is not None:
            while (line := self._readline()) is not None:
                if ready_message in line:
                    print('Server is ready')
                    break
            else:
                self.stop()
                raise RuntimeError(
                    'Cannot start the server, use verbose=True to debug'
                )
        
        # Print output if verbose=True until the server process is alive
        def logger_fn():
            while self._readline() is not None:
                pass
        # use daemon=True so that the logger_thread stops
        # when main the thread stops
        self.logger_thread = threading.Thread(target=logger_fn, daemon=True)
        self.logger_thread.start()
        
    def _readline(self) -> str | None:
        assert self.process and self.process.stdout
        line_bytes = self.process.stdout.readline()
        if not line_bytes:
            return None
        line = line_bytes.decode(errors='ignore').strip()
        if self.verbose:
            print(line)
        return line
    
    def stop(self):
        assert self.process
        self.process.send_signal(signal.SIGINT)
        self.process.wait()
        print('Server exited')
    
    def __del__(self):
        if self.process:
            self.stop()