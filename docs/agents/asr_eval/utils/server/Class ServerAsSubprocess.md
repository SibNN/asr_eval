# Class ServerAsSubprocess (defined in asr_eval/utils/server.py at lines 13-103)

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
    ...

    def stop(self):
        # safe to call multiple times
        ...