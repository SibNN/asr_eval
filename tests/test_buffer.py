import threading
import time

import pytest

from asr_eval.streaming.buffer import StreamingQueue


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_error_propagation():
    """
    The sender thread encounters an error, catches it and sends into the buffer. The error
    is raised in the consumer thread.
    """
    queue = StreamingQueue[int]()

    def producer():
        try:
            for i in range(11):
                queue.put(1 // (10 - i))  # ZeroDivisionError will occur on the last step
                time.sleep(0.1)
        except BaseException as e:
            queue.put_error(e)
            raise e
        
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
        
    with pytest.raises(RuntimeError) as exc_info:
        while True:
            assert queue.get() in (0, 1)
    
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)
        
    producer_thread.join()