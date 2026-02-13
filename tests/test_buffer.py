import threading
import time
from queue import Queue, Empty

import pytest

from asr_eval.streaming.buffer import ID_TYPE, StreamingQueue


def send_in_new_thread(
    q: StreamingQueue[str],
    sleep: float,
    id: ID_TYPE,
    data: str,
):
    def sender():
        time.sleep(sleep)
        q.put(data, id=id)
    
    threading.Thread(target=sender).start()

def wait_in_new_thread(
    q: StreamingQueue[str],
    sleep: float,
    id: ID_TYPE | None,
) -> tuple[threading.Thread, Queue[tuple[str, ID_TYPE]], Queue[Exception]]:
    thread_output = Queue[tuple[str, ID_TYPE]]()
    thread_exc = Queue[Exception]()
    def waiter():
        time.sleep(sleep)
        try:
            thread_output.put(q.get(id=id))
        except Exception as e:
            thread_exc.put(e)
    
    t = threading.Thread(target=waiter)
    t.start()
    return t, thread_output, thread_exc

def test_sendBA_waitAB():
    q = StreamingQueue[str]()
    send_in_new_thread(q=q, sleep=0, id='A', data='a')
    send_in_new_thread(q=q, sleep=0, id='B', data='b')
    time.sleep(0.1)
    assert q.get(id='B') == ('b', 'B')
    assert q.get(id='A') == ('a', 'A')

def test_waitBA_sendAB():
    q = StreamingQueue[str]()
    send_in_new_thread(q=q, sleep=0.1, id='A', data='a')
    send_in_new_thread(q=q, sleep=0.2, id='B', data='b')
    assert q.get(id='B') == ('b', 'B')
    assert q.get(id='A') == ('a', 'A')

def test_waitB_sendA_waitA_sendB():
    q = StreamingQueue[str]()
    send_in_new_thread(q=q, sleep=0.1, id='A', data='a')
    send_in_new_thread(q=q, sleep=0.3, id='B', data='b')
    assert q.get(id='B') == ('b', 'B')
    time.sleep(0.2)
    assert q.get(id='A') == ('a', 'A')

def test_sendA_waitAA_sendA():
    q = StreamingQueue[str]()
    q.put(data='a1', id='A')
    send_in_new_thread(q=q, sleep=0.1, id='A', data='a2')
    assert q.get(id='A') == ('a1', 'A')
    assert q.get(id='A') == ('a2', 'A')

def test_waitXX_sendAB():
    q = StreamingQueue[str]()
    q.put(data='a', id='A')
    send_in_new_thread(q=q, sleep=0.1, id='B', data='b')
    assert q.get() == ('a', 'A')
    assert q.get() == ('b', 'B')

def test_sendAB_waitX():
    q = StreamingQueue[str]()
    q.put(data='a', id='A')
    send_in_new_thread(q=q, sleep=0.1, id='B', data='b')
    time.sleep(0.2)
    assert q.get() == ('a', 'A')

def test_sendB_waitA():
    q = StreamingQueue[str]()
    q.put(data='b', id='B')
    with pytest.raises(TimeoutError):
        q.get(id='a', timeout=0.2)

def test_waitA():
    q = StreamingQueue[str]()
    with pytest.raises(TimeoutError):
        q.get(id='a', timeout=0.2)

def test_waitX():
    q = StreamingQueue[str]()
    with pytest.raises(TimeoutError):
        q.get(timeout=0.2)

def test_sendA_waitAA():
    q = StreamingQueue[str]()
    q.put(data='a', id='A')
    assert q.get(id='A') == ('a', 'A')
    with pytest.raises(TimeoutError):
        q.get(id='a', timeout=0.2)

def test_send1000_wait1001():
    q = StreamingQueue[str]()
    for _ in range(1000):
        send_in_new_thread(q=q, sleep=0.1, id='A', data='a')
    for i in range(1000):
        assert q.get(id='A' if i % 2 == 0 else None) == ('a', 'A')
    with pytest.raises(TimeoutError):
        q.get(timeout=0.1)

def test_sendAEE_waitXA():
    q = StreamingQueue[str]()
    q.put(data='a', id='A')
    q.put_error(ZeroDivisionError())
    q.put_error(TypeError())
    with pytest.raises(RuntimeError) as exc_info:
        q.get()
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)
    with pytest.raises(RuntimeError) as exc_info:
        q.get(id='A')
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

def test_sendE_waitX_waitA():
    q = StreamingQueue[str]()
    q.put_error(err := ZeroDivisionError())
    _t1, t1_out, t1_exc = wait_in_new_thread(q=q, sleep=0.1, id=None)
    _t2, t2_out, t2_exc = wait_in_new_thread(q=q, sleep=0.1, id='A')
    with pytest.raises(Empty):
        t1_out.get_nowait()
    with pytest.raises(Empty):
        t2_out.get_nowait()
    assert t1_exc.get().__cause__ is err
    assert t2_exc.get().__cause__ is err

def test_waitA_sendE():
    q = StreamingQueue[str]()
    def sender():
        time.sleep(0.1)
        q.put_error(ZeroDivisionError())
    threading.Thread(target=sender).start()
    with pytest.raises(RuntimeError) as exc_info:
        q.get(id='A')
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)