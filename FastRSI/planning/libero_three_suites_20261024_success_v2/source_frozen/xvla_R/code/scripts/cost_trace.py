"""Small append-only span ledger, usable by Python 3.8 simulator and 3.11 OS."""
import contextlib
import contextvars
import json
import os
import time
import uuid

_parent = contextvars.ContextVar('fastrsi_span_parent', default=None)


def emit(row):
    path = os.environ.get('FASTRSI_TRACE')
    if not path:
        return
    row = dict(row, process=os.getpid(), wall_ns=time.time_ns())
    data = (json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise OSError('Partial cost ledger write')
    finally:
        os.close(fd)


@contextlib.contextmanager
def span(name, **fields):
    span_id = uuid.uuid4().hex
    parent = _parent.get()
    token = _parent.set(span_id)
    begin = time.perf_counter_ns()
    emit(dict(event='begin',name=name,span_id=span_id,parent_id=parent,mono_ns=begin,**fields))
    status = 'ok'
    try:
        yield
    except BaseException:
        status = 'error'
        raise
    finally:
        end = time.perf_counter_ns()
        emit(dict(event='end',name=name,span_id=span_id,parent_id=parent,mono_ns=end,
                  duration_seconds=(end-begin)/1e9,status=status,**fields))
        _parent.reset(token)
