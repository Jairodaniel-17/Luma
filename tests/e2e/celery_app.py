"""A real Celery app whose broker and result backend are whatever BROKER_URL
points at. Used by both the worker subprocess and the producer."""
import os

from celery import Celery

BROKER = os.environ['BROKER_URL']

app = Celery('luma_e2e', broker=BROKER, backend=BROKER)
app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    broker_connection_retry_on_startup=True,
    result_expires=60,
)


@app.task(name='luma_e2e.add')
def add(a, b):
    return a + b


@app.task(name="luma_e2e.slow")
def slow(seconds):
    """Long enough to be killed while the worker still holds the message.

    Used by the unacked-recovery check, which needs a window in which the
    message has left the queue and has not been acked.
    """
    import time
    time.sleep(seconds)
    return seconds
