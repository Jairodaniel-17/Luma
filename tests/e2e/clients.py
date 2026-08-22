"""End-to-end against the real Python clients, not a model of them.

F3.4 of `docs/PLAN-MAESTRO.md` — the adoption milestone. Everything else in the
RESP work is verified against tests we wrote, which encode our understanding of
Redis. This runs the actual libraries a user would install.

It earned its place immediately: the first run found that a `PUBLISH` queued
inside `MULTI` was rejected as an unknown command. redis-py wraps every pipeline
in MULTI/EXEC and Celery's result backend stores its result with a pipelined
`SETEX` + `PUBLISH`, so a real worker consumed the task, executed it, and then
the caller hung forever waiting for a result that was never written. Nothing in
the unit tests or the differential corpus covered a transaction, so nothing
caught it.

## Running it

    pip install redis celery arq
    docker run -d --name luma-diff-redis -p 16379:6379 redis:7-alpine
    ./target/release/luma serve            # with resp_port set

    python tests/e2e/clients.py \\
        --redis redis://127.0.0.1:16379/0 \\
        --luma  redis://127.0.0.1:16380/0

Both URLs are exercised so a failure can be told apart from "this client needs
something neither Redis nor Luma provides". Only failures against Luma set the
exit code; a failure against Redis means the harness itself is wrong.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# ── redis-py: the client every other one is built on ────────────────────────
def redis_py_basics(url):
    import redis
    r = redis.from_url(url)
    r.flushdb()
    assert r.ping()
    r.set('k', 'v')
    assert r.get('k') == b'v'
    assert r.type('k') == b'string'
    r.delete('k')

    r.rpush('q', 'a', 'b', 'c')
    assert r.llen('q') == 3
    assert r.lrange('q', 0, -1) == [b'a', b'b', b'c']
    assert r.type('q') == b'list'
    assert r.exists('q') == 1

    # The pipeline path: several commands sent together, several replies parsed
    # at once. A reply-count mismatch desynchronises the client.
    with r.pipeline() as pipe:
        pipe.lpop('q')
        pipe.rpop('q')
        pipe.llen('q')
        assert pipe.execute() == [b'a', b'c', 1]

    r.hset('h', mapping={'f1': 'v1', 'f2': 'v2'})
    assert r.hgetall('h') == {b'f1': b'v1', b'f2': b'v2'}
    r.sadd('s', 'x', 'y')
    assert r.scard('s') == 2
    r.zadd('z', {'a': 1, 'b': 2})
    assert r.zrange('z', 0, -1, withscores=True) == [(b'a', 1.0), (b'b', 2.0)]
    assert r.zrangebyscore('z', 1, 1) == [b'a']

    # WRONGTYPE has to surface as the exception redis-py raises for it.
    try:
        r.lpush('h', 'x')
        raise AssertionError('a list push onto a hash must raise')
    except redis.ResponseError as e:
        assert 'WRONGTYPE' in str(e), str(e)
    r.flushdb()


def redis_py_result_backend_pipeline(url):
    """The exact shape Celery's result backend writes with.

    Isolated because when it broke, the symptom was a worker that hung — three
    layers away from the cause.
    """
    import redis
    r = redis.from_url(url)
    r.flushdb()
    key = 'celery-task-meta-e2e'
    with r.pipeline() as pipe:
        pipe.setex(key, 60, 'payload')
        pipe.publish(key, 'payload')
        pipe.execute()
    assert r.get(key) == b'payload', 'the result was not stored'
    assert 0 < r.ttl(key) <= 60
    r.flushdb()


def redis_py_transactions(url):
    import redis
    r = redis.from_url(url)
    r.flushdb()
    r.set('counter', '1')
    with r.pipeline() as pipe:
        pipe.watch('counter')
        current = int(pipe.get('counter'))
        pipe.multi()
        pipe.set('counter', current + 1)
        pipe.execute()
    assert r.get('counter') == b'2', r.get('counter')
    r.flushdb()


def redis_py_blocking(url):
    import redis
    r = redis.from_url(url)
    r.flushdb()
    served = []

    def worker():
        w = redis.from_url(url)
        served.append(w.blpop('jobs', timeout=5))

    t = threading.Thread(target=worker)
    t.start()
    # Park first, so this exercises the wakeup rather than the already-there
    # fast path.
    time.sleep(0.3)
    r.rpush('jobs', 'payload')
    t.join(timeout=8)
    assert served and served[0] == (b'jobs', b'payload'), served
    r.flushdb()


def redis_py_pubsub(url):
    import redis
    r = redis.from_url(url)
    sub = r.pubsub()
    sub.subscribe('events')
    assert sub.get_message(timeout=3)['type'] == 'subscribe'
    r.publish('events', 'hello')
    deadline = time.time() + 3
    got = None
    while time.time() < deadline:
        m = sub.get_message(timeout=1)
        if m and m['type'] == 'message':
            got = m
            break
    assert got and got['data'] == b'hello', got
    sub.close()


def redis_py_scan(url):
    import redis
    r = redis.from_url(url)
    r.flushdb()
    for i in range(25):
        r.set('scan:%d' % i, i)
    # A structure has to be listed by SCAN under its own name, like any key.
    r.rpush('scan:list', 'x')
    seen = set()
    for key in r.scan_iter(match='scan:*', count=5):
        seen.add(key)
    assert len(seen) == 26, 'expected 26 keys, saw %d' % len(seen)
    assert not any(k.startswith(b'struct:') for k in seen), \
        'the storage prefix leaked to the client: %r' % sorted(seen)[:5]

    r.hset('bighash', mapping={('f%d' % i): i for i in range(30)})
    fields = dict(r.hscan_iter('bighash', count=5))
    assert len(fields) == 30, len(fields)
    r.flushdb()


# ── kombu: the transport Celery actually speaks ─────────────────────────────
def kombu_roundtrip(url):
    from kombu import Connection, Exchange, Queue
    exchange = Exchange('luma-e2e', type='direct')
    queue = Queue('luma-e2e', exchange, routing_key='luma-e2e')
    with Connection(url) as conn:
        conn.Producer(serializer='json').publish(
            {'task': 'add', 'args': [2, 3]},
            exchange=exchange, routing_key='luma-e2e', declare=[queue])
        got = []

        def handle(body, message):
            got.append(body)
            message.ack()

        with conn.Consumer(queue, callbacks=[handle]):
            deadline = time.time() + 8
            while not got and time.time() < deadline:
                try:
                    conn.drain_events(timeout=1)
                except Exception:
                    pass
        assert got and got[0]['args'] == [2, 3], got


# ── celery: a real worker, executing a real task ────────────────────────────
def celery_worker_roundtrip(url):
    """The claim that matters.

    Not "the broker accepted a message" but "a worker consumed it, executed the
    task, and the caller got the result back". `--pool=solo` because the
    prefork pool does not work on Windows.
    """
    env = dict(os.environ, BROKER_URL=url, PYTHONPATH=HERE)
    worker = subprocess.Popen(
        [sys.executable, '-m', 'celery', '-A', 'celery_app', 'worker',
         '--pool=solo', '--loglevel=warning', '--without-gossip',
         '--without-mingle', '--without-heartbeat'],
        cwd=HERE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        sys.path.insert(0, HERE)
        os.environ['BROKER_URL'] = url
        for module in ('celery_app',):
            sys.modules.pop(module, None)
        from celery_app import add

        time.sleep(6)
        if worker.poll() is not None:
            raise AssertionError(
                'the worker exited before consuming:\n%s'
                % worker.stdout.read()[-2000:])

        value = add.delay(2, 3).get(timeout=30)
        assert value == 5, 'expected 5, got %r' % value
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()


# ── arq: asyncio, and a different command mix ───────────────────────────────
def arq_enqueue(url):
    import asyncio
    from arq import create_pool
    from arq.connections import RedisSettings

    async def run():
        pool = await create_pool(RedisSettings.from_dsn(url))
        job = await pool.enqueue_job('noop', 1, 2)
        assert job is not None, 'enqueue_job returned None'
        # arq keeps its queue in a sorted set.
        queued = await pool.zcard('arq:queue')
        assert queued >= 1, 'the job did not land in arq:queue (zcard=%s)' % queued
        await pool.aclose()

    asyncio.run(run())


CHECKS = [
    ('redis-py basics', redis_py_basics),
    ('redis-py result-backend pipeline', redis_py_result_backend_pipeline),
    ('redis-py MULTI/WATCH', redis_py_transactions),
    ('redis-py BLPOP', redis_py_blocking),
    ('redis-py pub/sub', redis_py_pubsub),
    ('redis-py SCAN/HSCAN', redis_py_scan),
    ('kombu roundtrip', kombu_roundtrip),
    ('celery worker roundtrip', celery_worker_roundtrip),
    ('arq enqueue', arq_enqueue),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--luma', required=True, help='redis:// URL of the Luma RESP port')
    parser.add_argument('--redis', help='redis:// URL of a real Redis, as a control')
    parser.add_argument('--only', help='substring of a check name, to run one')
    args = parser.parse_args()

    targets = [('luma', args.luma)]
    if args.redis:
        targets.insert(0, ('redis', args.redis))

    outcomes = {}
    for name, fn in CHECKS:
        if args.only and args.only not in name:
            continue
        for target, url in targets:
            try:
                fn(url)
                outcomes.setdefault(name, {})[target] = ('ok', '')
            except Exception as e:
                outcomes.setdefault(name, {})[target] = (
                    'FAIL', '%s: %s' % (type(e).__name__, e))

    if not outcomes:
        print('no checks matched --only %r' % args.only)
        return 1

    width = max(len(n) for n in outcomes) + 2
    header = ['%-*s' % (width, 'check')] + ['%-10s' % t for t, _ in targets]
    print(''.join(header))
    failures = []
    for name, per in outcomes.items():
        row = ['%-*s' % (width, name)]
        for target, _ in targets:
            status = per.get(target, ('-', ''))[0]
            row.append('%-10s' % status)
        print(''.join(row))
        if per.get('luma', ('-', ''))[0] == 'FAIL':
            failures.append((name, per['luma'][1]))

    for name, per in outcomes.items():
        for target, _ in targets:
            status, detail = per.get(target, ('-', ''))
            if status == 'FAIL':
                print('\n  [%s] %s\n      %s' % (target, name, detail))

    # Only Luma failures are fatal: a failure against Redis means this harness
    # is wrong, and saying so is more useful than failing the build for it.
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
