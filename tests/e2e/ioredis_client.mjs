/**
 * End-to-end against ioredis — the Node client the README promises works.
 *
 * `clients.py` covers redis-py, Celery, kombu and arq, which is every Python
 * client a user would reach for. The README's adoption claim names one more:
 *
 *   "que Celery, arq, redis-py o ioredis apunten a Luma sin cambiar código"
 *
 * and ioredis had never been run against it. A promise with no test behind it is
 * a guess, and ioredis is not a thin client: it does its own RESP parsing, its
 * own pipelining, its own `INFO`-driven server detection and its own reconnect
 * logic. Exactly the places where a reimplementation of the protocol diverges.
 *
 * Both Redis and Luma are driven, for the same reason `clients.py` does it: a
 * failure against Redis means this harness is wrong, not the server.
 *
 * ## Running it
 *
 *     npm install ioredis
 *     docker run -d --name luma-diff-redis -p 16379:6379 redis:7-alpine
 *     ./target/release/luma serve          # with resp_port set
 *
 *     node tests/e2e/ioredis_client.mjs \
 *         --redis redis://127.0.0.1:16379/0 \
 *         --luma  redis://127.0.0.1:16380/0
 */
import Redis from 'ioredis';
import assert from 'node:assert/strict';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const CHECKS = [];
const check = (name, fn) => CHECKS.push({ name, fn });

// ── strings, counters and expiry ────────────────────────────────────────────
check('strings and counters', async (r, p) => {
  await r.set(`${p}:s`, 'hello');
  assert.equal(await r.get(`${p}:s`), 'hello');
  assert.equal(await r.append(`${p}:s`, ' world'), 11);
  assert.equal(await r.get(`${p}:s`), 'hello world');
  assert.equal(await r.strlen(`${p}:s`), 11);

  assert.equal(await r.incr(`${p}:n`), 1);
  assert.equal(await r.incrby(`${p}:n`, 41), 42);
  assert.equal(await r.decr(`${p}:n`), 41);
  // ioredis returns bumped floats as strings, like Redis does on the wire.
  assert.equal(await r.incrbyfloat(`${p}:f`, '1.5'), '1.5');

  // A missing key is null, not an empty string — the distinction every client
  // branches on.
  assert.equal(await r.get(`${p}:absent`), null);
});

check('SET options', async (r, p) => {
  assert.equal(await r.set(`${p}:nx`, 'first', 'NX'), 'OK');
  assert.equal(await r.set(`${p}:nx`, 'second', 'NX'), null);
  assert.equal(await r.get(`${p}:nx`), 'first');
  assert.equal(await r.set(`${p}:nx`, 'third', 'XX'), 'OK');

  await r.set(`${p}:ttl`, 'v', 'EX', 100);
  const ttl = await r.ttl(`${p}:ttl`);
  assert.ok(ttl > 90 && ttl <= 100, `ttl was ${ttl}`);
  assert.equal(await r.persist(`${p}:ttl`), 1);
  assert.equal(await r.ttl(`${p}:ttl`), -1);
  // A key that does not exist reports -2, not -1.
  assert.equal(await r.ttl(`${p}:nothing`), -2);
});

// ── the structures ──────────────────────────────────────────────────────────
check('lists', async (r, p) => {
  await r.del(`${p}:l`);
  await r.rpush(`${p}:l`, 'a', 'b', 'c');
  await r.lpush(`${p}:l`, 'z');
  assert.deepEqual(await r.lrange(`${p}:l`, 0, -1), ['z', 'a', 'b', 'c']);
  assert.equal(await r.llen(`${p}:l`), 4);
  assert.equal(await r.lindex(`${p}:l`, 1), 'a');
  assert.equal(await r.lpop(`${p}:l`), 'z');
  assert.equal(await r.rpop(`${p}:l`), 'c');
  await r.ltrim(`${p}:l`, 0, 0);
  assert.deepEqual(await r.lrange(`${p}:l`, 0, -1), ['a']);
});

check('hashes', async (r, p) => {
  await r.del(`${p}:h`);
  await r.hset(`${p}:h`, 'one', '1', 'two', '2');
  assert.equal(await r.hget(`${p}:h`, 'one'), '1');
  assert.deepEqual(await r.hgetall(`${p}:h`), { one: '1', two: '2' });
  assert.deepEqual((await r.hkeys(`${p}:h`)).sort(), ['one', 'two']);
  assert.equal(await r.hlen(`${p}:h`), 2);
  assert.equal(await r.hincrby(`${p}:h`, 'one', 9), 10);
  assert.equal(await r.hdel(`${p}:h`, 'two'), 1);
  assert.equal(await r.hexists(`${p}:h`, 'two'), 0);
});

check('sets', async (r, p) => {
  await r.del(`${p}:set`);
  assert.equal(await r.sadd(`${p}:set`, 'a', 'b', 'c'), 3);
  assert.equal(await r.scard(`${p}:set`), 3);
  assert.equal(await r.sismember(`${p}:set`, 'b'), 1);
  assert.deepEqual((await r.smembers(`${p}:set`)).sort(), ['a', 'b', 'c']);
  assert.equal(await r.srem(`${p}:set`, 'a'), 1);
  assert.equal(await r.scard(`${p}:set`), 2);
});

check('sorted sets', async (r, p) => {
  await r.del(`${p}:z`);
  await r.zadd(`${p}:z`, 1, 'one', 2, 'two', 3, 'three');
  assert.equal(await r.zcard(`${p}:z`), 3);
  assert.deepEqual(await r.zrange(`${p}:z`, 0, -1), ['one', 'two', 'three']);
  assert.deepEqual(await r.zrevrange(`${p}:z`, 0, 0), ['three']);
  assert.equal(await r.zscore(`${p}:z`, 'two'), '2');
  assert.equal(await r.zrank(`${p}:z`, 'two'), 1);
  // WITHSCORES flattens to [member, score, ...], which is the wire shape.
  assert.deepEqual(await r.zrange(`${p}:z`, 0, 0, 'WITHSCORES'), ['one', '1']);
  assert.equal(await r.zrem(`${p}:z`, 'one'), 1);
});

// ── the parts ioredis implements itself ─────────────────────────────────────
check('pipelining', async (r, p) => {
  // ioredis batches these into one write and demultiplexes the replies itself.
  // A server that answered out of order, or merged two replies, would show up
  // here and nowhere else.
  const results = await r
    .pipeline()
    .set(`${p}:pipe`, 'v')
    .incr(`${p}:pipecount`)
    .get(`${p}:pipe`)
    .exec();
  assert.deepEqual(results.map(([err]) => err), [null, null, null]);
  assert.deepEqual(results.map(([, value]) => value), ['OK', 1, 'v']);
});

check('MULTI/EXEC through ioredis', async (r, p) => {
  const results = await r.multi().set(`${p}:tx`, '1').incr(`${p}:tx`).exec();
  assert.deepEqual(results.map(([, v]) => v), ['OK', 2]);

  // A queued command that fails at execution time reports per-command, and the
  // rest of the transaction still runs.
  await r.set(`${p}:notanumber`, 'abc');
  const mixed = await r.multi().incr(`${p}:notanumber`).set(`${p}:after`, 'ok').exec();
  assert.ok(mixed[0][0] instanceof Error, 'INCR on a string must report an error');
  assert.equal(mixed[1][1], 'OK');
});

check('scan iteration', async (r, p) => {
  for (let i = 0; i < 40; i++) await r.set(`${p}:scan:${i}`, String(i));
  // ioredis drives SCAN with its own cursor loop; a cursor that never returned
  // to 0 would hang here rather than fail.
  const found = new Set();
  let cursor = '0';
  do {
    const [next, keys] = await r.scan(cursor, 'MATCH', `${p}:scan:*`, 'COUNT', 7);
    cursor = next;
    keys.forEach((k) => found.add(k));
  } while (cursor !== '0');
  assert.equal(found.size, 40, `found ${found.size} of 40`);
});

check('pub/sub with a second connection', async (r, p, makeClient) => {
  const sub = makeClient();
  try {
    const received = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('no message arrived')), 5000);
      sub.on('message', (channel, message) => {
        clearTimeout(timer);
        resolve({ channel, message });
      });
    });
    await sub.subscribe(`${p}:chan`);
    // Published after the subscribe is acknowledged, or the message can race
    // ahead of the subscription and this would flake rather than fail.
    await r.publish(`${p}:chan`, 'hello');
    const got = await received;
    assert.equal(got.channel, `${p}:chan`);
    assert.equal(got.message, 'hello');
  } finally {
    sub.disconnect();
  }
});

check('blocking pop wakes on a push', async (r, p, makeClient) => {
  const blocker = makeClient();
  try {
    await r.del(`${p}:bl`);
    const waiting = blocker.blpop(`${p}:bl`, 5);
    // The push happens after the BLPOP is on the wire; a server that answered
    // the block immediately with nil would fail the assertion below.
    await new Promise((done) => setTimeout(done, 200));
    await r.rpush(`${p}:bl`, 'woken');
    assert.deepEqual(await waiting, [`${p}:bl`, 'woken']);
  } finally {
    blocker.disconnect();
  }
});

check('type errors keep their shape', async (r, p) => {
  await r.del(`${p}:str`);
  await r.set(`${p}:str`, 'v');
  // ioredis surfaces a server error as a rejected promise whose message starts
  // with the error token. Clients branch on that token, so it has to be the one
  // Redis uses.
  await assert.rejects(
    () => r.lpush(`${p}:str`, 'x'),
    (e) => /WRONGTYPE/.test(e.message),
    'pushing to a string must be WRONGTYPE',
  );
  assert.equal(await r.type(`${p}:str`), 'string');
});

async function run(label, url) {
  const password = new URL(url).password || undefined;
  const makeClient = () =>
    new Redis(url, {
      // Fail fast instead of retrying forever when the server is not there:
      // a harness that hangs teaches nothing.
      maxRetriesPerRequest: 2,
      retryStrategy: (times) => (times > 3 ? null : 200),
      password,
    });

  // Without this an unhandled 'error' event takes the whole process down and
  // the remaining checks never report. A harness that dies mid-run tells you
  // less than one that finishes with failures.
  const guarded = () => {
    const c = makeClient();
    c.on('error', () => {});
    return c;
  };

  const client = guarded();
  const prefix = `ioredis-e2e:${Date.now()}`;
  let failed = 0;

  for (const { name, fn } of CHECKS) {
    try {
      await fn(client, prefix, guarded);
      console.log(`  ${label.padEnd(6)} ${name.padEnd(38)} ok`);
    } catch (e) {
      failed++;
      console.log(`  ${label.padEnd(6)} ${name.padEnd(38)} FAIL ${e.message}`);
    }
  }

  // Leave nothing behind, so a rerun against the same server is clean.
  try {
    const keys = await client.keys(`${prefix}*`);
    if (keys.length) await client.del(...keys);
  } catch {
    // A connection already broken by a failing check cannot clean up; the keys
    // are prefixed with a timestamp so they cannot collide with a later run.
  }
  client.disconnect();
  return failed;
}

const redisUrl = flag('--redis', null);
const lumaUrl = flag('--luma', null);
if (!lumaUrl) {
  console.error('usage: ioredis_client.mjs [--redis <url>] --luma <url>');
  process.exit(2);
}

let redisFailures = 0;
if (redisUrl) {
  console.log('── ioredis against a real Redis 7 ───────────────────────────────');
  redisFailures = await run('redis', redisUrl);
}
console.log('── ioredis against Luma ─────────────────────────────────────────');
const lumaFailures = await run('luma', lumaUrl);

if (redisFailures) {
  // Same rule as `clients.py`: a failure here means this harness is wrong about
  // Redis, not that Luma is broken, so it must not be reported as Luma's fault.
  console.log(`\n${redisFailures} check(s) failed against real Redis — the harness is wrong, not the server`);
}
if (lumaFailures) {
  console.log(`\n${lumaFailures} of ${CHECKS.length} checks failed against Luma`);
}
process.exit(lumaFailures ? 1 : 0);
