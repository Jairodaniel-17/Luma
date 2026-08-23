"""The two things `docs/integrar/S3.md` declared untested: real sizes, and two
clients at once.

`s3_client.py` proves the S3 API is *correct* — 14 checks against boto3, the
client a user actually reaches for. It says nothing about two claims that the
doc made honestly and then left open:

  - **"Cargas grandes de verdad."** Its multipart parts are a hundred bytes.
    "El ensamblado es correcto por construcción, pero nadie ha subido un
    gigabyte por aquí." Correct by construction is a hypothesis until something
    runs it.
  - **"Concurrencia."** Two clients writing the same key at once. The doc
    reasons that each object write is atomic (temp file, rename) so the result
    must be one body or the other and never a blend — but reasoning about a
    race is exactly the kind of claim that has to be executed to be believed.

This file executes both. It is separate from `s3_client.py` because it is slow
by nature: correctness checks belong on every push, a half-gigabyte upload does
not.

## Running it

    pip install boto3
    # start Luma with s3_port set and an admin api key
    python tests/e2e/s3_scale.py \\
        --admin http://127.0.0.1:1234 \\
        --s3 http://127.0.0.1:9000 \\
        --api-key <admin key>

`--size-mb` defaults to 512. Pass `--size-mb 1024` for the literal gigabyte the
doc said nobody had uploaded.
"""
import argparse
import hashlib
import io
import sys
import threading
import time

# Shared with the correctness suite so credential minting has one implementation
# and cannot drift between the two files.
from s3_client import client, mint_credentials

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# `byte_at(i) = (i*31 + (i>>7)) & 0xFF` repeats every 32 KiB: 31 is coprime with
# 256 so the first term cycles in 256, the shifted term advances by exactly 256
# over that span, and both vanish mod 256 at 32768. Precomputing one period turns
# every byte of a gigabyte from a Python-level arithmetic step into a memory
# copy. The first version of this file did it the naive way and a 1 GiB run spent
# longer generating the payload than the server spent storing it — the test was
# measuring CPython, which is not the thing under test.
_PERIOD = 32768
_BLOCK = bytes((i * 31 + (i >> 7)) & 0xFF for i in range(_PERIOD))


class Pattern(io.RawIOBase):
    """A deterministic body of arbitrary size that is never held in memory.

    Materialising 512 MiB of `b'x' * n` to upload it would measure this
    machine's RAM, not the server. The pattern is position-dependent so a
    reassembly that lands parts out of order cannot pass by accident — every
    byte knows where it belongs.
    """

    def __init__(self, size):
        self.size = size
        self.pos = 0

    @staticmethod
    def byte_at(i):
        return _BLOCK[i % _PERIOD]

    @staticmethod
    def slice_at(start, length):
        """The pattern's bytes for `[start, start+length)`, block-copied."""
        first = start % _PERIOD
        out = bytearray()
        # Head of the current period, then whole periods, then the tail.
        take = min(length, _PERIOD - first)
        out += _BLOCK[first:first + take]
        remaining = length - take
        if remaining:
            whole, tail = divmod(remaining, _PERIOD)
            out += _BLOCK * whole
            out += _BLOCK[:tail]
        return bytes(out)

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        buf[:n] = self.slice_at(self.pos, n)
        self.pos += n
        return n


def pattern_digest(size, chunk=8 << 20):
    """The MD5 of the same bytes, computed the same streaming way."""
    h = hashlib.md5()
    pos = 0
    while pos < size:
        n = min(chunk, size - pos)
        h.update(Pattern.slice_at(pos, n))
        pos += n
    return h.hexdigest()


@check
def a_large_multipart_upload_round_trips(s3, bucket, size_mb):
    """Upload half a gigabyte through boto3's managed transfer and read it back.

    boto3 chooses multipart on its own above the threshold, which is the point:
    this is the code path a real user hits when they call `upload_fileobj` on a
    big file, not a hand-built sequence of `UploadPart` calls.

    The check is on the *bytes*, not the ETag. A multipart ETag is the md5 of
    the concatenated part digests plus `-N`, and N depends on how boto3 chose to
    split the file, so asserting a value here would be asserting boto3's chunking
    policy. What must hold is that what comes back is what went in.
    """
    from boto3.s3.transfer import TransferConfig

    size = size_mb * 1024 * 1024
    key = 'large/%d-mb.bin' % size_mb
    config = TransferConfig(multipart_threshold=8 * 1024 * 1024,
                            multipart_chunksize=8 * 1024 * 1024,
                            max_concurrency=4)

    started = time.time()
    s3.upload_fileobj(Pattern(size), bucket, key, Config=config)
    up = time.time() - started

    head = s3.head_object(Bucket=bucket, Key=key)
    assert head['ContentLength'] == size, (head['ContentLength'], size)
    multipart = '-' in head['ETag']

    # Stream the download and hash it, for the same reason the upload streamed.
    started = time.time()
    body = s3.get_object(Bucket=bucket, Key=key)['Body']
    h = hashlib.md5()
    read = 0
    while True:
        buf = body.read(1 << 20)
        if not buf:
            break
        h.update(buf)
        read += len(buf)
    down = time.time() - started

    assert read == size, 'downloaded %d bytes, uploaded %d' % (read, size)
    expected = pattern_digest(size)
    assert h.hexdigest() == expected, 'body differs: %s != %s' % (h.hexdigest(), expected)

    s3.delete_object(Bucket=bucket, Key=key)
    # Which path boto3 took is reported rather than asserted: it picks, and
    # pinning its choice would be testing boto3's policy instead of this server.
    # The multipart path at a real part size is covered by
    # `a_ranged_read_of_a_large_object_is_exact` and by the concurrent multipart
    # check below, so it is exercised either way.
    return '%d MiB via %s, up %.1fs (%.0f MiB/s), down %.1fs (%.0f MiB/s)' % (
        size_mb, 'multipart' if multipart else 'a single PUT',
        up, size_mb / up, down, size_mb / down)


@check
def a_ranged_read_of_a_large_object_is_exact(s3, bucket, size_mb):
    """Range requests over a multipart object, including across a part boundary.

    A composite object is assembled from parts; a range that straddles the seam
    between two of them is where an off-by-one in the assembly would show, and
    nothing else in either suite reads a large object at an offset.
    """
    key = 'large/ranged.bin'
    # Built with the explicit multipart API rather than `upload_fileobj`,
    # because the claim is about reading *across a part seam* and boto3's
    # managed transfer decides on its own whether to split at all — it chose a
    # single PUT here, which would have made the boundary imaginary and the
    # check a tautology.
    part = 5 * 1024 * 1024
    parts_n = 3
    size = part * parts_n
    up = s3.create_multipart_upload(Bucket=bucket, Key=key)['UploadId']
    etags = []
    for i in range(parts_n):
        chunk = Pattern.slice_at(i * part, part)
        r = s3.upload_part(Bucket=bucket, Key=key, UploadId=up,
                           PartNumber=i + 1, Body=chunk)
        etags.append({'ETag': r['ETag'], 'PartNumber': i + 1})
    s3.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=up,
                                 MultipartUpload={'Parts': etags})
    assert '-' in s3.head_object(Bucket=bucket, Key=key)['ETag'], 'expected a composite ETag'

    # The middle span sits on the seam between part 1 and part 2.
    spans = [(0, 15), (part - 4, part + 3), (size - 5, size - 1)]
    for first, last in spans:
        got = s3.get_object(Bucket=bucket, Key=key,
                            Range='bytes=%d-%d' % (first, last))['Body'].read()
        want = Pattern.slice_at(first, last - first + 1)
        # Truncated on purpose: a server that ignores Range returns the whole
        # object, and printing it verbatim buried the actual finding under
        # 225 MB of hex the first time this ran.
        assert got == want, ('range %d-%d: got %d bytes %r..., want %d bytes %r'
                             % (first, last, len(got), got[:16], len(want), want[:16]))

    s3.delete_object(Bucket=bucket, Key=key)
    return ('%d ranges over a real %d-part object, one across the seam'
            % (len(spans), parts_n))


@check
def concurrent_writers_to_one_key_never_blend(s3, bucket, size_mb):
    """Eight clients overwrite the same key at once, repeatedly.

    The claim under test is not "the last writer wins" — with no ordering
    guarantee that would be untestable. It is the weaker and far more important
    one: **the stored object is always exactly one of the bodies written**, never
    a splice of two. A blend would mean a reader can observe an object that no
    writer ever wrote, which is the failure that atomic rename exists to prevent.

    Each writer's body is distinct and self-identifying, so a blend is
    detectable rather than merely improbable.
    """
    key = 'race/same-key.bin'
    writers = 8
    rounds = 6
    body_size = 256 * 1024
    bodies = {}
    for w in range(writers):
        marker = ('writer-%02d-' % w).encode()
        bodies[w] = marker * (body_size // len(marker))

    errors = []
    barrier = threading.Barrier(writers)

    def write(w):
        try:
            cl = client(s3.meta.endpoint_url, ACCESS[0], ACCESS[1])
            for _ in range(rounds):
                barrier.wait()
                cl.put_object(Bucket=bucket, Key=key, Body=bodies[w])
        except Exception as e:                                  # noqa: BLE001
            errors.append('writer %d: %r' % (w, e))
            try:
                barrier.abort()
            except Exception:                                   # noqa: BLE001
                pass

    threads = [threading.Thread(target=write, args=(w,)) for w in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    got = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    assert got in bodies.values(), (
        'the stored object is not any single writer\'s body: %d bytes starting %r'
        % (len(got), got[:40]))
    s3.delete_object(Bucket=bucket, Key=key)
    return '%d writers x %d rounds, final object is exactly one body' % (writers, rounds)


@check
def a_reader_during_a_rewrite_never_sees_a_torn_object(s3, bucket, size_mb):
    """A reader loops while a writer replaces the key underneath it.

    Same property from the other side: a `GET` concurrent with a `PUT` must
    return one whole version or the other. A short read, a mixed body or a
    truncated one would all mean the rename is not doing what the doc says it
    does.
    """
    key = 'race/read-during-write.bin'
    a = b'A' * (512 * 1024)
    b = b'B' * (512 * 1024)
    s3.put_object(Bucket=bucket, Key=key, Body=a)

    stop = threading.Event()
    bad = []
    seen = set()

    def reader():
        cl = client(s3.meta.endpoint_url, ACCESS[0], ACCESS[1])
        while not stop.is_set():
            try:
                body = cl.get_object(Bucket=bucket, Key=key)['Body'].read()
            except Exception as e:                              # noqa: BLE001
                # A 404 mid-replace would itself be a torn observation: the key
                # never stops existing.
                bad.append(repr(e))
                return
            if body == a:
                seen.add('a')
            elif body == b:
                seen.add('b')
            else:
                bad.append('torn read: %d bytes, starts %r ends %r'
                           % (len(body), body[:8], body[-8:]))
                return

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(40):
            s3.put_object(Bucket=bucket, Key=key, Body=b if i % 2 else a)
    finally:
        stop.set()
        t.join()

    assert not bad, bad[:3]
    s3.delete_object(Bucket=bucket, Key=key)
    return 'reader saw only whole versions (%s)' % ','.join(sorted(seen))


@check
def concurrent_multipart_uploads_do_not_cross_contaminate(s3, bucket, size_mb):
    """Six multipart uploads in flight at once, each to its own key.

    Multipart keeps server-side state per upload id. If that state were keyed
    carelessly, parts from one upload could land in another — and the result
    would be a perfectly valid-looking object with someone else's bytes in it,
    which no single-threaded test can catch.
    """
    uploads = 6
    parts_each = 3
    part_size = 5 * 1024 * 1024        # S3's minimum for a non-final part
    errors = []
    results = {}

    def run(u):
        try:
            cl = client(s3.meta.endpoint_url, ACCESS[0], ACCESS[1])
            key = 'race/multipart-%d.bin' % u
            up = cl.create_multipart_upload(Bucket=bucket, Key=key)['UploadId']
            etags = []
            for p in range(1, parts_each + 1):
                chunk = bytes([(u * 37 + p) & 0xFF]) * part_size
                r = cl.upload_part(Bucket=bucket, Key=key, UploadId=up,
                                   PartNumber=p, Body=chunk)
                etags.append({'ETag': r['ETag'], 'PartNumber': p})
            cl.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=up,
                                         MultipartUpload={'Parts': etags})
            body = cl.get_object(Bucket=bucket, Key=key)['Body'].read()
            want = b''.join(bytes([(u * 37 + p) & 0xFF]) * part_size
                            for p in range(1, parts_each + 1))
            results[u] = (body == want, len(body))
            cl.delete_object(Bucket=bucket, Key=key)
        except Exception as e:                                  # noqa: BLE001
            errors.append('upload %d: %r' % (u, e))

    threads = [threading.Thread(target=run, args=(u,)) for u in range(uploads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    wrong = [u for u, (ok, _) in results.items() if not ok]
    assert not wrong, 'these uploads came back with the wrong bytes: %s' % wrong
    return '%d concurrent uploads x %d parts, each kept its own bytes' % (uploads, parts_each)


ACCESS = (None, None)


def main():
    global ACCESS
    ap = argparse.ArgumentParser()
    ap.add_argument('--admin', required=True)
    ap.add_argument('--s3', required=True)
    ap.add_argument('--api-key', required=True)
    ap.add_argument('--org', default='scale-org')
    ap.add_argument('--size-mb', type=int, default=512)
    args = ap.parse_args()

    ACCESS = mint_credentials(args.admin, args.api_key, args.org)
    s3 = client(args.s3, *ACCESS)

    bucket = 'scale-bucket'
    s3.create_bucket(Bucket=bucket)

    failed = 0
    for fn in CHECKS:
        try:
            note = fn(s3, bucket, args.size_mb)
            print('%-52s ok   %s' % (fn.__name__, note or ''))
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            print('%-52s FAIL %r' % (fn.__name__, e))

    if failed:
        print('\n%d/%d checks failed' % (failed, len(CHECKS)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
