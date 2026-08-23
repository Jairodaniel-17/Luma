"""End-to-end against boto3, the client that decides whether the S3 API works.

W3.2 of `docs/PLAN-MAESTRO.md`. SigV4 has published test vectors and this repo
checks them, but a passing vector says the *cryptography* is right; it says
nothing about whether the canonical request agrees with what a real client
builds. Those are different claims and only one of them can be checked in a unit
test.

boto3 rather than MinIO's mint suite: mint tests a server against a matrix that
includes versioning, lifecycle and object locking, which this API deliberately
refuses. Running it would produce a long list of expected failures, and a
green-vs-red signal nobody would read. boto3 is the client an actual user reaches
for.

## Running it

    pip install boto3
    # start Luma with s3_port set and an admin api key
    python tests/e2e/s3_client.py \\
        --admin http://127.0.0.1:1234 \\
        --s3 http://127.0.0.1:9000 \\
        --api-key <admin key>
"""
import argparse
import io
import json
import sys
import urllib.error
import urllib.request


def mint_credentials(admin_url, api_key, org):
    """Ask the admin API for an access key.

    Through the real endpoint rather than by writing the table directly: if
    minting is broken, an S3 test that worked around it would be testing
    something no user can reach.
    """
    request = urllib.request.Request(
        admin_url.rstrip('/') + '/v1/admin/s3-credentials',
        method='POST',
        headers={'Authorization': 'Bearer ' + api_key,
                 'Content-Type': 'application/json'},
        # A platform-scoped caller (the static instance key) has no organization
        # of its own, so it has to name one. An org-scoped key may omit this.
        data=json.dumps({'org_id': org}).encode(),
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.load(response)
    return body['access_key_id'], body['secret_access_key']


def client(s3_url, access_key, secret_key):
    import boto3
    from botocore.config import Config
    return boto3.client(
        's3',
        endpoint_url=s3_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1',
        # Path style, because there is no DNS for `bucket.host` here. Virtual
        # host style would send the bucket in the Host header and every request
        # would 404.
        # SigV4 explicitly. boto3 still presigns with SigV2 by default, and
        # Luma only supports SigV4 — a deprecated scheme is not something to
        # implement so that a default keeps working.
        config=Config(signature_version='s3v4',
                      s3={'addressing_style': 'path'},
                      retries={'max_attempts': 1}),
    )


CHECKS = []

# Filled in by main(), for the checks that need a second client.
CREDENTIALS = (None, None)


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def create_and_list_buckets(s3):
    s3.create_bucket(Bucket='luma-e2e')
    names = [b['Name'] for b in s3.list_buckets().get('Buckets', [])]
    assert 'luma-e2e' in names, names


@check
def put_get_head_delete(s3):
    s3.put_object(Bucket='luma-e2e', Key='hello.txt', Body=b'hello world')

    got = s3.get_object(Bucket='luma-e2e', Key='hello.txt')
    assert got['Body'].read() == b'hello world'

    head = s3.head_object(Bucket='luma-e2e', Key='hello.txt')
    assert head['ContentLength'] == 11, head

    s3.delete_object(Bucket='luma-e2e', Key='hello.txt')
    from botocore.exceptions import ClientError
    try:
        s3.get_object(Bucket='luma-e2e', Key='hello.txt')
        raise AssertionError('a deleted object must not be readable')
    except ClientError as e:
        assert e.response['Error']['Code'] in ('NoSuchKey', '404'), e.response['Error']


@check
def a_key_with_slashes_and_spaces_round_trips(s3):
    """The case double URI encoding breaks, and most real keys look like this."""
    key = 'reports/2026 Q1/summary v2.csv'
    s3.put_object(Bucket='luma-e2e', Key=key, Body=b'data')
    got = s3.get_object(Bucket='luma-e2e', Key=key)
    assert got['Body'].read() == b'data'
    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def a_key_with_an_ampersand_survives_the_listing(s3):
    """Unescaped, the XML parses as truncated and the listing silently ends."""
    key = 'a&b.txt'
    s3.put_object(Bucket='luma-e2e', Key=key, Body=b'x')
    keys = [o['Key'] for o in s3.list_objects_v2(Bucket='luma-e2e').get('Contents', [])]
    assert key in keys, keys
    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def listing_honours_prefix_and_delimiter(s3):
    for key in ['logs/2025/a.log', 'logs/2026/b.log', 'other.txt']:
        s3.put_object(Bucket='luma-e2e', Key=key, Body=b'x')

    flat = s3.list_objects_v2(Bucket='luma-e2e', Prefix='logs/')
    assert sorted(o['Key'] for o in flat.get('Contents', [])) == [
        'logs/2025/a.log', 'logs/2026/b.log']

    grouped = s3.list_objects_v2(Bucket='luma-e2e', Prefix='logs/', Delimiter='/')
    prefixes = sorted(p['Prefix'] for p in grouped.get('CommonPrefixes', []))
    assert prefixes == ['logs/2025/', 'logs/2026/'], prefixes
    assert not grouped.get('Contents'), grouped.get('Contents')

    for key in ['logs/2025/a.log', 'logs/2026/b.log', 'other.txt']:
        s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def listing_paginates(s3):
    for i in range(5):
        s3.put_object(Bucket='luma-e2e', Key='page/%02d' % i, Body=b'x')

    seen = []
    token = None
    for _ in range(10):
        kwargs = {'Bucket': 'luma-e2e', 'Prefix': 'page/', 'MaxKeys': 2}
        if token:
            kwargs['ContinuationToken'] = token
        page = s3.list_objects_v2(**kwargs)
        seen.extend(o['Key'] for o in page.get('Contents', []))
        if not page.get('IsTruncated'):
            break
        token = page['NextContinuationToken']
    assert sorted(seen) == ['page/%02d' % i for i in range(5)], seen

    for i in range(5):
        s3.delete_object(Bucket='luma-e2e', Key='page/%02d' % i)


@check
def multipart_upload_assembles_in_order(s3):
    """The reason multipart exists: parts can arrive out of order.

    Uploaded backwards on purpose. A server that concatenated in arrival order
    would pass a sequential test and corrupt every real upload.
    """
    key = 'big.bin'
    created = s3.create_multipart_upload(Bucket='luma-e2e', Key=key)
    upload_id = created['UploadId']

    chunks = [b'A' * 100, b'B' * 100, b'C' * 100]
    parts = []
    for number in (3, 1, 2):
        result = s3.upload_part(
            Bucket='luma-e2e', Key=key, UploadId=upload_id,
            PartNumber=number, Body=chunks[number - 1])
        parts.append({'PartNumber': number, 'ETag': result['ETag']})

    parts.sort(key=lambda p: p['PartNumber'])
    s3.complete_multipart_upload(
        Bucket='luma-e2e', Key=key, UploadId=upload_id,
        MultipartUpload={'Parts': parts})

    got = s3.get_object(Bucket='luma-e2e', Key=key)['Body'].read()
    assert got == b''.join(chunks), 'assembled %d bytes in the wrong order' % len(got)
    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def an_aborted_multipart_upload_leaves_nothing(s3):
    key = 'abandoned.bin'
    created = s3.create_multipart_upload(Bucket='luma-e2e', Key=key)
    s3.upload_part(Bucket='luma-e2e', Key=key, UploadId=created['UploadId'],
                   PartNumber=1, Body=b'x' * 10)
    s3.abort_multipart_upload(Bucket='luma-e2e', Key=key, UploadId=created['UploadId'])

    from botocore.exceptions import ClientError
    try:
        s3.get_object(Bucket='luma-e2e', Key=key)
        raise AssertionError('an aborted upload must not produce an object')
    except ClientError:
        pass


@check
def the_etag_is_the_md5_of_the_body(s3):
    """A client that verifies a download recomputes this."""
    import hashlib
    body = b'the quick brown fox'
    put = s3.put_object(Bucket='luma-e2e', Key='etag.txt', Body=body)
    expected = hashlib.md5(body).hexdigest()
    assert put['ETag'].strip('"') == expected, (put['ETag'], expected)

    head = s3.head_object(Bucket='luma-e2e', Key='etag.txt')
    assert head['ETag'].strip('"') == expected, head['ETag']
    s3.delete_object(Bucket='luma-e2e', Key='etag.txt')


@check
def a_multipart_etag_has_the_dash_and_part_count(s3):
    """S3's multipart ETag is md5(concat(part digests)) + "-" + count.

    The dash is load-bearing: it is how a client knows not to compare the ETag
    against the MD5 of the bytes it received.
    """
    import hashlib
    key = 'etag-multi.bin'
    created = s3.create_multipart_upload(Bucket='luma-e2e', Key=key)
    upload_id = created['UploadId']
    chunks = [b'A' * 100, b'B' * 100]
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        result = s3.upload_part(
            Bucket='luma-e2e', Key=key, UploadId=upload_id,
            PartNumber=number, Body=chunk)
        assert result['ETag'].strip('"') == hashlib.md5(chunk).hexdigest()
        parts.append({'PartNumber': number, 'ETag': result['ETag']})

    done = s3.complete_multipart_upload(
        Bucket='luma-e2e', Key=key, UploadId=upload_id,
        MultipartUpload={'Parts': parts})

    digests = b''.join(hashlib.md5(c).digest() for c in chunks)
    expected = '%s-%d' % (hashlib.md5(digests).hexdigest(), len(chunks))
    assert done['ETag'].strip('"') == expected, (done['ETag'], expected)

    # And it has to *stay* that. This assertion is the one that was missing, and
    # its absence hid a real defect for as long as the check existed: the
    # composite was computed at completion, returned, and then thrown away, so
    # every later HEAD and GET answered with the plain MD5 of the assembled
    # bytes instead. Reading the ETag only out of the completion response is
    # exactly the shape of a test that passes with the bug present — `aws s3
    # sync`, rclone and any ETag-validating cache ask the *server* later, see a
    # different value every time, and re-transfer the object.
    head = s3.head_object(Bucket='luma-e2e', Key=key)
    assert head['ETag'].strip('"') == expected, (
        'HEAD disagrees with CompleteMultipartUpload: %s != %s'
        % (head['ETag'], expected))
    got = s3.get_object(Bucket='luma-e2e', Key=key)
    assert got['ETag'].strip('"') == expected, (
        'GET disagrees with CompleteMultipartUpload: %s != %s'
        % (got['ETag'], expected))

    # Overwriting it with an ordinary PUT must retire the composite: the object
    # is no longer multipart, and answering with the old value would describe
    # bytes that are gone.
    s3.put_object(Bucket='luma-e2e', Key=key, Body=b'plain')
    after = s3.head_object(Bucket='luma-e2e', Key=key)
    assert after['ETag'].strip('"') == hashlib.md5(b'plain').hexdigest(), after['ETag']

    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def a_ranged_get_returns_only_the_range(s3):
    """`Range` in all four shapes, plus the unsatisfiable one.

    This is not an optional nicety. boto3 downloads anything above its
    threshold as concurrent ranged GETs and writes each reply at its own
    offset, so a server that *ignores* `Range` does not fail loudly — it returns
    the whole object for every range, and hands the caller a corrupt file with a
    200 on every request and nothing in any log.

    That is exactly what this server did until `tests/e2e/s3_scale.py` read a
    large object at an offset: `Range` appeared nowhere in the S3 router.
    """
    key = 'ranged.txt'
    s3.put_object(Bucket='luma-e2e', Key=key, Body=b'0123456789')

    cases = [
        ('bytes=2-5', b'2345', 'bytes 2-5/10'),      # closed
        ('bytes=7-', b'789', 'bytes 7-9/10'),        # open ended
        ('bytes=-3', b'789', 'bytes 7-9/10'),        # suffix
        ('bytes=0-100', b'0123456789', 'bytes 0-9/10'),  # end past the object
    ]
    for header, want, content_range in cases:
        got = s3.get_object(Bucket='luma-e2e', Key=key, Range=header)
        assert got['ResponseMetadata']['HTTPStatusCode'] == 206, (
            '%s answered %d, not 206' % (header, got['ResponseMetadata']['HTTPStatusCode']))
        body = got['Body'].read()
        assert body == want, '%s returned %r, wanted %r' % (header, body, want)
        assert got.get('ContentRange') == content_range, (
            '%s: Content-Range %r != %r' % (header, got.get('ContentRange'), content_range))

    # Past the end is 416, not a 200 with everything and not an empty 206.
    try:
        s3.get_object(Bucket='luma-e2e', Key=key, Range='bytes=50-60')
        raise AssertionError('a range past the end must be refused with 416')
    except s3.exceptions.ClientError as e:
        status = e.response['ResponseMetadata']['HTTPStatusCode']
        assert status == 416, 'expected 416, got %d' % status

    # A header this server does not act on is ignored, per RFC 9110 — the whole
    # object with a 200, never a 400.
    whole = s3.get_object(Bucket='luma-e2e', Key=key, Range='rows=1-2')
    assert whole['ResponseMetadata']['HTTPStatusCode'] == 200
    assert whole['Body'].read() == b'0123456789'

    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def an_object_larger_than_the_axum_default_is_accepted(s3):
    """Four megabytes — over axum's 2 MiB default, under any real limit.

    The S3 router carried no body limit of its own, so it inherited that default
    and closed the connection on anything bigger, with no status line and
    nothing logged. Every S3 test used hundred-byte bodies, so the API looked
    healthy while being unable to store a photograph.

    Deliberately small enough to stay on the fast suite: the point is the cliff
    at 2 MiB, not throughput.
    """
    key = 'four-megabytes.bin'
    body = b'M' * (4 * 1024 * 1024)
    s3.put_object(Bucket='luma-e2e', Key=key, Body=body)
    assert s3.head_object(Bucket='luma-e2e', Key=key)['ContentLength'] == len(body)
    assert s3.get_object(Bucket='luma-e2e', Key=key)['Body'].read() == body
    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def content_type_and_user_metadata_survive_a_round_trip(s3):
    """`Content-Type` and `x-amz-meta-*`, through a PUT and through multipart.

    Neither was stored. Every object came back as `application/octet-stream`
    with no metadata, so a JPEG uploaded here is served to a browser as a
    download, and any client that round-trips its own metadata loses it. mint's
    `test_put_object` fails on exactly that — with an 11 MiB body, which is why
    the multipart half is checked here too: S3 takes a multipart object's
    content type from the *initiating* request, and the first version of the fix
    read that record back after deleting the directory it lived in, so the
    composite ETag survived and the metadata silently did not.
    """
    key = 'typed.png'
    s3.put_object(Bucket='luma-e2e', Key=key, Body=b'notreallyapng',
                  ContentType='image/png', Metadata={'author': 'jairo', 'n': '1'})
    for how, got in (('HEAD', s3.head_object(Bucket='luma-e2e', Key=key)),
                     ('GET', s3.get_object(Bucket='luma-e2e', Key=key))):
        assert got['ContentType'] == 'image/png', '%s: %s' % (how, got['ContentType'])
        assert got['Metadata'] == {'author': 'jairo', 'n': '1'}, '%s: %s' % (how, got['Metadata'])

    # A plain PUT over it retires both: they described bytes that are gone.
    s3.put_object(Bucket='luma-e2e', Key=key, Body=b'plain')
    after = s3.head_object(Bucket='luma-e2e', Key=key)
    assert after['ContentType'] == 'application/octet-stream', after['ContentType']
    assert after['Metadata'] == {}, after['Metadata']
    s3.delete_object(Bucket='luma-e2e', Key=key)

    # And through multipart, where the metadata comes from CreateMultipartUpload.
    key = 'typed-multipart.png'
    upload = s3.create_multipart_upload(
        Bucket='luma-e2e', Key=key, ContentType='image/png',
        Metadata={'testing': 'value'})['UploadId']
    parts = []
    for number in (1, 2):
        chunk = bytes([number]) * (5 * 1024 * 1024)
        result = s3.upload_part(Bucket='luma-e2e', Key=key, UploadId=upload,
                                PartNumber=number, Body=chunk)
        parts.append({'PartNumber': number, 'ETag': result['ETag']})
    s3.complete_multipart_upload(Bucket='luma-e2e', Key=key, UploadId=upload,
                                 MultipartUpload={'Parts': parts})
    got = s3.head_object(Bucket='luma-e2e', Key=key)
    assert got['ContentType'] == 'image/png', got['ContentType']
    assert got['Metadata'] == {'testing': 'value'}, got['Metadata']
    assert '-' in got['ETag'], got['ETag']
    s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def copy_object_copies_the_bytes_and_not_nothing(s3):
    """`CopyObject`, which used to store an empty object and report success.

    It is a `PUT` carrying `x-amz-copy-source` and no body, so with no handler
    for it the request fell through to the ordinary write path: the destination
    became a 0-byte object and the reply was a 200 with no XML. `aws s3 cp`
    between two keys destroyed the destination and said it worked, which is the
    worst shape a bug can take.
    """
    s3.put_object(Bucket='luma-e2e', Key='copy-src', Body=b'fourteen bytes',
                  ContentType='text/plain', Metadata={'origin': 'src'})

    # COPY (the default) carries the source's content type and metadata across.
    result = s3.copy_object(Bucket='luma-e2e', Key='copy-dst',
                            CopySource={'Bucket': 'luma-e2e', 'Key': 'copy-src'})
    assert 'ETag' in result['CopyObjectResult'], result
    got = s3.get_object(Bucket='luma-e2e', Key='copy-dst')
    assert got['Body'].read() == b'fourteen bytes'
    assert got['ContentType'] == 'text/plain', got['ContentType']
    assert got['Metadata'] == {'origin': 'src'}, got['Metadata']

    # REPLACE takes them from the copy request instead.
    s3.copy_object(Bucket='luma-e2e', Key='copy-replaced',
                   CopySource={'Bucket': 'luma-e2e', 'Key': 'copy-src'},
                   MetadataDirective='REPLACE',
                   ContentType='application/json', Metadata={'origin': 'copy'})
    got = s3.head_object(Bucket='luma-e2e', Key='copy-replaced')
    assert got['ContentType'] == 'application/json', got['ContentType']
    assert got['Metadata'] == {'origin': 'copy'}, got['Metadata']

    # A missing source is a 404, not an empty object at the destination.
    try:
        s3.copy_object(Bucket='luma-e2e', Key='copy-nowhere',
                       CopySource={'Bucket': 'luma-e2e', 'Key': 'does-not-exist'})
        raise AssertionError('copying a missing key must fail')
    except s3.exceptions.ClientError as e:
        assert e.response['ResponseMetadata']['HTTPStatusCode'] == 404, e.response

    for key in ('copy-src', 'copy-dst', 'copy-replaced'):
        s3.delete_object(Bucket='luma-e2e', Key=key)


@check
def head_bucket_is_not_a_403(s3):
    """`HeadBucket`, which answered AccessDenied on a perfectly signed request.

    No HEAD route existed for `/{bucket}`, so axum served it with the GET
    handler — and that handler signed the canonical request as "GET" while the
    client had signed "HEAD". SigV4 covers the method, so the signatures could
    not match, and the reply was a 403 indistinguishable from a wrong secret.
    Every S3 client calls this to check a bucket exists; it is mint's first test.
    """
    assert s3.head_bucket(Bucket='luma-e2e')['ResponseMetadata']['HTTPStatusCode'] == 200
    try:
        s3.head_bucket(Bucket='luma-e2e-does-not-exist')
        raise AssertionError('a missing bucket must not answer 200')
    except s3.exceptions.ClientError as e:
        status = e.response['ResponseMetadata']['HTTPStatusCode']
        assert status in (403, 404), 'expected 404 (or 403 for another org), got %d' % status


@check
def the_last_modified_header_is_an_http_date(s3):
    """`Last-Modified` has to be parseable as an HTTP date.

    It was emitted in the ISO 8601 shape S3 uses *inside its XML*, which is not
    the format RFC 9110 fixes for the header. boto3 never parses it, so 14
    checks against boto3 said nothing; minio-py does, and refused the reply with
    "time data does not match HTTP header format".
    """
    from email.utils import parsedate_to_datetime
    s3.put_object(Bucket='luma-e2e', Key='dated', Body=b'x')
    raw = s3.head_object(Bucket='luma-e2e', Key='dated')
    header = raw['ResponseMetadata']['HTTPHeaders']['last-modified']
    # Raises if the format is wrong, which is the whole assertion.
    parsedate_to_datetime(header)
    assert header.endswith('GMT'), header
    s3.delete_object(Bucket='luma-e2e', Key='dated')


@check
def a_wrong_secret_is_refused(s3):
    """The signature has to actually be checked."""
    from botocore.exceptions import ClientError
    bad = client(s3.meta.endpoint_url, 'AKIDDOESNOTEXISTXXXX', 'wrong-secret')
    try:
        bad.list_buckets()
        raise AssertionError('an unknown credential must be refused')
    except ClientError as e:
        assert e.response['Error']['Code'] in ('AccessDenied', '403'), e.response['Error']


@check
def an_unsupported_subresource_is_refused_not_ignored(s3):
    """A client that sets an ACL and gets a 200 believes the object is private."""
    from botocore.exceptions import ClientError
    try:
        s3.put_bucket_acl(Bucket='luma-e2e', ACL='public-read')
        raise AssertionError('setting an ACL must not silently succeed')
    except ClientError as e:
        assert e.response['Error']['Code'] in ('NotImplemented', '501'), e.response['Error']


@check
def a_non_empty_bucket_cannot_be_deleted(s3):
    from botocore.exceptions import ClientError
    s3.put_object(Bucket='luma-e2e', Key='keeper.txt', Body=b'x')
    try:
        s3.delete_bucket(Bucket='luma-e2e')
        raise AssertionError('deleting a non-empty bucket must be refused')
    except ClientError as e:
        assert e.response['Error']['Code'] in ('BucketNotEmpty', '409'), e.response['Error']
    s3.delete_object(Bucket='luma-e2e', Key='keeper.txt')


@check
def a_presigned_url_works_and_then_expires(s3):
    """A presigned URL is a credential with a fuse.

    The fuse is the whole point: a server that skips the expiry check turns
    every one of these into a permanent credential handed to whoever the URL
    reached.
    """
    s3.put_object(Bucket='luma-e2e', Key='shared.txt', Body=b'shared bytes')
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': 'luma-e2e', 'Key': 'shared.txt'},
        ExpiresIn=300)

    with urllib.request.urlopen(url, timeout=10) as response:
        assert response.read() == b'shared bytes'

    # And one that has already run out.
    stale = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': 'luma-e2e', 'Key': 'shared.txt'},
        ExpiresIn=1)
    import time
    time.sleep(2)
    try:
        urllib.request.urlopen(stale, timeout=10)
        raise AssertionError('an expired presigned URL must not work')
    except urllib.error.HTTPError as e:
        assert e.code in (400, 403), e.code

    s3.delete_object(Bucket='luma-e2e', Key='shared.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admin', required=True, help='base URL of the /v1 API')
    parser.add_argument('--s3', required=True, help='base URL of the S3 port')
    parser.add_argument('--api-key', required=True, help='an admin api key')
    parser.add_argument('--org', default='e2e-org', help='organization to mint for')
    parser.add_argument('--only', help='substring of a check name')
    args = parser.parse_args()

    access_key, secret_key = mint_credentials(args.admin, args.api_key, args.org)
    global CREDENTIALS
    CREDENTIALS = (access_key, secret_key)
    s3 = client(args.s3, access_key, secret_key)

    width = max(len(c.__name__) for c in CHECKS) + 2
    failures = []
    for fn in CHECKS:
        if args.only and args.only not in fn.__name__:
            continue
        try:
            fn(s3)
            print('%-*s ok' % (width, fn.__name__))
        except Exception as e:
            print('%-*s FAIL  %s: %s' % (width, fn.__name__, type(e).__name__, e))
            failures.append(fn.__name__)

    if failures:
        print('\n%d failed: %s' % (len(failures), ', '.join(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
