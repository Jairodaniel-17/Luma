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
    s3.delete_object(Bucket='luma-e2e', Key=key)


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
