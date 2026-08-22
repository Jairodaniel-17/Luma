"""A chunk-framed S3 upload, built by hand.

boto3 will not produce one on demand. `payload_signing_enabled` plus an
unseekable body was not enough — botocore chose an ordinary signed upload, and a
check written on that assumption passed with the server bug still present. That
false green is the reason this file exists: the request is constructed here, to
AWS's documented format, so the path is exercised or the test fails.

What it proves that the unit tests cannot: that a real HTTP request carrying
`STREAMING-AWS4-HMAC-SHA256-PAYLOAD` reaches the de-framing code at all. The
unit tests already cover the parser and the signature chain.

## Running it

    python tests/e2e/s3_chunked.py \\
        --admin http://127.0.0.1:18080 \\
        --s3 http://127.0.0.1:19000 \\
        --api-key <admin key>
"""
import argparse
import datetime
import hashlib
import hmac
import json
import sys
import urllib.error
import urllib.request

ALGORITHM = 'AWS4-HMAC-SHA256'
CHUNK_ALGORITHM = 'AWS4-HMAC-SHA256-PAYLOAD'
STREAMING_HASH = 'STREAMING-AWS4-HMAC-SHA256-PAYLOAD'
REGION = 'us-east-1'
SERVICE = 's3'


def sign(key, message):
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def signing_key(secret, date):
    key = sign(('AWS4' + secret).encode(), date)
    key = sign(key, REGION)
    key = sign(key, SERVICE)
    return sign(key, 'aws4_request')


def mint_credentials(admin_url, api_key, org):
    request = urllib.request.Request(
        admin_url.rstrip('/') + '/v1/admin/s3-credentials',
        method='POST',
        headers={'Authorization': 'Bearer ' + api_key,
                 'Content-Type': 'application/json'},
        data=json.dumps({'org_id': org}).encode(),
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.load(response)
    return body['access_key_id'], body['secret_access_key']


def build_chunked_put(s3_url, access_key, secret_key, bucket, key, chunks):
    """Return (url, headers, body) for a chunk-framed PUT.

    The two signatures are separate and both matter. The request signature is
    computed over the canonical request exactly as any other PUT, with the
    payload hash replaced by the streaming sentinel. Each chunk then signs the
    *previous* signature, starting from the request's — which is what makes the
    chain fix the chunks' order, not just their contents.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime('%Y%m%dT%H%M%SZ')
    date = now.strftime('%Y%m%d')
    scope = '%s/%s/%s/aws4_request' % (date, REGION, SERVICE)

    decoded_length = sum(len(c) for c in chunks)
    host = s3_url.split('://', 1)[1].rstrip('/')
    path = '/%s/%s' % (bucket, key)

    # Only these headers are signed, so only these may be sent with values that
    # differ from what was signed.
    headers = {
        'host': host,
        'x-amz-content-sha256': STREAMING_HASH,
        'x-amz-date': amz_date,
        'x-amz-decoded-content-length': str(decoded_length),
        'content-encoding': 'aws-chunked',
    }
    signed_headers = ';'.join(sorted(headers))
    canonical_headers = ''.join(
        '%s:%s\n' % (name, headers[name]) for name in sorted(headers))

    canonical_request = '\n'.join([
        'PUT', path, '', canonical_headers, signed_headers, STREAMING_HASH,
    ])
    to_sign = '\n'.join([
        ALGORITHM, amz_date, scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])
    key_bytes = signing_key(secret_key, date)
    seed = hmac.new(key_bytes, to_sign.encode(), hashlib.sha256).hexdigest()

    empty = hashlib.sha256(b'').hexdigest()

    def chunk_signature(data, previous):
        payload = '\n'.join([
            CHUNK_ALGORITHM, amz_date, scope, previous, empty,
            hashlib.sha256(data).hexdigest(),
        ])
        return hmac.new(key_bytes, payload.encode(), hashlib.sha256).hexdigest()

    body = b''
    previous = seed
    for data in chunks:
        previous = chunk_signature(data, previous)
        body += b'%x;chunk-signature=%s\r\n' % (len(data), previous.encode())
        body += data + b'\r\n'
    previous = chunk_signature(b'', previous)
    body += b'0;chunk-signature=%s\r\n' % previous.encode()

    headers['Authorization'] = (
        '%s Credential=%s/%s, SignedHeaders=%s, Signature=%s'
        % (ALGORITHM, access_key, scope, signed_headers, seed))
    return s3_url.rstrip('/') + path, headers, body


def put(url, headers, body, method='PUT'):
    request = urllib.request.Request(url, method=method, data=body)
    for name, value in headers.items():
        if name != 'host':
            request.add_header(name, value)
    return urllib.request.urlopen(request, timeout=15)


def signed_get(s3_url, access_key, secret_key, bucket, key):
    """An ordinary signed GET, to read back what was stored."""
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime('%Y%m%dT%H%M%SZ')
    date = now.strftime('%Y%m%d')
    scope = '%s/%s/%s/aws4_request' % (date, REGION, SERVICE)
    host = s3_url.split('://', 1)[1].rstrip('/')
    path = '/%s/%s' % (bucket, key)
    empty = hashlib.sha256(b'').hexdigest()

    headers = {'host': host, 'x-amz-content-sha256': empty,
               'x-amz-date': amz_date}
    signed_headers = ';'.join(sorted(headers))
    canonical_headers = ''.join(
        '%s:%s\n' % (n, headers[n]) for n in sorted(headers))
    canonical_request = '\n'.join(
        ['GET', path, '', canonical_headers, signed_headers, empty])
    to_sign = '\n'.join([ALGORITHM, amz_date, scope,
                         hashlib.sha256(canonical_request.encode()).hexdigest()])
    signature = hmac.new(signing_key(secret_key, date),
                         to_sign.encode(), hashlib.sha256).hexdigest()
    headers['Authorization'] = (
        '%s Credential=%s/%s, SignedHeaders=%s, Signature=%s'
        % (ALGORITHM, access_key, scope, signed_headers, signature))
    request = urllib.request.Request(s3_url.rstrip('/') + path, method='GET')
    for name, value in headers.items():
        if name != 'host':
            request.add_header(name, value)
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admin', required=True)
    parser.add_argument('--s3', required=True)
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--org', default='chunked-org')
    args = parser.parse_args()

    access, secret = mint_credentials(args.admin, args.api_key, args.org)

    # The bucket, through the ordinary path.
    url, headers, body = build_chunked_put(
        args.s3, access, secret, 'chunked-e2e', 'x', [b'x'])
    del url, headers, body
    import boto3
    from botocore.config import Config
    s3 = boto3.client('s3', endpoint_url=args.s3,
                      aws_access_key_id=access, aws_secret_access_key=secret,
                      region_name=REGION,
                      config=Config(signature_version='s3v4',
                                    s3={'addressing_style': 'path'}))
    s3.create_bucket(Bucket='chunked-e2e')

    failures = []
    chunks = [b'A' * 300, b'B' * 200, b'C' * 100]
    expected = b''.join(chunks)

    # 1. A well-formed chunked upload stores the payload, not the framing.
    url, headers, body = build_chunked_put(
        args.s3, access, secret, 'chunked-e2e', 'framed.bin', chunks)
    assert b'chunk-signature=' in body, 'the fixture must be framed'
    try:
        with put(url, headers, body) as response:
            assert response.status in (200, 204), response.status
        stored = signed_get(args.s3, access, secret, 'chunked-e2e', 'framed.bin')
        if stored == expected:
            print('%-58s ok' % 'a_chunked_upload_stores_the_payload')
        else:
            print('%-58s FAIL  stored %d bytes, expected %d'
                  % ('a_chunked_upload_stores_the_payload',
                     len(stored), len(expected)))
            failures.append('a_chunked_upload_stores_the_payload')
    except urllib.error.HTTPError as e:
        print('%-58s FAIL  HTTP %s: %s'
              % ('a_chunked_upload_stores_the_payload', e.code,
                 e.read()[:200]))
        failures.append('a_chunked_upload_stores_the_payload')

    # 2. A tampered chunk is refused. The framing stays valid; one payload byte
    #    changes, so only the chunk signature can catch it.
    url, headers, body = build_chunked_put(
        args.s3, access, secret, 'chunked-e2e', 'tampered.bin', chunks)
    at = body.index(b'A' * 300)
    body = body[:at] + b'Z' + body[at + 1:]
    name = 'a_tampered_chunk_is_refused'
    try:
        put(url, headers, body)
        print('%-58s FAIL  the tampered body was accepted' % name)
        failures.append(name)
    except urllib.error.HTTPError as e:
        if e.code in (400, 403):
            print('%-58s ok' % name)
        else:
            print('%-58s FAIL  HTTP %s' % (name, e.code))
            failures.append(name)

    # 3. A body cut before its terminating chunk is refused rather than stored
    #    short — a truncated upload must not answer 200.
    url, headers, body = build_chunked_put(
        args.s3, access, secret, 'chunked-e2e', 'truncated.bin', chunks)
    body = body[:len(body) // 2]
    name = 'a_truncated_body_is_refused'
    try:
        put(url, headers, body)
        print('%-58s FAIL  the truncated body was accepted' % name)
        failures.append(name)
    except urllib.error.HTTPError as e:
        if e.code in (400, 403):
            print('%-58s ok' % name)
        else:
            print('%-58s FAIL  HTTP %s' % (name, e.code))
            failures.append(name)

    for key in ('framed.bin', 'tampered.bin', 'truncated.bin', 'x'):
        try:
            s3.delete_object(Bucket='chunked-e2e', Key=key)
        except Exception:
            pass

    if failures:
        print('\n%d failed: %s' % (len(failures), ', '.join(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
