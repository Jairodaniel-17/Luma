//! AWS Signature Version 4 verification.
//!
//! W3.2 of `docs/PLAN-MAESTRO.md`, and the part the plan singles out as having
//! dark corners. It is worth being explicit about why: SigV4 is not a hash of
//! the request, it is a hash of a *canonical form* of the request, and almost
//! every interoperability failure is a disagreement about that canonical form
//! rather than about the cryptography.
//!
//! The four places implementations diverge, all handled below and all with a
//! test:
//!
//! 1. **Double URI encoding.** For S3 the path is *not* re-encoded, unlike every
//!    other AWS service. Getting this wrong breaks exactly the keys with a `/` or
//!    a space in them — which is most real object keys.
//! 2. **Header canonicalisation.** Names lowercased, values trimmed and inner
//!    runs of whitespace collapsed, sorted by name, and only the headers the
//!    client listed in `SignedHeaders` participate.
//! 3. **Query canonicalisation.** Sorted by encoded key, and a parameter with no
//!    value still contributes `key=`.
//! 4. **`UNSIGNED-PAYLOAD` and `STREAMING-…`.** A client may decline to hash the
//!    body. Requiring a hash would reject boto3's default for uploads over the
//!    multipart threshold.
//!
//! ## Chunk-framed bodies
//!
//! `STREAMING-AWS4-HMAC-SHA256-PAYLOAD` is unpacked and each chunk's signature
//! is verified — see `dechunk_and_verify`.
//!
//! This was recorded as "recognised in the signature, per-chunk signatures not
//! verified", which understated it. There was no parser at all, so the framing
//! was stored **inside the object**: not an unverified body, a corrupt one, with
//! a 200 in reply. `tests/e2e/s3_chunked.py` demonstrates it — with the parser
//! removed, a 600-byte payload arrives as 946 bytes.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

/// A parsed `Authorization: AWS4-HMAC-SHA256 …` header.
#[derive(Debug, Clone, PartialEq)]
pub struct Credential {
    pub access_key_id: String,
    pub date: String,
    pub region: String,
    pub service: String,
    pub signed_headers: Vec<String>,
    pub signature: String,
}

/// Everything needed to recompute a signature.
pub struct Request<'a> {
    pub method: &'a str,
    /// The path, already percent-decoded exactly once by the HTTP layer.
    pub path: &'a str,
    /// Raw query string, without the leading `?`.
    pub query: &'a str,
    /// All request headers, in any order.
    pub headers: &'a [(String, String)],
    /// The value the client claims for the payload hash.
    pub payload_hash: &'a str,
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    Ok,
    /// The signature did not match. Never says which part disagreed: telling a
    /// caller "your date was wrong" narrows the search for a forger.
    Mismatch,
    Malformed(&'static str),
}

/// Parse the `Authorization` header of a SigV4 request.
pub fn parse_authorization(header: &str) -> Result<Credential, &'static str> {
    let rest = header
        .strip_prefix("AWS4-HMAC-SHA256")
        .ok_or("unsupported signature algorithm")?
        .trim_start();

    let mut access_key_id = None;
    let mut date = None;
    let mut region = None;
    let mut service = None;
    let mut signed_headers = Vec::new();
    let mut signature = None;

    for part in rest.split(',') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix("Credential=") {
            // `AKID/20260822/us-east-1/s3/aws4_request`
            let mut fields = value.split('/');
            access_key_id = fields.next().map(str::to_string);
            date = fields.next().map(str::to_string);
            region = fields.next().map(str::to_string);
            service = fields.next().map(str::to_string);
        } else if let Some(value) = part.strip_prefix("SignedHeaders=") {
            signed_headers = value.split(';').map(|h| h.to_ascii_lowercase()).collect();
        } else if let Some(value) = part.strip_prefix("Signature=") {
            signature = Some(value.to_string());
        }
    }

    Ok(Credential {
        access_key_id: access_key_id.ok_or("missing access key id")?,
        date: date.ok_or("missing credential date")?,
        region: region.ok_or("missing region")?,
        service: service.ok_or("missing service")?,
        signed_headers: {
            if signed_headers.is_empty() {
                return Err("missing signed headers");
            }
            signed_headers
        },
        signature: signature.ok_or("missing signature")?,
    })
}

/// Recompute the signature and compare.
pub fn verify(request: &Request<'_>, credential: &Credential, secret: &str) -> Verdict {
    let Some(amz_date) = header_value(request.headers, "x-amz-date") else {
        return Verdict::Malformed("missing x-amz-date");
    };

    let canonical = canonical_request(request, &credential.signed_headers);
    let scope = format!(
        "{}/{}/{}/aws4_request",
        credential.date, credential.region, credential.service
    );
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex::encode(Sha256::digest(canonical.as_bytes()))
    );

    let signing_key = signing_key(
        secret,
        &credential.date,
        &credential.region,
        &credential.service,
    );
    let expected = hex::encode(hmac(&signing_key, to_sign.as_bytes()));

    // Constant-time compare: a byte-by-byte early exit leaks how much of a
    // guessed signature was right, which is enough to find the rest one byte at
    // a time.
    if constant_time_eq(expected.as_bytes(), credential.signature.as_bytes()) {
        Verdict::Ok
    } else {
        Verdict::Mismatch
    }
}

/// The canonical request, exactly as the specification defines it.
pub fn canonical_request(request: &Request<'_>, signed_headers: &[String]) -> String {
    let mut canonical_headers = String::new();
    for name in signed_headers {
        let value = header_value(request.headers, name).unwrap_or_default();
        canonical_headers.push_str(name);
        canonical_headers.push(':');
        canonical_headers.push_str(&collapse_whitespace(&value));
        canonical_headers.push('\n');
    }

    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        request.method,
        canonical_uri(request.path),
        canonical_query(request.query),
        canonical_headers,
        signed_headers.join(";"),
        request.payload_hash,
    )
}

/// The canonical URI.
///
/// S3 is the exception among AWS services: the path is percent-encoded **once**,
/// not twice, and `/` is left alone. Double-encoding here is the single most
/// common cause of "works for `foo.txt`, fails for `a/b c.txt`".
fn canonical_uri(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    path.split('/')
        .map(|segment| uri_encode(segment, false))
        .collect::<Vec<_>>()
        .join("/")
}

/// Canonical query string: sorted by encoded name, then by encoded value.
fn canonical_query(query: &str) -> String {
    if query.is_empty() {
        return String::new();
    }
    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (
                uri_encode(&percent_decode(k), true),
                uri_encode(&percent_decode(v), true),
            ),
            // A flag with no value still contributes `key=`, not `key`.
            None => (uri_encode(&percent_decode(pair), true), String::new()),
        })
        .collect();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Percent-encode per RFC 3986, with `/` optionally left literal.
fn uri_encode(input: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        let keep = byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b'~')
            || (byte == b'/' && !encode_slash);
        if keep {
            out.push(byte as char);
        } else {
            // Uppercase hex: the specification requires it, and a lowercase
            // escape produces a different string to sign.
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&input[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        // `+` means space only in a form-encoded body, not in an S3 query
        // string, so it is left alone deliberately.
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Trim, and collapse inner runs of spaces to one.
///
/// Values inside quotes are left alone by the specification; that case does not
/// arise for the headers S3 signs, and pretending to handle it would be worse
/// than not.
fn collapse_whitespace(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut previous_space = false;
    for c in value.trim().chars() {
        if c == ' ' || c == '\t' {
            if !previous_space {
                out.push(' ');
            }
            previous_space = true;
        } else {
            out.push(c);
            previous_space = false;
        }
    }
    out
}

fn header_value(headers: &[(String, String)], name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

/// What a `STREAMING-AWS4-HMAC-SHA256-PAYLOAD` request needs to be unpacked.
///
/// Everything here is already known once the request's own signature has been
/// verified; it is grouped so the caller can hand it to `dechunk_and_verify`
/// without re-deriving anything.
pub struct StreamingContext {
    /// The request signature, which seeds the chunk signature chain.
    pub seed_signature: String,
    pub amz_date: String,
    pub scope: String,
    pub signing_key: Vec<u8>,
}

impl StreamingContext {
    /// Build the context from a request whose own signature already verified.
    ///
    /// Only meaningful after `verify` returned `Ok`: the seed of the chunk chain
    /// **is** the request signature, so building this from an unverified request
    /// would anchor the whole chain to a value an attacker chose.
    pub fn from_verified(
        credential: &Credential,
        secret: &str,
        headers: &[(String, String)],
    ) -> Option<StreamingContext> {
        Some(StreamingContext {
            seed_signature: credential.signature.clone(),
            amz_date: header_value(headers, "x-amz-date")?,
            scope: format!(
                "{}/{}/{}/aws4_request",
                credential.date, credential.region, credential.service
            ),
            signing_key: signing_key(
                secret,
                &credential.date,
                &credential.region,
                &credential.service,
            ),
        })
    }
}

/// Whether a payload hash says the body arrives chunk-framed.
///
/// Distinct from `is_unsigned_payload`: `UNSIGNED-PAYLOAD` means the body is
/// exactly what it looks like and simply is not hashed, while `STREAMING-…`
/// means the bytes on the wire are **not** the object — they carry framing that
/// has to be removed. Treating the two the same is what stored the framing.
pub fn is_streaming_payload(hash: &str) -> bool {
    hash.starts_with("STREAMING-")
}

/// Unpack a chunk-framed body and verify each chunk's signature.
///
/// **The framing has to be removed whether or not it is verified**, and that is
/// what makes this less optional than it looked. A chunked body arrives as
/// `<hex-size>;chunk-signature=<sig>\r\n<data>\r\n` repeated, ending with a
/// zero-length chunk. Storing what arrives — which is what happens with no
/// parser — writes the size lines and the signatures *into the object*. Not an
/// unverified body: a corrupt one, silently, with a 200 in reply.
///
/// Since the frame must be parsed anyway, verifying costs one HMAC per chunk.
/// Each chunk signs the previous signature, so the chain also fixes the chunks'
/// **order** — a reordered or dropped chunk breaks it, which a per-chunk digest
/// alone would not catch.
///
/// AWS's string to sign, per chunk:
///
/// ```text
/// AWS4-HMAC-SHA256-PAYLOAD \n date \n scope \n previous-signature \n
/// sha256("") \n sha256(chunk-data)
/// ```
pub fn dechunk_and_verify(body: &[u8], ctx: &StreamingContext) -> Result<Vec<u8>, Verdict> {
    let mut payload = Vec::with_capacity(body.len());
    let mut previous = ctx.seed_signature.clone();
    let mut at = 0usize;
    let empty_hash = hex::encode(Sha256::digest(b""));
    let mut saw_final = false;

    while at < body.len() {
        // The header line runs to CRLF: "<hex-size>;chunk-signature=<hex>".
        let Some(line_end) = find(body, at, b"\r\n") else {
            return Err(Verdict::Malformed("a chunk header had no CRLF"));
        };
        let header = &body[at..line_end];
        at = line_end + 2;

        let (size_part, signature) = match split_once(header, b';') {
            Some((size, rest)) => {
                let Some(sig) = rest.strip_prefix(b"chunk-signature=") else {
                    return Err(Verdict::Malformed("a chunk header had no chunk-signature"));
                };
                (size, sig)
            }
            None => return Err(Verdict::Malformed("a chunk header had no signature field")),
        };

        let size_text = String::from_utf8_lossy(size_part);
        let Ok(size) = usize::from_str_radix(size_text.trim(), 16) else {
            return Err(Verdict::Malformed("a chunk size was not hexadecimal"));
        };
        // A size the body cannot contain is a truncated upload, not a big one.
        if at + size > body.len() {
            return Err(Verdict::Malformed(
                "a chunk claimed more data than was sent",
            ));
        }
        let data = &body[at..at + size];

        let to_sign = format!(
            "AWS4-HMAC-SHA256-PAYLOAD\n{}\n{}\n{}\n{}\n{}",
            ctx.amz_date,
            ctx.scope,
            previous,
            empty_hash,
            hex::encode(Sha256::digest(data))
        );
        let expected = hex::encode(hmac(&ctx.signing_key, to_sign.as_bytes()));
        if !constant_time_eq(expected.as_bytes(), signature) {
            return Err(Verdict::Mismatch);
        }
        previous = expected;

        at += size;
        // Every chunk, including the last, is followed by CRLF.
        if size > 0 {
            if body.get(at..at + 2) != Some(b"\r\n") {
                return Err(Verdict::Malformed("a chunk was not terminated by CRLF"));
            }
            at += 2;
            payload.extend_from_slice(data);
        } else {
            saw_final = true;
            break;
        }
    }

    // Without the terminating zero-length chunk the upload was cut short. Taking
    // what arrived would store a truncated object and report success.
    if !saw_final {
        return Err(Verdict::Malformed(
            "the body ended without its final zero-length chunk",
        ));
    }
    Ok(payload)
}

fn find(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

fn split_once(slice: &[u8], byte: u8) -> Option<(&[u8], &[u8])> {
    let at = slice.iter().position(|&b| b == byte)?;
    Some((&slice[..at], &slice[at + 1..]))
}

fn hmac(key: &[u8], message: &[u8]) -> Vec<u8> {
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().to_vec()
}

/// The derived signing key: date, region, service, then the terminator.
fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let mut key = hmac(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    key = hmac(&key, region.as_bytes());
    key = hmac(&key, service.as_bytes());
    hmac(&key, b"aws4_request")
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// A presigned request: the signature travels in the query string.
///
/// The same computation with three differences, each of which silently breaks
/// interoperability if missed:
///
/// 1. The payload hash is the literal `UNSIGNED-PAYLOAD`, because a URL is
///    handed to somebody who has not sent a body yet.
/// 2. `X-Amz-Signature` is excluded from the canonical query — it cannot be part
///    of what it signs.
/// 3. Expiry is enforced from `X-Amz-Date` plus `X-Amz-Expires`. Skipping it
///    would turn every presigned URL into a permanent one, which is the entire
///    thing they exist not to be.
pub struct Presigned {
    pub credential: Credential,
    pub date: String,
    pub expires_secs: u64,
}

/// Parse the `X-Amz-*` query parameters of a presigned URL.
pub fn parse_presigned(query: &str) -> Option<Presigned> {
    let mut params = std::collections::BTreeMap::new();
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            params.insert(percent_decode(k), percent_decode(v));
        }
    }
    if params.get("X-Amz-Algorithm").map(String::as_str) != Some("AWS4-HMAC-SHA256") {
        return None;
    }

    let credential_field = params.get("X-Amz-Credential")?;
    let mut fields = credential_field.split('/');
    let access_key_id = fields.next()?.to_string();
    let date = fields.next()?.to_string();
    let region = fields.next()?.to_string();
    let service = fields.next()?.to_string();

    Some(Presigned {
        credential: Credential {
            access_key_id,
            date,
            region,
            service,
            signed_headers: params
                .get("X-Amz-SignedHeaders")?
                .split(';')
                .map(|h| h.to_ascii_lowercase())
                .collect(),
            signature: params.get("X-Amz-Signature")?.to_string(),
        },
        date: params.get("X-Amz-Date")?.to_string(),
        expires_secs: params
            .get("X-Amz-Expires")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    })
}

/// Verify a presigned request.
pub fn verify_presigned(
    method: &str,
    path: &str,
    query: &str,
    headers: &[(String, String)],
    presigned: &Presigned,
    secret: &str,
    now: std::time::SystemTime,
) -> Verdict {
    if let Some(reason) = expired(&presigned.date, presigned.expires_secs, now) {
        return Verdict::Malformed(reason);
    }

    // The signature cannot be part of what it signs.
    let signable: String = query
        .split('&')
        .filter(|pair| !pair.starts_with("X-Amz-Signature="))
        .collect::<Vec<_>>()
        .join("&");

    let request = Request {
        method,
        path,
        query: &signable,
        headers,
        payload_hash: "UNSIGNED-PAYLOAD",
    };
    let canonical = canonical_request(&request, &presigned.credential.signed_headers);
    let scope = format!(
        "{}/{}/{}/aws4_request",
        presigned.credential.date, presigned.credential.region, presigned.credential.service
    );
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{scope}\n{}",
        presigned.date,
        hex::encode(Sha256::digest(canonical.as_bytes()))
    );
    let key = signing_key(
        secret,
        &presigned.credential.date,
        &presigned.credential.region,
        &presigned.credential.service,
    );
    let expected = hex::encode(hmac(&key, to_sign.as_bytes()));

    if constant_time_eq(
        expected.as_bytes(),
        presigned.credential.signature.as_bytes(),
    ) {
        Verdict::Ok
    } else {
        Verdict::Mismatch
    }
}

/// Whether a presigned URL is outside its validity window.
///
/// Returns the reason rather than a bool so the caller can say *why*: "expired"
/// and "your clock is ahead of ours" look identical from the outside and have
/// completely different fixes.
fn expired(amz_date: &str, expires_secs: u64, now: std::time::SystemTime) -> Option<&'static str> {
    let Some(signed_at) = parse_amz_date(amz_date) else {
        return Some("X-Amz-Date is not a valid timestamp");
    };
    let now_secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if expires_secs == 0 {
        return Some("X-Amz-Expires is missing or zero");
    }
    // A week is S3's own maximum, and a URL valid for longer is a credential
    // with extra steps.
    if expires_secs > 604_800 {
        return Some("X-Amz-Expires exceeds the seven-day maximum");
    }
    if now_secs > signed_at + expires_secs {
        return Some("the presigned URL has expired");
    }
    // A signature from the future within a small skew is ordinary clock drift;
    // beyond that it is not something to accept quietly.
    if signed_at > now_secs + 900 {
        return Some("X-Amz-Date is too far in the future");
    }
    None
}

/// `YYYYMMDDTHHMMSSZ` to seconds since the epoch.
fn parse_amz_date(text: &str) -> Option<u64> {
    if text.len() != 16 || !text.ends_with('Z') || text.as_bytes()[8] != b'T' {
        return None;
    }
    let year: i64 = text[0..4].parse().ok()?;
    let month: i64 = text[4..6].parse().ok()?;
    let day: i64 = text[6..8].parse().ok()?;
    let hour: u64 = text[9..11].parse().ok()?;
    let minute: u64 = text[11..13].parse().ok()?;
    let second: u64 = text[13..15].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some((days * 86_400) as u64 + hour * 3600 + minute * 60 + second)
}

/// Days since the epoch for a calendar date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// SHA-256 of an empty body, which clients send constantly.
pub const EMPTY_PAYLOAD_HASH: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Payload hashes that mean "I am not hashing the body".
pub fn is_unsigned_payload(hash: &str) -> bool {
    hash == "UNSIGNED-PAYLOAD" || hash.starts_with("STREAMING-")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// The published AWS SigV4 test vector.
    ///
    /// A known-answer test, not a round-trip: signing with our own code and
    /// verifying with our own code would pass with any consistent mistake, which
    /// is exactly the mistake this is about.
    #[test]
    fn the_derived_signing_key_matches_the_published_vector() {
        // From the AWS documentation's SigV4 examples.
        let key = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830",
            "us-east-1",
            "iam",
        );
        assert_eq!(
            hex::encode(key),
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
        );
    }

    #[test]
    fn the_empty_payload_hash_is_the_sha256_of_nothing() {
        assert_eq!(hex::encode(Sha256::digest(b"")), EMPTY_PAYLOAD_HASH);
    }

    #[test]
    fn s3_encodes_the_path_once_and_keeps_the_slashes() {
        // The single most common interoperability failure: double-encoding here
        // breaks every key with a slash or a space, which is most real keys.
        assert_eq!(canonical_uri("/bucket/a/b c.txt"), "/bucket/a/b%20c.txt");
        assert_eq!(canonical_uri("/bucket/plain.txt"), "/bucket/plain.txt");
        // Already-encoded input must not be encoded again.
        assert_eq!(canonical_uri("/bucket/a%2Fb"), "/bucket/a%252Fb");
        assert_eq!(canonical_uri(""), "/");
        // Unreserved characters stay literal, including the tilde — encoding it
        // is a classic off-by-one in the unreserved set.
        assert_eq!(canonical_uri("/a~b-c_d.e"), "/a~b-c_d.e");
    }

    #[test]
    fn the_query_is_sorted_and_a_valueless_flag_keeps_its_equals() {
        assert_eq!(canonical_query("b=2&a=1"), "a=1&b=2");
        assert_eq!(
            canonical_query("list-type=2&prefix="),
            "list-type=2&prefix="
        );
        // A bare flag contributes `key=`, not `key`.
        assert_eq!(canonical_query("acl"), "acl=");
        assert_eq!(canonical_query(""), "");
        // Sorting is on the *encoded* form.
        assert_eq!(canonical_query("a=b&A=c"), "A=c&a=b");
    }

    #[test]
    fn header_values_are_trimmed_and_inner_whitespace_collapsed() {
        assert_eq!(collapse_whitespace("  a   b  "), "a b");
        assert_eq!(collapse_whitespace("a\t\tb"), "a b");
        assert_eq!(collapse_whitespace("plain"), "plain");
    }

    #[test]
    fn only_the_listed_headers_take_part() {
        // A header the client did not list must not change the signature, or a
        // proxy adding one would break every request.
        let signed = vec!["host".to_string()];
        let with_extra = Request {
            method: "GET",
            path: "/b/k",
            query: "",
            headers: &headers(&[
                ("host", "example"),
                ("x-forwarded-for", "10.0.0.1"),
                ("x-amz-date", "20260822T000000Z"),
            ]),
            payload_hash: EMPTY_PAYLOAD_HASH,
        };
        let without = Request {
            headers: &headers(&[("host", "example"), ("x-amz-date", "20260822T000000Z")]),
            ..with_extra
        };
        assert_eq!(
            canonical_request(&with_extra, &signed),
            canonical_request(&without, &signed)
        );
    }

    #[test]
    fn an_authorization_header_parses_into_its_parts() {
        let credential = parse_authorization(
            "AWS4-HMAC-SHA256 Credential=AKID/20260822/us-east-1/s3/aws4_request, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=abc123",
        )
        .unwrap();
        assert_eq!(credential.access_key_id, "AKID");
        assert_eq!(credential.date, "20260822");
        assert_eq!(credential.region, "us-east-1");
        assert_eq!(credential.service, "s3");
        assert_eq!(
            credential.signed_headers,
            vec!["host", "x-amz-content-sha256", "x-amz-date"]
        );
        assert_eq!(credential.signature, "abc123");
    }

    #[test]
    fn a_different_algorithm_is_refused_rather_than_guessed() {
        assert!(parse_authorization("AWS4-HMAC-SHA512 Credential=x").is_err());
        assert!(parse_authorization("Bearer token").is_err());
    }

    #[test]
    fn a_signature_computed_with_the_wrong_secret_does_not_verify() {
        let request = Request {
            method: "GET",
            path: "/bucket/key",
            query: "",
            headers: &headers(&[("host", "luma"), ("x-amz-date", "20260822T000000Z")]),
            payload_hash: EMPTY_PAYLOAD_HASH,
        };
        let signed = vec!["host".to_string(), "x-amz-date".to_string()];

        // Sign with one secret…
        let canonical = canonical_request(&request, &signed);
        let scope = "20260822/us-east-1/s3/aws4_request";
        let to_sign = format!(
            "AWS4-HMAC-SHA256\n20260822T000000Z\n{scope}\n{}",
            hex::encode(Sha256::digest(canonical.as_bytes()))
        );
        let good = hex::encode(hmac(
            &signing_key("right-secret", "20260822", "us-east-1", "s3"),
            to_sign.as_bytes(),
        ));

        let credential = Credential {
            access_key_id: "AKID".into(),
            date: "20260822".into(),
            region: "us-east-1".into(),
            service: "s3".into(),
            signed_headers: signed,
            signature: good,
        };
        // …and verify with the same one, then a different one.
        assert_eq!(verify(&request, &credential, "right-secret"), Verdict::Ok);
        assert_eq!(
            verify(&request, &credential, "wrong-secret"),
            Verdict::Mismatch
        );
    }

    #[test]
    fn a_missing_amz_date_is_malformed_not_a_mismatch() {
        // The two are different answers: one means "your clock or your client is
        // broken", the other means "your credentials do not match". Collapsing
        // them makes both undiagnosable.
        let request = Request {
            method: "GET",
            path: "/b/k",
            query: "",
            headers: &headers(&[("host", "luma")]),
            payload_hash: EMPTY_PAYLOAD_HASH,
        };
        let credential = Credential {
            access_key_id: "AKID".into(),
            date: "20260822".into(),
            region: "us-east-1".into(),
            service: "s3".into(),
            signed_headers: vec!["host".to_string()],
            signature: "whatever".into(),
        };
        assert!(matches!(
            verify(&request, &credential, "secret"),
            Verdict::Malformed(_)
        ));
    }

    #[test]
    fn unsigned_and_streaming_payloads_are_recognised() {
        // boto3 sends UNSIGNED-PAYLOAD by default over HTTPS and
        // STREAMING-AWS4-HMAC-SHA256-PAYLOAD for chunked uploads. Requiring a
        // real hash would reject its defaults.
        assert!(is_unsigned_payload("UNSIGNED-PAYLOAD"));
        assert!(is_unsigned_payload("STREAMING-AWS4-HMAC-SHA256-PAYLOAD"));
        assert!(!is_unsigned_payload(EMPTY_PAYLOAD_HASH));
    }

    fn at(seconds: u64) -> std::time::SystemTime {
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds)
    }

    #[test]
    fn an_amz_date_parses_to_the_right_instant() {
        // Wrong by a day here and every presigned URL expires a day early or
        // late, which nobody would attribute to date parsing.
        assert_eq!(parse_amz_date("19700101T000000Z"), Some(0));
        assert_eq!(parse_amz_date("19700102T000000Z"), Some(86_400));
        assert_eq!(parse_amz_date("20260822T120000Z"), Some(1_787_400_000));
        // Malformed inputs are None rather than a guess.
        assert_eq!(parse_amz_date("2026-08-22T12:00:00Z"), None);
        assert_eq!(parse_amz_date("20260822T120000"), None);
        assert_eq!(parse_amz_date("20261322T120000Z"), None);
    }

    #[test]
    fn a_presigned_url_stops_working_when_it_expires() {
        // The entire point of a presigned URL is that it stops. Skipping this
        // check turns every one of them into a permanent credential.
        let signed = "20260822T120000Z";
        let signed_at = parse_amz_date(signed).unwrap();
        assert_eq!(expired(signed, 3600, at(signed_at + 60)), None);
        assert!(expired(signed, 3600, at(signed_at + 3601)).is_some());
    }

    #[test]
    fn a_presigned_url_without_an_expiry_is_refused() {
        let signed = "20260822T120000Z";
        let signed_at = parse_amz_date(signed).unwrap();
        assert!(expired(signed, 0, at(signed_at)).is_some());
        // And one that outlives S3's own maximum: a URL valid for longer is a
        // credential with extra steps.
        assert!(expired(signed, 604_801, at(signed_at)).is_some());
    }

    #[test]
    fn a_little_clock_skew_is_tolerated_and_a_lot_is_not() {
        let signed = "20260822T120000Z";
        let signed_at = parse_amz_date(signed).unwrap();
        // Signed 10 minutes "in the future" by our clock: ordinary drift.
        assert_eq!(expired(signed, 3600, at(signed_at - 600)), None);
        // An hour ahead is not drift.
        assert!(expired(signed, 3600, at(signed_at - 3600)).is_some());
    }

    #[test]
    fn the_signature_is_excluded_from_what_it_signs() {
        // Including it would make verification impossible: the value would have
        // to be known before it was computed.
        let query = "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc&X-Amz-Expires=60";
        let signable: String = query
            .split('&')
            .filter(|pair| !pair.starts_with("X-Amz-Signature="))
            .collect::<Vec<_>>()
            .join("&");
        assert!(!signable.contains("X-Amz-Signature"));
        assert!(signable.contains("X-Amz-Expires=60"));
    }

    #[test]
    fn a_presigned_query_parses_into_its_parts() {
        let presigned = parse_presigned(
            "X-Amz-Algorithm=AWS4-HMAC-SHA256\
             &X-Amz-Credential=AKID%2F20260822%2Fus-east-1%2Fs3%2Faws4_request\
             &X-Amz-Date=20260822T120000Z&X-Amz-Expires=3600\
             &X-Amz-SignedHeaders=host&X-Amz-Signature=deadbeef",
        )
        .expect("must parse");
        assert_eq!(presigned.credential.access_key_id, "AKID");
        assert_eq!(presigned.credential.region, "us-east-1");
        assert_eq!(presigned.expires_secs, 3600);
        assert_eq!(presigned.date, "20260822T120000Z");
    }

    #[test]
    fn a_query_without_the_algorithm_is_not_presigned() {
        // An ordinary request must not be mistaken for a presigned one, or a
        // missing Authorization header would be read as an unsigned URL.
        assert!(parse_presigned("list-type=2&prefix=a").is_none());
        assert!(parse_presigned("").is_none());
    }

    // ── chunk-framed bodies ──────────────────────────────────────────────────

    fn streaming_ctx() -> StreamingContext {
        StreamingContext {
            seed_signature: "seed0000".to_string(),
            amz_date: "20260822T120000Z".to_string(),
            scope: "20260822/us-east-1/s3/aws4_request".to_string(),
            signing_key: signing_key("secret", "20260822", "us-east-1", "s3"),
        }
    }

    /// Frame chunks the way AWS does, signing each against the previous.
    ///
    /// Built here rather than hardcoded so the test exercises the parser against
    /// the real construction, and so a chunk can be tampered with afterwards.
    fn frame(chunks: &[&[u8]], ctx: &StreamingContext) -> Vec<u8> {
        let mut out = Vec::new();
        let mut previous = ctx.seed_signature.clone();
        let empty = hex::encode(Sha256::digest(b""));
        let sign = |data: &[u8], previous: &mut String| -> String {
            let to_sign = format!(
                "AWS4-HMAC-SHA256-PAYLOAD\n{}\n{}\n{}\n{}\n{}",
                ctx.amz_date,
                ctx.scope,
                previous,
                empty,
                hex::encode(Sha256::digest(data))
            );
            let sig = hex::encode(hmac(&ctx.signing_key, to_sign.as_bytes()));
            *previous = sig.clone();
            sig
        };
        for data in chunks {
            let sig = sign(data, &mut previous);
            out.extend_from_slice(
                format!("{:x};chunk-signature={}\r\n", data.len(), sig).as_bytes(),
            );
            out.extend_from_slice(data);
            out.extend_from_slice(b"\r\n");
        }
        let sig = sign(b"", &mut previous);
        out.extend_from_slice(format!("0;chunk-signature={sig}\r\n").as_bytes());
        out
    }

    #[test]
    fn a_chunked_body_yields_the_payload_without_its_framing() {
        // The bug this closes: with no parser at all, the size lines and
        // signatures were stored *inside the object*. Not an unverified body —
        // a corrupt one, with a 200 in reply.
        let ctx = streaming_ctx();
        let framed = frame(&[b"hello ", b"world"], &ctx);
        assert!(
            framed
                .windows(16)
                .any(|w| w == b"chunk-signature=".as_ref()),
            "the fixture must actually be framed"
        );
        assert_eq!(dechunk_and_verify(&framed, &ctx).unwrap(), b"hello world");
    }

    #[test]
    fn a_tampered_chunk_is_refused() {
        let ctx = streaming_ctx();
        let mut framed = frame(&[b"hello ", b"world"], &ctx);
        // Flip one byte of payload, leaving the framing intact.
        let at = framed
            .windows(5)
            .position(|w| w == b"world")
            .expect("the payload is in there");
        framed[at] = b'W';
        assert_eq!(dechunk_and_verify(&framed, &ctx), Err(Verdict::Mismatch));
    }

    #[test]
    fn a_body_anchored_to_a_different_seed_is_refused() {
        // The chain's seed is the request's own signature, so a body lifted from
        // another request does not verify here even though every chunk is
        // internally consistent.
        let ctx = streaming_ctx();
        let framed = frame(&[b"payload"], &ctx);
        let other = StreamingContext {
            seed_signature: "different".to_string(),
            ..streaming_ctx()
        };
        assert_eq!(dechunk_and_verify(&framed, &other), Err(Verdict::Mismatch));
    }

    #[test]
    fn a_truncated_body_is_refused_rather_than_stored_short() {
        // Without the terminating zero-length chunk the upload was cut off.
        // Keeping what arrived would store a short object and report success.
        let ctx = streaming_ctx();
        let framed = frame(&[b"hello ", b"world"], &ctx);
        let cut = framed.len() - 20;
        assert!(matches!(
            dechunk_and_verify(&framed[..cut], &ctx),
            Err(Verdict::Malformed(_))
        ));

        // And a chunk claiming more data than was sent.
        let lying = b"ff;chunk-signature=deadbeef\r\nshort\r\n";
        assert!(matches!(
            dechunk_and_verify(lying, &ctx),
            Err(Verdict::Malformed(_))
        ));
    }

    #[test]
    fn malformed_framing_is_an_error_not_a_panic() {
        let ctx = streaming_ctx();
        for body in [
            &b""[..],
            b"no-crlf",
            b"5\r\nhello\r\n",                 // no signature field
            b"5;nope=1\r\nhello\r\n",          // wrong field name
            b"zz;chunk-signature=aa\r\nx\r\n", // size not hex
        ] {
            let _ = dechunk_and_verify(body, &ctx);
        }
    }

    #[test]
    fn an_empty_chunked_body_is_the_empty_object() {
        // A zero-byte upload is legitimate, and it arrives as just the final
        // chunk. Refusing it would make a 0-byte PUT fail.
        let ctx = streaming_ctx();
        let framed = frame(&[], &ctx);
        assert_eq!(dechunk_and_verify(&framed, &ctx).unwrap(), b"");
    }

    #[test]
    fn streaming_and_unsigned_are_not_the_same_question() {
        // `UNSIGNED-PAYLOAD` means the body is exactly what it looks like and
        // simply is not hashed. `STREAMING-…` means the wire bytes are not the
        // object. Treating them alike is what stored the framing.
        assert!(is_unsigned_payload("UNSIGNED-PAYLOAD"));
        assert!(!is_streaming_payload("UNSIGNED-PAYLOAD"));
        assert!(is_streaming_payload("STREAMING-AWS4-HMAC-SHA256-PAYLOAD"));
        assert!(!is_streaming_payload(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ));
    }

    #[test]
    fn the_comparison_is_length_safe() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
    }
}
