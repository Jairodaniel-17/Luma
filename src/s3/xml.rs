//! S3 XML responses.
//!
//! Written by hand rather than derived. The documents are small and fixed, and
//! the thing that matters is that the element names and nesting match S3
//! **exactly** — clients parse by XPath-ish lookups and a renamed element is a
//! silent empty result rather than an error. Serde's XML support would hide that
//! decision behind attribute macros; here it is visible in the strings.
//!
//! Every document carries the `xmlns` S3 uses. Some clients ignore it; some
//! refuse a document without it, and a refusal at that layer is very hard to
//! diagnose from the other side.

const NS: &str = "http://s3.amazonaws.com/doc/2006-03-01/";

/// Escape the five XML entities.
///
/// Object keys are arbitrary bytes and routinely contain `&`. Skipping this
/// produces a document that parses as truncated, which reads to the client as
/// "the bucket ends here".
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// An error document.
///
/// The `Code` is what clients branch on — botocore maps it to an exception
/// class — so it must be one of S3's own codes rather than something
/// descriptive. The human-readable part goes in `Message`.
pub fn error(code: &str, message: &str, resource: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <Error><Code>{}</Code><Message>{}</Message><Resource>{}</Resource></Error>",
        escape(code),
        escape(message),
        escape(resource)
    )
}

pub struct ObjectEntry {
    pub key: String,
    pub size: u64,
    pub last_modified: String,
    pub etag: String,
}

/// `ListObjectsV2` — the listing every client and tool uses.
#[allow(clippy::too_many_arguments)]
pub fn list_objects_v2(
    bucket: &str,
    prefix: &str,
    delimiter: &str,
    max_keys: usize,
    truncated: bool,
    next_token: Option<&str>,
    objects: &[ObjectEntry],
    common_prefixes: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!("<ListBucketResult xmlns=\"{NS}\">"));
    out.push_str(&format!("<Name>{}</Name>", escape(bucket)));
    out.push_str(&format!("<Prefix>{}</Prefix>", escape(prefix)));
    if !delimiter.is_empty() {
        out.push_str(&format!("<Delimiter>{}</Delimiter>", escape(delimiter)));
    }
    out.push_str(&format!("<MaxKeys>{max_keys}</MaxKeys>"));
    out.push_str(&format!("<KeyCount>{}</KeyCount>", objects.len()));
    out.push_str(&format!("<IsTruncated>{truncated}</IsTruncated>"));
    if let Some(token) = next_token {
        out.push_str(&format!(
            "<NextContinuationToken>{}</NextContinuationToken>",
            escape(token)
        ));
    }
    for object in objects {
        out.push_str("<Contents>");
        out.push_str(&format!("<Key>{}</Key>", escape(&object.key)));
        out.push_str(&format!(
            "<LastModified>{}</LastModified>",
            escape(&object.last_modified)
        ));
        // Quoted, because S3 quotes it and clients compare the quoted form
        // against the `ETag` response header.
        out.push_str(&format!(
            "<ETag>&quot;{}&quot;</ETag>",
            escape(&object.etag)
        ));
        out.push_str(&format!("<Size>{}</Size>", object.size));
        out.push_str("<StorageClass>STANDARD</StorageClass>");
        out.push_str("</Contents>");
    }
    for prefix in common_prefixes {
        out.push_str(&format!(
            "<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>",
            escape(prefix)
        ));
    }
    out.push_str("</ListBucketResult>");
    out
}

/// `ListBuckets`.
pub fn list_buckets(buckets: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!("<ListAllMyBucketsResult xmlns=\"{NS}\">"));
    // The owner block is required by the schema even though Luma has no concept
    // of an S3 owner; the org is the owner and its id goes here.
    out.push_str("<Owner><ID>luma</ID><DisplayName>luma</DisplayName></Owner>");
    out.push_str("<Buckets>");
    for (name, created) in buckets {
        out.push_str(&format!(
            "<Bucket><Name>{}</Name><CreationDate>{}</CreationDate></Bucket>",
            escape(name),
            escape(created)
        ));
    }
    out.push_str("</Buckets></ListAllMyBucketsResult>");
    out
}

/// `InitiateMultipartUploadResult`.
pub fn initiate_multipart(bucket: &str, key: &str, upload_id: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <InitiateMultipartUploadResult xmlns=\"{NS}\">\
         <Bucket>{}</Bucket><Key>{}</Key><UploadId>{}</UploadId>\
         </InitiateMultipartUploadResult>",
        escape(bucket),
        escape(key),
        escape(upload_id)
    )
}

/// `CompleteMultipartUploadResult`.
pub fn complete_multipart(location: &str, bucket: &str, key: &str, etag: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <CompleteMultipartUploadResult xmlns=\"{NS}\">\
         <Location>{}</Location><Bucket>{}</Bucket><Key>{}</Key><ETag>&quot;{}&quot;</ETag>\
         </CompleteMultipartUploadResult>",
        escape(location),
        escape(bucket),
        escape(key),
        escape(etag)
    )
}

/// The `CopyObject` result document.
///
/// `CopyObject` answers `200` with a body, which is a shape worth noticing: a
/// client cannot treat the status alone as success, and a server that returns an
/// empty body — as this one did before the operation existed — makes every
/// client fail while parsing instead of at the request.
pub fn copy_object(etag: &str, last_modified: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <CopyObjectResult xmlns=\"{NS}\">\
         <ETag>&quot;{}&quot;</ETag><LastModified>{}</LastModified>\
         </CopyObjectResult>",
        escape(etag),
        escape(last_modified)
    )
}

/// Part numbers from a `CompleteMultipartUpload` body, in the order given.
///
/// A tiny reader rather than a parser: the document has one repeated element and
/// the only field that matters is `PartNumber`. The ETags a client echoes back
/// are not checked, and that is a stated gap — S3 verifies them, and a client
/// that sent the wrong ones would get a success here where S3 gives
/// `InvalidPart`.
pub fn parse_complete_parts(body: &str) -> Vec<u32> {
    let mut parts = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("<PartNumber>") {
        let after = &rest[start + "<PartNumber>".len()..];
        let Some(end) = after.find("</PartNumber>") else {
            break;
        };
        if let Ok(number) = after[..end].trim().parse::<u32>() {
            parts.push(number);
        }
        rest = &after[end..];
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ampersand_in_a_key_does_not_truncate_the_document() {
        // Object keys are arbitrary bytes and `&` is common. Unescaped, the
        // document parses as truncated, which reads to a client as "the bucket
        // ends here" rather than as an error.
        let xml = list_objects_v2(
            "b",
            "",
            "",
            1000,
            false,
            None,
            &[ObjectEntry {
                key: "reports/q1&q2.csv".into(),
                size: 10,
                last_modified: "2026-08-22T00:00:00.000Z".into(),
                etag: "abc".into(),
            }],
            &[],
        );
        assert!(xml.contains("reports/q1&amp;q2.csv"));
        assert!(!xml.contains("q1&q2"));
    }

    #[test]
    fn the_listing_carries_the_namespace_clients_expect() {
        let xml = list_objects_v2("b", "", "", 1000, false, None, &[], &[]);
        assert!(xml.contains("xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\""));
        assert!(xml.contains("<ListBucketResult"));
        assert!(xml.contains("<KeyCount>0</KeyCount>"));
        assert!(xml.contains("<IsTruncated>false</IsTruncated>"));
    }

    #[test]
    fn the_etag_is_quoted_inside_the_element() {
        // S3 quotes it, and clients compare the quoted form against the ETag
        // response header. An unquoted one silently fails every conditional
        // request.
        let xml = list_objects_v2(
            "b",
            "",
            "",
            10,
            false,
            None,
            &[ObjectEntry {
                key: "k".into(),
                size: 1,
                last_modified: "x".into(),
                etag: "deadbeef".into(),
            }],
            &[],
        );
        assert!(xml.contains("<ETag>&quot;deadbeef&quot;</ETag>"));
    }

    #[test]
    fn a_delimiter_produces_common_prefixes() {
        let xml = list_objects_v2(
            "b",
            "logs/",
            "/",
            10,
            false,
            None,
            &[],
            &["logs/2026/".to_string()],
        );
        assert!(xml.contains("<Delimiter>/</Delimiter>"));
        assert!(xml.contains("<CommonPrefixes><Prefix>logs/2026/</Prefix></CommonPrefixes>"));
    }

    #[test]
    fn a_delimiter_is_omitted_when_absent_rather_than_sent_empty() {
        // An empty `<Delimiter/>` is not the same as no delimiter to some
        // clients, and S3 omits it.
        let xml = list_objects_v2("b", "", "", 10, false, None, &[], &[]);
        assert!(!xml.contains("<Delimiter>"));
    }

    #[test]
    fn the_error_document_uses_the_code_clients_branch_on() {
        let xml = error("NoSuchKey", "The specified key does not exist.", "/b/k");
        assert!(xml.contains("<Code>NoSuchKey</Code>"));
        assert!(xml.contains("<Resource>/b/k</Resource>"));
    }

    #[test]
    fn part_numbers_are_read_in_the_order_given() {
        let body = "<CompleteMultipartUpload>\
            <Part><PartNumber>1</PartNumber><ETag>\"a\"</ETag></Part>\
            <Part><PartNumber>2</PartNumber><ETag>\"b\"</ETag></Part>\
            </CompleteMultipartUpload>";
        assert_eq!(parse_complete_parts(body), vec![1, 2]);
    }

    #[test]
    fn a_body_with_no_parts_reads_as_empty_rather_than_looping() {
        assert!(parse_complete_parts("<CompleteMultipartUpload/>").is_empty());
        // Malformed: an opening tag with no closing one must terminate.
        assert!(parse_complete_parts("<PartNumber>1").is_empty());
    }
}
