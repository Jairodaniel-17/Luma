//! RESP wire format: encoding and incremental decoding.
//!
//! F1.1 of `docs/PLAN-MAESTRO.md`, decision D2 of `docs/SPEC-resp.md`. Written
//! here rather than pulled from a crate: RESP is a prefix-length protocol that
//! has been stable for over a decade, the whole thing is a few hundred lines,
//! and a dependency on the parsing of untrusted network bytes is one we would
//! rather own — see the `lru` use-after-free that `cargo deny` caught during
//! block 3.
//!
//! ## Incremental by construction
//!
//! [`Decoder::decode`] returns `Ok(None)` when the buffer holds only part of a
//! frame, and consumes nothing. A connection loop can therefore read whatever
//! the socket gave it and try again, which is the only correct way to handle a
//! stream: TCP is free to split a frame across any number of reads, and a parser
//! that assumes one read is one command works right up until it meets a real
//! network.
//!
//! ## Inline commands
//!
//! `PING\r\n` — a bare line with no array framing — is what `redis-cli` sends
//! when you type into it, and what a person reaches for with `nc` to check a
//! server is alive. Supporting it costs a few lines and its absence looks like
//! a broken server.

use std::fmt;

/// Maximum elements in one array frame, and maximum bytes in one bulk string.
///
/// The lengths come off the wire as untrusted numbers, so an unbounded
/// `with_capacity` on them is a one-packet out-of-memory. Redis uses 512 MiB for
/// bulk strings; the array bound is well above any real pipeline.
const MAX_BULK_LEN: usize = 512 * 1024 * 1024;
const MAX_ARRAY_LEN: usize = 1024 * 1024;

/// A RESP value.
///
/// Deliberately not modelling every RESP3 type: the clients this targets speak
/// RESP2 plus `HELLO`, and each extra type is another shape to get wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `+OK\r\n`
    Simple(String),
    /// `-ERR message\r\n`
    Error(String),
    /// `:42\r\n`
    Integer(i64),
    /// `$3\r\nfoo\r\n`, or `$-1\r\n` for a null bulk string.
    Bulk(Option<Vec<u8>>),
    /// `*2\r\n...`, or `*-1\r\n` for a null array.
    Array(Option<Vec<Value>>),
    /// `%2\r\n...` — RESP3 map, used only to answer `HELLO 3`.
    Map(Vec<(Value, Value)>),
}

impl Value {
    pub fn ok() -> Value {
        Value::Simple("OK".to_string())
    }

    pub fn nil() -> Value {
        Value::Bulk(None)
    }

    pub fn bulk(bytes: impl Into<Vec<u8>>) -> Value {
        Value::Bulk(Some(bytes.into()))
    }

    /// Interpret this value as command arguments.
    ///
    /// A command is an array of bulk strings; anything else is a protocol
    /// error rather than something to coerce, because guessing at a malformed
    /// command is how a proxy ends up executing something nobody sent.
    pub fn into_command(self) -> Result<Vec<Vec<u8>>, ProtocolError> {
        match self {
            Value::Array(Some(items)) => items
                .into_iter()
                .map(|item| match item {
                    Value::Bulk(Some(bytes)) => Ok(bytes),
                    // Redis itself sends inline integers in some replies, but a
                    // *command* argument is always a bulk string.
                    _ => Err(ProtocolError::Malformed(
                        "command arguments must be bulk strings",
                    )),
                })
                .collect(),
            _ => Err(ProtocolError::Malformed("expected an array of arguments")),
        }
    }

    /// Serialize to the wire.
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Value::Simple(text) => {
                out.push(b'+');
                out.extend_from_slice(text.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Value::Error(text) => {
                out.push(b'-');
                out.extend_from_slice(text.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Value::Integer(n) => {
                out.push(b':');
                out.extend_from_slice(n.to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Value::Bulk(None) => out.extend_from_slice(b"$-1\r\n"),
            Value::Bulk(Some(bytes)) => {
                out.push(b'$');
                out.extend_from_slice(bytes.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                out.extend_from_slice(bytes);
                out.extend_from_slice(b"\r\n");
            }
            Value::Array(None) => out.extend_from_slice(b"*-1\r\n"),
            Value::Array(Some(items)) => {
                out.push(b'*');
                out.extend_from_slice(items.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for item in items {
                    item.encode(out);
                }
            }
            Value::Map(pairs) => {
                out.push(b'%');
                out.extend_from_slice(pairs.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for (key, value) in pairs {
                    key.encode(out);
                    value.encode(out);
                }
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }
}

#[derive(Debug, PartialEq)]
pub enum ProtocolError {
    /// The bytes are not valid RESP. The connection must be closed: a stream
    /// whose framing is wrong cannot be resynchronised by skipping ahead.
    Malformed(&'static str),
    /// A declared length exceeds what we are willing to allocate.
    TooLarge(&'static str),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolError::Malformed(why) => write!(f, "Protocol error: {why}"),
            ProtocolError::TooLarge(what) => write!(f, "Protocol error: {what}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Incremental RESP decoder over a caller-owned buffer.
pub struct Decoder;

impl Decoder {
    /// Try to decode one frame from the front of `buf`.
    ///
    /// Returns `Ok(Some((value, consumed)))` on success, where `consumed` is how
    /// many bytes the caller should drop. Returns `Ok(None)` when the frame is
    /// incomplete — in that case **nothing** has been consumed and the caller
    /// should read more bytes and retry.
    pub fn decode(buf: &[u8]) -> Result<Option<(Value, usize)>, ProtocolError> {
        if buf.is_empty() {
            return Ok(None);
        }
        match buf[0] {
            b'+' | b'-' | b':' | b'$' | b'*' | b'%' => Self::decode_typed(buf),
            // Anything else starts an inline command.
            _ => Self::decode_inline(buf),
        }
    }

    fn decode_typed(buf: &[u8]) -> Result<Option<(Value, usize)>, ProtocolError> {
        let Some(line_end) = find_crlf(buf, 1) else {
            return Ok(None);
        };
        let line = &buf[1..line_end];
        let after_line = line_end + 2;

        match buf[0] {
            b'+' => Ok(Some((Value::Simple(to_text(line)?), after_line))),
            b'-' => Ok(Some((Value::Error(to_text(line)?), after_line))),
            b':' => Ok(Some((Value::Integer(parse_int(line)?), after_line))),
            b'$' => {
                let len = parse_int(line)?;
                if len < 0 {
                    return Ok(Some((Value::Bulk(None), after_line)));
                }
                let len = len as usize;
                if len > MAX_BULK_LEN {
                    return Err(ProtocolError::TooLarge("bulk string too long"));
                }
                // The payload plus its own trailing CRLF.
                if buf.len() < after_line + len + 2 {
                    return Ok(None);
                }
                let payload = buf[after_line..after_line + len].to_vec();
                if &buf[after_line + len..after_line + len + 2] != b"\r\n" {
                    return Err(ProtocolError::Malformed("bulk string not CRLF-terminated"));
                }
                Ok(Some((Value::Bulk(Some(payload)), after_line + len + 2)))
            }
            b'*' | b'%' => {
                let declared = parse_int(line)?;
                if declared < 0 {
                    return Ok(Some((Value::Array(None), after_line)));
                }
                let declared = declared as usize;
                if declared > MAX_ARRAY_LEN {
                    return Err(ProtocolError::TooLarge("array too long"));
                }
                // A map declares pairs, so it carries twice as many frames.
                let count = if buf[0] == b'%' {
                    declared * 2
                } else {
                    declared
                };

                let mut offset = after_line;
                // No `with_capacity(declared)`: the count is untrusted, and
                // reserving on it lets one 5-byte header ask for gigabytes.
                let mut items = Vec::new();
                for _ in 0..count {
                    match Self::decode(&buf[offset..])? {
                        Some((value, used)) => {
                            offset += used;
                            items.push(value);
                        }
                        None => return Ok(None),
                    }
                }
                if buf[0] == b'%' {
                    let pairs = items
                        .chunks_exact(2)
                        .map(|pair| (pair[0].clone(), pair[1].clone()))
                        .collect();
                    Ok(Some((Value::Map(pairs), offset)))
                } else {
                    Ok(Some((Value::Array(Some(items)), offset)))
                }
            }
            _ => unreachable!("decode_typed called on an untyped frame"),
        }
    }

    /// Decode a bare `PING\r\n`-style line into the same shape a typed command
    /// would produce, so the command layer never has to care which form arrived.
    fn decode_inline(buf: &[u8]) -> Result<Option<(Value, usize)>, ProtocolError> {
        let Some(line_end) = find_crlf(buf, 0) else {
            // Guard against a peer that never sends a newline: without this the
            // buffer would grow until the connection limit or memory ran out.
            if buf.len() > 64 * 1024 {
                return Err(ProtocolError::TooLarge("inline command too long"));
            }
            return Ok(None);
        };
        let line = &buf[..line_end];
        let args: Vec<Value> = line
            .split(|b| b.is_ascii_whitespace())
            .filter(|part| !part.is_empty())
            .map(|part| Value::Bulk(Some(part.to_vec())))
            .collect();
        if args.is_empty() {
            // An empty line is a no-op in Redis, not an error.
            return Ok(Some((Value::Array(Some(Vec::new())), line_end + 2)));
        }
        Ok(Some((Value::Array(Some(args)), line_end + 2)))
    }
}

fn find_crlf(buf: &[u8], from: usize) -> Option<usize> {
    if from >= buf.len() {
        return None;
    }
    buf[from..]
        .windows(2)
        .position(|pair| pair == b"\r\n")
        .map(|pos| from + pos)
}

fn to_text(bytes: &[u8]) -> Result<String, ProtocolError> {
    String::from_utf8(bytes.to_vec())
        .map_err(|_| ProtocolError::Malformed("line is not valid UTF-8"))
}

fn parse_int(bytes: &[u8]) -> Result<i64, ProtocolError> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|text| text.trim().parse::<i64>().ok())
        .ok_or(ProtocolError::Malformed("expected an integer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(bytes: &[u8]) -> Value {
        let (value, used) = Decoder::decode(bytes).unwrap().expect("frame incomplete");
        assert_eq!(used, bytes.len(), "decoder must consume the whole frame");
        value
    }

    // ── encoding ─────────────────────────────────────────────────────────────

    #[test]
    fn encodes_every_type_to_the_documented_bytes() {
        assert_eq!(Value::ok().to_bytes(), b"+OK\r\n");
        assert_eq!(Value::Error("ERR nope".into()).to_bytes(), b"-ERR nope\r\n");
        assert_eq!(Value::Integer(-42).to_bytes(), b":-42\r\n");
        assert_eq!(Value::bulk("foo").to_bytes(), b"$3\r\nfoo\r\n");
        assert_eq!(Value::nil().to_bytes(), b"$-1\r\n");
        assert_eq!(Value::Array(None).to_bytes(), b"*-1\r\n");
        assert_eq!(
            Value::Array(Some(vec![Value::bulk("a"), Value::Integer(1)])).to_bytes(),
            b"*2\r\n$1\r\na\r\n:1\r\n"
        );
    }

    #[test]
    fn an_empty_bulk_string_is_not_a_nil() {
        // `$0\r\n\r\n` and `$-1\r\n` mean different things — an empty value that
        // exists, versus no value. Conflating them makes GET on an empty string
        // look like a missing key.
        assert_eq!(Value::bulk("").to_bytes(), b"$0\r\n\r\n");
        assert_ne!(Value::bulk("").to_bytes(), Value::nil().to_bytes());
        assert_eq!(decode_all(b"$0\r\n\r\n"), Value::Bulk(Some(Vec::new())));
        assert_eq!(decode_all(b"$-1\r\n"), Value::Bulk(None));
    }

    #[test]
    fn an_empty_array_is_not_a_nil_array() {
        // The other classic compatibility trap: LRANGE on a missing key returns
        // an empty array, not a null one, and clients branch on the difference.
        assert_eq!(Value::Array(Some(Vec::new())).to_bytes(), b"*0\r\n");
        assert_eq!(decode_all(b"*0\r\n"), Value::Array(Some(Vec::new())));
        assert_eq!(decode_all(b"*-1\r\n"), Value::Array(None));
    }

    // ── decoding ─────────────────────────────────────────────────────────────

    #[test]
    fn decodes_a_typed_command() {
        let frame = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n";
        let args = decode_all(frame).into_command().unwrap();
        assert_eq!(args, vec![b"SET".to_vec(), b"k".to_vec(), b"v".to_vec()]);
    }

    #[test]
    fn decodes_binary_payloads_unchanged() {
        // The reason arguments are bytes and not strings: a pickled Celery body
        // is not valid UTF-8, and mangling it would corrupt every task.
        let mut frame = b"*2\r\n$3\r\nSET\r\n$4\r\n".to_vec();
        frame.extend_from_slice(&[0x00, 0xFF, 0xFE, 0x80]);
        frame.extend_from_slice(b"\r\n");
        let args = decode_all(&frame).into_command().unwrap();
        assert_eq!(args[1], vec![0x00, 0xFF, 0xFE, 0x80]);
    }

    #[test]
    fn a_payload_containing_crlf_is_read_by_length_not_by_scanning() {
        // Bulk strings are length-prefixed precisely so their contents can hold
        // CRLF. A scanning parser would truncate here.
        // The payload is five bytes; the length header is what tells the parser
        // where it ends, so the embedded CRLF is just data.
        let frame = b"*1\r\n$5\r\na\r\nb\r\r\n";
        let args = decode_all(frame).into_command().unwrap();
        assert_eq!(args[0], b"a\r\nb\r".to_vec());
    }

    #[test]
    fn decodes_an_inline_command() {
        // What redis-cli sends when a human types, and what `nc` gets used for.
        let args = decode_all(b"PING\r\n").into_command().unwrap();
        assert_eq!(args, vec![b"PING".to_vec()]);

        let args = decode_all(b"SET  key   value\r\n").into_command().unwrap();
        assert_eq!(
            args,
            vec![b"SET".to_vec(), b"key".to_vec(), b"value".to_vec()],
            "runs of whitespace must collapse"
        );
    }

    #[test]
    fn an_empty_inline_line_is_a_no_op_not_an_error() {
        assert_eq!(decode_all(b"\r\n"), Value::Array(Some(Vec::new())));
    }

    // ── incremental behaviour ────────────────────────────────────────────────

    #[test]
    fn a_partial_frame_consumes_nothing_and_asks_for_more() {
        // TCP splits frames wherever it likes. A parser that assumes one read is
        // one command works until it meets a real network.
        let full = b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n";
        for split in 1..full.len() {
            assert_eq!(
                Decoder::decode(&full[..split]).unwrap(),
                None,
                "a {split}-byte prefix must decode as incomplete, not as a frame"
            );
        }
        assert!(Decoder::decode(full).unwrap().is_some());
    }

    #[test]
    fn pipelined_commands_are_decoded_one_at_a_time() {
        // Pipelining is a client sending N commands before reading any reply,
        // so several frames arrive in one read.
        let buf = b"*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nECHO\r\n";
        let mut offset = 0;
        let mut commands = Vec::new();
        while offset < buf.len() {
            let (value, used) = Decoder::decode(&buf[offset..]).unwrap().unwrap();
            offset += used;
            commands.push(value.into_command().unwrap());
        }
        assert_eq!(commands.len(), 3);
        assert_eq!(commands[2][0], b"ECHO".to_vec());
    }

    #[test]
    fn a_partially_arrived_array_element_leaves_the_whole_array_pending() {
        // The nested case: the outer header is complete but an inner bulk string
        // is not. Consuming the header now would lose the framing.
        let partial = b"*2\r\n$3\r\nGET\r\n$3\r\nfo";
        assert_eq!(Decoder::decode(partial).unwrap(), None);
    }

    // ── hostile input ────────────────────────────────────────────────────────

    #[test]
    fn an_absurd_bulk_length_is_refused_before_allocating() {
        // Five bytes of header asking for a gigabyte is a one-packet OOM.
        let err = Decoder::decode(b"$999999999999\r\n").unwrap_err();
        assert_eq!(err, ProtocolError::TooLarge("bulk string too long"));
    }

    #[test]
    fn an_absurd_array_length_is_refused_before_allocating() {
        let err = Decoder::decode(b"*999999999\r\n").unwrap_err();
        assert_eq!(err, ProtocolError::TooLarge("array too long"));
    }

    #[test]
    fn a_huge_declared_array_does_not_reserve_memory_up_front() {
        // Just under the cap: this must simply report "incomplete" rather than
        // reserving a million slots for elements that never arrive.
        assert_eq!(Decoder::decode(b"*1000000\r\n").unwrap(), None);
    }

    #[test]
    fn a_non_numeric_length_is_a_protocol_error() {
        assert_eq!(
            Decoder::decode(b"$abc\r\n").unwrap_err(),
            ProtocolError::Malformed("expected an integer")
        );
    }

    #[test]
    fn a_bulk_string_that_is_not_crlf_terminated_is_rejected() {
        // Trusting the declared length without checking the terminator lets a
        // desynchronised stream be read as valid commands.
        assert_eq!(
            Decoder::decode(b"$3\r\nfooXX").unwrap_err(),
            ProtocolError::Malformed("bulk string not CRLF-terminated")
        );
    }

    #[test]
    fn an_endless_inline_line_is_bounded() {
        // A peer that opens a connection and never sends a newline must not be
        // able to grow our buffer without limit.
        let flood = vec![b'A'; 64 * 1024 + 1];
        assert_eq!(
            Decoder::decode(&flood).unwrap_err(),
            ProtocolError::TooLarge("inline command too long")
        );
    }

    #[test]
    fn a_command_with_a_non_bulk_argument_is_refused() {
        let value = Value::Array(Some(vec![Value::Integer(1)]));
        assert!(value.into_command().is_err());
    }

    #[test]
    fn a_bare_scalar_is_not_a_command() {
        assert!(Value::Integer(1).into_command().is_err());
    }

    // ── round trip ───────────────────────────────────────────────────────────

    #[test]
    fn every_value_survives_encode_then_decode() {
        let values = vec![
            Value::Simple("OK".into()),
            Value::Error("WRONGTYPE nope".into()),
            Value::Integer(0),
            Value::Integer(i64::MIN),
            Value::Bulk(None),
            Value::bulk(""),
            Value::bulk(vec![0u8, 255, 128]),
            Value::Array(None),
            Value::Array(Some(vec![])),
            Value::Array(Some(vec![
                Value::bulk("nested"),
                Value::Array(Some(vec![Value::Integer(7)])),
            ])),
        ];
        for value in values {
            let bytes = value.to_bytes();
            assert_eq!(decode_all(&bytes), value, "round trip failed for {value:?}");
        }
    }

    #[test]
    fn a_resp3_map_round_trips_for_hello() {
        let map = Value::Map(vec![
            (Value::bulk("server"), Value::bulk("luma")),
            (Value::bulk("proto"), Value::Integer(3)),
        ]);
        assert_eq!(decode_all(&map.to_bytes()), map);
    }
}
