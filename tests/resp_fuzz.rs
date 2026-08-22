//! The RESP parser must never panic, however hostile the bytes.
//!
//! F4.5 of `docs/PLAN-MAESTRO.md`. The parser is the first thing an unauthenticated
//! peer reaches on an open TCP port: a panic there is a crashed process for the
//! cost of one packet, and `AUTH` cannot protect it because parsing happens
//! before the command is known.
//!
//! ## Why this is not `cargo-fuzz`
//!
//! A real fuzzer explores far better than a seeded generator, and it belongs in
//! the nightly job. This is the part that has to run on **every** commit: a
//! bounded, deterministic sweep that a contributor sees fail in seconds rather
//! than the morning after. The two are complements, not alternatives.
//!
//! Determinism matters more than coverage here. A random seed would make a
//! failure unreproducible, which is the difference between a bug report and a
//! shrug — so the generator is a fixed-seed LCG, and any input it finds can be
//! replayed by re-running the test.
//!
//! ## The corpus
//!
//! `CORPUS` is the versioned part: frames that broke something once, or that
//! encode a distinction the protocol depends on. Entries are added, never
//! removed. Every one is also used as a seed for mutation, because the
//! interesting neighbourhood of a bug is usually right next to the bug.

use luma::resp::protocol::Decoder;

/// Frames worth keeping forever.
///
/// Each line is either a real distinction in RESP2 or something that once went
/// wrong. Nothing here is arbitrary: a corpus of noise is just a slower random
/// generator.
const CORPUS: &[&[u8]] = &[
    // The empty case and the smallest valid frames.
    b"",
    b"\r\n",
    b"PING\r\n",
    b"+OK\r\n",
    b"-ERR something\r\n",
    b":0\r\n",
    b":-1\r\n",
    // `$0` is an empty string and `$-1` is a nil. Conflating them is the
    // classic RESP bug, so both stay in the corpus.
    b"$0\r\n\r\n",
    b"$-1\r\n",
    b"$5\r\nhello\r\n",
    // `*0` is an empty array and `*-1` is a nil array — the same distinction one
    // level up.
    b"*0\r\n",
    b"*-1\r\n",
    b"*1\r\n$4\r\nPING\r\n",
    b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n",
    // Nesting, which is where a recursive parser blows its stack.
    b"*1\r\n*1\r\n*1\r\n$1\r\na\r\n",
    // Inline commands: redis-cli sends them, so they are not a curiosity.
    b"SET k v\r\n",
    b"  \r\n",
    b"SET  k   v  \r\n",
    // Lengths that do not match the body.
    b"$100\r\nshort\r\n",
    b"$2\r\ntoolong\r\n",
    b"*5\r\n$1\r\na\r\n",
    // Negative and absurd lengths.
    b"$-2\r\n",
    b"*-2\r\n",
    b"$99999999999999999999\r\n",
    b"*99999999999999999999\r\n",
    b"$-99999999999999999999\r\n",
    // Not a number at all.
    b"$abc\r\n",
    b"*abc\r\n",
    b":abc\r\n",
    b"$\r\n",
    b"*\r\n",
    // A bare CR or LF where CRLF is required.
    b"$5\rhello\r\n",
    b"$5\nhello\n",
    b"+OK\r",
    b"+OK\n",
    // An unknown type byte.
    b"!oops\r\n",
    b"\x00\r\n",
    b"\xff\xfe\xfd",
    // RESP3 map, which the parser recognises even though the server speaks 2.
    b"%1\r\n$1\r\na\r\n$1\r\nb\r\n",
    // A binary payload, because keys and values are bytes and not text.
    b"*2\r\n$3\r\nGET\r\n$3\r\n\x00\xff\x80\r\n",
];

/// A fixed-seed linear congruential generator.
///
/// Deterministic on purpose: a failure found here is reproducible by re-running
/// the test, with no seed to copy out of a log.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes constants; adequate for shuffling bytes and no
        // cryptographic claim is being made.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// Mutate a frame in one of a few ways that have historically found parser bugs.
fn mutate(seed: &[u8], rng: &mut Lcg) -> Vec<u8> {
    let mut out = seed.to_vec();
    match rng.below(6) {
        // Flip a byte: the length prefixes and type bytes are where this bites.
        0 if !out.is_empty() => {
            let at = rng.below(out.len());
            out[at] ^= 1 << rng.below(8);
        }
        // Truncate: every prefix of a valid frame must be "need more bytes",
        // never a panic and never a wrong parse.
        1 if !out.is_empty() => {
            let keep = rng.below(out.len());
            out.truncate(keep);
        }
        // Splice in a byte.
        2 => {
            let at = rng.below(out.len() + 1);
            out.insert(at, rng.below(256) as u8);
        }
        // Delete a byte, which most often kills a CR or an LF.
        3 if !out.is_empty() => {
            let at = rng.below(out.len());
            out.remove(at);
        }
        // Repeat the frame: pipelining, and the decoder must stop at the first.
        4 => {
            let copy = out.clone();
            out.extend_from_slice(&copy);
        }
        // Append junk after a complete frame.
        _ => {
            for _ in 0..rng.below(8) {
                out.push(rng.below(256) as u8);
            }
        }
    }
    out
}

/// Decode, and require that whatever happens is one of the three legal outcomes.
///
/// The panic message carries the input, because a fuzz failure with no input is
/// a note saying something is wrong somewhere.
fn decode_must_not_panic(input: &[u8]) {
    // Incomplete and protocol errors are both fine; a parsed frame must not
    // claim to have consumed more than it was given, or the connection loop
    // would drain past the end of its buffer.
    if let Ok(Some((_, used))) = Decoder::decode(input) {
        assert!(
            used <= input.len(),
            "decoder consumed {used} of {} bytes: {input:?}",
            input.len()
        );
    }
}

#[test]
fn the_corpus_never_panics() {
    for frame in CORPUS {
        decode_must_not_panic(frame);
    }
}

#[test]
fn every_prefix_of_every_corpus_frame_is_safe() {
    // A frame arrives one packet at a time. Every prefix is a state the parser
    // will really be handed, and the shortest path to a slice-out-of-bounds.
    for frame in CORPUS {
        for end in 0..=frame.len() {
            decode_must_not_panic(&frame[..end]);
        }
    }
}

#[test]
fn mutations_of_the_corpus_never_panic() {
    let mut rng = Lcg(0x5eed_1234_abcd_0001);
    for _ in 0..2_000 {
        let seed = CORPUS[rng.below(CORPUS.len())];
        let mut input = mutate(seed, &mut rng);
        // Chains of mutations reach further than single ones.
        for _ in 0..rng.below(3) {
            input = mutate(&input, &mut rng);
        }
        decode_must_not_panic(&input);
    }
}

#[test]
fn arbitrary_bytes_never_panic() {
    // No seed at all: purely random input, which is what an internet-facing port
    // gets from scanners within minutes of opening.
    let mut rng = Lcg(0x5eed_1234_abcd_0002);
    for _ in 0..2_000 {
        let len = rng.below(64);
        let input: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
        decode_must_not_panic(&input);
    }
}

#[test]
fn a_huge_declared_length_is_refused_rather_than_allocated() {
    // The denial-of-service shape: a small frame that asks the server to reserve
    // gigabytes. It must be rejected on the length, before any allocation.
    for frame in [
        b"$1073741824\r\n".as_slice(),
        b"*1073741824\r\n".as_slice(),
        b"$9223372036854775807\r\n".as_slice(),
        b"*9223372036854775807\r\n".as_slice(),
    ] {
        match Decoder::decode(frame) {
            Err(_) => {}
            Ok(None) => panic!(
                "a length of this size must be refused, not treated as \
                 'send me the rest': {frame:?}"
            ),
            Ok(Some(_)) => panic!("a length of this size must not parse: {frame:?}"),
        }
    }
}

#[test]
fn deep_nesting_does_not_blow_the_stack() {
    // A recursive descent parser is one `*1\r\n` away from a stack overflow, and
    // an overflow is not catchable — the process simply dies.
    let mut frame = Vec::new();
    for _ in 0..10_000 {
        frame.extend_from_slice(b"*1\r\n");
    }
    frame.extend_from_slice(b"$1\r\na\r\n");
    decode_must_not_panic(&frame);
}

#[test]
fn an_inline_command_of_only_separators_is_not_a_command() {
    // `into_command` has to survive whatever `decode` accepted. An inline line of
    // spaces parses as an empty argument list, and treating that as a command
    // would index an empty vector.
    for frame in [b"   \r\n".as_slice(), b"\t\t\r\n".as_slice(), b"\r\n"] {
        if let Ok(Some((value, _))) = Decoder::decode(frame) {
            if let Ok(args) = value.into_command() {
                assert!(
                    args.is_empty() || !args[0].is_empty(),
                    "an argument list must not start with an empty name: {args:?}"
                );
            }
        }
    }
}

#[test]
fn whatever_decodes_can_be_turned_into_a_command_without_panicking() {
    // The connection loop calls `into_command` on everything `decode` returns, so
    // the two have to agree about what is representable.
    let mut rng = Lcg(0x5eed_1234_abcd_0003);
    for _ in 0..2_000 {
        let seed = CORPUS[rng.below(CORPUS.len())];
        let input = mutate(seed, &mut rng);
        if let Ok(Some((value, _))) = Decoder::decode(&input) {
            let _ = value.into_command();
        }
    }
}
