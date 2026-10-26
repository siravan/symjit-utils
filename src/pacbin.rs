//! pyAmpliCol's `pacbin-v1` evaluator containers (`evaluators.pacbin` of an artifact): a reader
//! with the checks of pyAmpliCol's own (`rusticol-core/src/pacbin.rs`, 0BSD, pyamplicol 1.0.1),
//! and a writer that produces the bytes its writer produces for the same members.
//!
//! Layout (little endian): a 64-byte header, the member payloads in the order of their paths,
//! each aligned to 64 bytes with zero padding, the index (a 32-byte index header, then per
//! member a 56-byte entry, its path and zero padding to 8 bytes; the paths sorted bytewise),
//! and a 64-byte footer holding the SHA-256 of the index. Each entry holds the SHA-256 of its
//! payload.
//!
//! ```text
//! header  0 "PACBIN\0\0"  8 version u16 = 1  10 size u16 = 64  12 flags u32 = 0
//!         16 alignment u32 = 64  20 reserved u32  24 index offset u64  32 member count u64
//!         40..64 reserved
//! index   0 "PACIDX\0\0"  8 version u16  10 size u16 = 32  12 flags u32  16 member count u64
//!         24 reserved u64
//! entry   0 path length u32  4 kind u16  6 flags u16  8 offset u64  16 length u64  24 sha256
//! footer  0 "PACEND\0\0"  8 version u16  10 size u16 = 64  12 flags u32  16 index offset u64
//!         24 member count u64  32 index sha256
//! ```

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, Result};

use crate::sha256::Sha256;

const HEADER_MAGIC: &[u8; 8] = b"PACBIN\0\0";
const INDEX_MAGIC: &[u8; 8] = b"PACIDX\0\0";
const FOOTER_MAGIC: &[u8; 8] = b"PACEND\0\0";
const VERSION: u16 = 1;
const ALIGNMENT: u64 = 64;
const HEADER_SIZE: usize = 64;
const INDEX_HEADER_SIZE: usize = 32;
const ENTRY_SIZE: usize = 56;
const FOOTER_SIZE: usize = 64;
const INDEX_ALIGNMENT: u64 = 8;

/// The member kinds of `pacbin-v1` (pyAmpliCol's `PacbinMemberKind`; 6 is unassigned).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// a saved symjit application (`.symjit`, `.plane.symjit`): a kernel
    SymjitApplication = 1,
    /// a saved Symbolica evaluator (`.evaluator.bin`)
    SymbolicaExactState = 2,
    NativeLibrary = 3,
    EagerRuntimeMetadata = 4,
    EagerRuntimeTable = 5,
    RecurrenceDirectPlan = 7,
    RecurrenceColorProjectionCertificate = 8,
    OnTheFlyProcessSeed = 9,
    ColorContraction = 10,
}

impl Kind {
    pub const ALL: [Kind; 9] = [
        Kind::SymjitApplication,
        Kind::SymbolicaExactState,
        Kind::NativeLibrary,
        Kind::EagerRuntimeMetadata,
        Kind::EagerRuntimeTable,
        Kind::RecurrenceDirectPlan,
        Kind::RecurrenceColorProjectionCertificate,
        Kind::OnTheFlyProcessSeed,
        Kind::ColorContraction,
    ];

    pub fn from_code(code: u16) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.code() == code)
    }

    pub fn code(self) -> u16 {
        self as u16
    }

    /// The name pyAmpliCol's Python codec uses.
    pub fn name(self) -> &'static str {
        match self {
            Kind::SymjitApplication => "symjit-application",
            Kind::SymbolicaExactState => "symbolica-exact-state",
            Kind::NativeLibrary => "native-library",
            Kind::EagerRuntimeMetadata => "eager-runtime-metadata",
            Kind::EagerRuntimeTable => "eager-runtime-table",
            Kind::RecurrenceDirectPlan => "recurrence-direct-plan",
            Kind::RecurrenceColorProjectionCertificate => "recurrence-color-projection-certificate",
            Kind::OnTheFlyProcessSeed => "on-the-fly-process-seed",
            Kind::ColorContraction => "color-contraction",
        }
    }

    /// A kind by its name or code.
    pub fn parse(s: &str) -> Option<Kind> {
        if let Ok(code) = s.parse::<u16>() {
            return Kind::from_code(code);
        }
        Kind::ALL.into_iter().find(|k| k.name() == s)
    }
}

/// One member of a container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub path: String,
    pub kind: Kind,
    pub offset: u64,
    pub length: u64,
    pub sha256: [u8; 32],
}

/// A container read and checked by `Pacbin::read`.
#[derive(Debug)]
pub struct Pacbin {
    bytes: Vec<u8>,
    pub members: Vec<Member>,
    pub index_offset: u64,
    pub index_sha256: [u8; 32],
}

impl Pacbin {
    /// Reads a container, with every check of pyAmpliCol's reader: magic numbers, versions,
    /// sizes, flags and reserved fields; canonical, sorted, unique (also ignoring ASCII case)
    /// paths; the canonical payload layout with zero padding; the index digest and, with
    /// `verify_payloads`, every payload digest (pyAmpliCol verifies them unless the container
    /// is authenticated by the artifact's digest of the whole file).
    pub fn read(bytes: Vec<u8>, verify_payloads: bool) -> Result<Pacbin> {
        let (members, index_offset, index_sha256) = parse(&bytes, verify_payloads)?;
        Ok(Pacbin {
            bytes,
            members,
            index_offset,
            index_sha256,
        })
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The whole container.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The payload of `member`.
    pub fn payload(&self, member: &Member) -> &[u8] {
        &self.bytes[member.offset as usize..(member.offset + member.length) as usize]
    }

    /// The member at `path` (normalized as pyAmpliCol normalizes it).
    pub fn member(&self, path: &str) -> Option<&Member> {
        let path = normalize_path(path).ok()?;
        self.members
            .binary_search_by(|m| m.path.as_bytes().cmp(path.as_bytes()))
            .ok()
            .map(|i| &self.members[i])
    }

    /// The members as `write` takes them.
    pub fn contents(&self) -> Vec<(String, Kind, &[u8])> {
        self.members
            .iter()
            .map(|m| (m.path.clone(), m.kind, self.payload(m)))
            .collect()
    }
}

/// A container of `members` (path, kind, payload), as pyAmpliCol's `write_pacbin_atomic` writes
/// it: paths normalized and sorted, duplicates (also ignoring ASCII case) rejected.
pub fn write(members: &[(String, Kind, &[u8])]) -> Result<Vec<u8>> {
    let mut sorted: Vec<(String, Kind, &[u8])> = members
        .iter()
        .map(|(p, k, b)| Ok((normalize_path(p)?, *k, *b)))
        .collect::<Result<_>>()?;
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut seen = BTreeSet::new();
    for (path, _, _) in sorted.iter() {
        if !seen.insert(path.to_ascii_lowercase()) {
            bail!("duplicate pacbin member path (ignoring case): {}", path);
        }
    }

    let mut out = vec![0u8; HEADER_SIZE];
    let mut entries = Vec::with_capacity(sorted.len());
    for (path, kind, payload) in sorted.iter() {
        pad_to(&mut out, ALIGNMENT);
        entries.push(Member {
            path: path.clone(),
            kind: *kind,
            offset: out.len() as u64,
            length: payload.len() as u64,
            sha256: Sha256::digest(payload),
        });
        out.extend_from_slice(payload);
    }
    pad_to(&mut out, ALIGNMENT);

    let index_offset = out.len() as u64;
    let count = entries.len() as u64;
    let mut index = Vec::new();
    index.extend_from_slice(INDEX_MAGIC);
    index.extend_from_slice(&VERSION.to_le_bytes());
    index.extend_from_slice(&(INDEX_HEADER_SIZE as u16).to_le_bytes());
    index.extend_from_slice(&0u32.to_le_bytes());
    index.extend_from_slice(&count.to_le_bytes());
    index.extend_from_slice(&0u64.to_le_bytes());
    for m in entries.iter() {
        let start = index.len();
        index.extend_from_slice(&(m.path.len() as u32).to_le_bytes());
        index.extend_from_slice(&m.kind.code().to_le_bytes());
        index.extend_from_slice(&0u16.to_le_bytes());
        index.extend_from_slice(&m.offset.to_le_bytes());
        index.extend_from_slice(&m.length.to_le_bytes());
        index.extend_from_slice(&m.sha256);
        index.extend_from_slice(m.path.as_bytes());
        let record = (index.len() - start) as u64;
        index.resize(index.len() + padding(record, INDEX_ALIGNMENT) as usize, 0);
    }
    let index_sha256 = Sha256::digest(&index);
    out.extend_from_slice(&index);

    out.extend_from_slice(FOOTER_MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(FOOTER_SIZE as u16).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&index_offset.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&index_sha256);

    let header = &mut out[..HEADER_SIZE];
    header[0..8].copy_from_slice(HEADER_MAGIC);
    header[8..10].copy_from_slice(&VERSION.to_le_bytes());
    header[10..12].copy_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
    header[16..20].copy_from_slice(&(ALIGNMENT as u32).to_le_bytes());
    header[24..32].copy_from_slice(&index_offset.to_le_bytes());
    header[32..40].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

/// pyAmpliCol's normalization of a member path: relative, ASCII, `/` separators, no NUL or
/// `..`; empty and `.` components dropped.
pub fn normalize_path(path: &str) -> Result<String> {
    if path.is_empty() || path.contains('\0') {
        bail!("pacbin member path must be non-empty and contain no NUL");
    }
    if !path.is_ascii() {
        bail!("pacbin member path must be ASCII: {:?}", path);
    }
    if path.contains('\\') {
        bail!("pacbin member path must use `/` separators: {:?}", path);
    }
    if path.starts_with('/') {
        bail!("pacbin member path must be relative: {:?}", path);
    }
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => bail!("pacbin member path must not contain `..`: {:?}", path),
            _ => parts.push(part),
        }
    }
    if parts.is_empty() {
        bail!("pacbin member path must name a member: {:?}", path);
    }
    Ok(parts.join("/"))
}

fn parse(bytes: &[u8], verify_payloads: bool) -> Result<(Vec<Member>, u64, [u8; 32])> {
    if bytes.len() < HEADER_SIZE + INDEX_HEADER_SIZE + FOOTER_SIZE {
        if bytes.len() < 8 || &bytes[..8] != HEADER_MAGIC {
            bail!("not a pacbin container (wrong magic number)");
        }
        bail!("truncated pacbin container");
    }
    let header = &bytes[..HEADER_SIZE];
    if &header[0..8] != HEADER_MAGIC {
        bail!("not a pacbin container (wrong magic number)");
    }
    contract("header", header, HEADER_SIZE)?;
    let alignment = u32_at(header, 16);
    if alignment as u64 != ALIGNMENT {
        bail!("unsupported pacbin payload alignment: {}", alignment);
    }
    if u32_at(header, 20) != 0 || header[40..64].iter().any(|b| *b != 0) {
        bail!("pacbin header reserved fields must be zero");
    }
    let index_offset = u64_at(header, 24);
    let count = u64_at(header, 32);
    if !index_offset.is_multiple_of(ALIGNMENT) {
        bail!("pacbin index offset is not payload-aligned");
    }

    let footer_offset = (bytes.len() - FOOTER_SIZE) as u64;
    if index_offset < HEADER_SIZE as u64 || index_offset >= footer_offset {
        bail!("pacbin index offset is out of bounds");
    }
    let footer = &bytes[footer_offset as usize..];
    if &footer[0..8] != FOOTER_MAGIC {
        bail!("invalid pacbin footer magic");
    }
    contract("footer", footer, FOOTER_SIZE)?;
    if u64_at(footer, 16) != index_offset {
        bail!("pacbin footer index offset disagrees with the header");
    }
    if u64_at(footer, 24) != count {
        bail!("pacbin footer member count disagrees with the header");
    }
    let index_sha256: [u8; 32] = footer[32..64].try_into().unwrap();

    let index = &bytes[index_offset as usize..footer_offset as usize];
    if count > index.len() as u64 / (ENTRY_SIZE as u64 + INDEX_ALIGNMENT) {
        bail!("pacbin member count cannot fit in the index");
    }
    let index_header = slice(index, 0, INDEX_HEADER_SIZE, "pacbin index header")?;
    if &index_header[0..8] != INDEX_MAGIC {
        bail!("invalid pacbin index magic");
    }
    contract("index", index_header, INDEX_HEADER_SIZE)?;
    if u64_at(index_header, 24) != 0 {
        bail!("pacbin index reserved field must be zero");
    }
    if u64_at(index_header, 16) != count {
        bail!("pacbin index member count disagrees with the header");
    }

    let mut members = Vec::with_capacity(count as usize);
    let mut folded = BTreeSet::new();
    let mut cursor = INDEX_HEADER_SIZE;
    for _ in 0..count {
        let entry = slice(index, cursor, ENTRY_SIZE, "pacbin index entry")?;
        let path_length = u32_at(entry, 0) as usize;
        let code = u16_at(entry, 4);
        let kind =
            Kind::from_code(code).ok_or_else(|| anyhow!("unknown pacbin member kind: {}", code))?;
        let flags = u16_at(entry, 6);
        if flags != 0 {
            bail!("unknown pacbin member flags: {}", flags);
        }
        let offset = u64_at(entry, 8);
        let length = u64_at(entry, 16);
        let sha256: [u8; 32] = entry[24..56].try_into().unwrap();
        cursor += ENTRY_SIZE;

        let path_bytes = slice(index, cursor, path_length, "pacbin member path")?;
        let path = std::str::from_utf8(path_bytes)
            .map_err(|_| anyhow!("pacbin member path is not valid UTF-8"))?;
        if normalize_path(path)? != path {
            bail!("pacbin member path is not canonical: {:?}", path);
        }
        if let Some(prev) = members.last().map(|m: &Member| m.path.as_bytes()) {
            if path_bytes <= prev {
                bail!("pacbin index paths are not strictly sorted");
            }
        }
        if !folded.insert(path.to_ascii_lowercase()) {
            bail!("case-colliding pacbin member path: {}", path);
        }
        cursor += path_length;
        let pad = padding((ENTRY_SIZE + path_length) as u64, INDEX_ALIGNMENT) as usize;
        if slice(index, cursor, pad, "pacbin index padding")?
            .iter()
            .any(|b| *b != 0)
        {
            bail!("pacbin index padding must be zero");
        }
        cursor += pad;
        members.push(Member {
            path: path.to_string(),
            kind,
            offset,
            length,
            sha256,
        });
    }
    if cursor != index.len() {
        bail!("pacbin index has trailing or missing bytes");
    }
    if Sha256::digest(index) != index_sha256 {
        bail!("pacbin index digest mismatch");
    }

    // the payloads: in index order, each at the next 64-byte boundary, zero padding between
    let mut expected = HEADER_SIZE as u64;
    for m in members.iter() {
        let end = m
            .offset
            .checked_add(m.length)
            .ok_or_else(|| anyhow!("pacbin member range overflows: {}", m.path))?;
        let pad = padding(expected, ALIGNMENT);
        if m.offset != expected + pad {
            bail!(
                "{} pacbin member payload: {}",
                if m.offset < expected + pad {
                    "overlapping"
                } else {
                    "non-canonical gap before the"
                },
                m.path
            );
        }
        if end > index_offset {
            bail!("pacbin member payload is out of bounds: {}", m.path);
        }
        zeros(bytes, expected, pad)?;
        expected = end;
    }
    let pad = padding(expected, ALIGNMENT);
    if expected + pad != index_offset {
        bail!("pacbin payload region does not end at the index");
    }
    zeros(bytes, expected, pad)?;

    if verify_payloads {
        for m in members.iter() {
            let payload = &bytes[m.offset as usize..(m.offset + m.length) as usize];
            if Sha256::digest(payload) != m.sha256 {
                bail!("pacbin member digest mismatch: {}", m.path);
            }
        }
    }

    Ok((members, index_offset, index_sha256))
}

fn contract(label: &str, bytes: &[u8], size: usize) -> Result<()> {
    let version = u16_at(bytes, 8);
    if version != VERSION {
        bail!("unsupported pacbin {} version: {}", label, version);
    }
    if u16_at(bytes, 10) as usize != size {
        bail!("unsupported pacbin {} size: {}", label, u16_at(bytes, 10));
    }
    if u32_at(bytes, 12) != 0 {
        bail!("unknown pacbin {} flags: {}", label, u32_at(bytes, 12));
    }
    Ok(())
}

fn zeros(bytes: &[u8], offset: u64, length: u64) -> Result<()> {
    if bytes[offset as usize..(offset + length) as usize]
        .iter()
        .any(|b| *b != 0)
    {
        bail!("pacbin payload padding must be zero");
    }
    Ok(())
}

fn slice<'a>(bytes: &'a [u8], offset: usize, length: usize, label: &str) -> Result<&'a [u8]> {
    offset
        .checked_add(length)
        .and_then(|end| bytes.get(offset..end))
        .ok_or_else(|| anyhow!("truncated {}", label))
}

fn padding(position: u64, alignment: u64) -> u64 {
    (alignment - position % alignment) % alignment
}

fn pad_to(out: &mut Vec<u8>, alignment: u64) {
    out.resize(out.len() + padding(out.len() as u64, alignment) as usize, 0);
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let a = vec![1u8; 100];
        let b = b"abc".to_vec();
        let c: Vec<u8> = Vec::new();
        write(&[
            ("z/b.symjit".into(), Kind::SymjitApplication, &b[..]),
            (
                "./a//x.evaluator.bin".into(),
                Kind::SymbolicaExactState,
                &a[..],
            ),
            ("z/a.plane.symjit".into(), Kind::SymjitApplication, &c[..]),
        ])
        .unwrap()
    }

    #[test]
    fn round_trip() {
        let bytes = sample();
        let p = Pacbin::read(bytes.clone(), true).unwrap();
        let paths: Vec<_> = p.members.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(
            paths,
            ["a/x.evaluator.bin", "z/a.plane.symjit", "z/b.symjit"]
        );
        assert_eq!(p.payload(p.member("z/b.symjit").unwrap()), b"abc");
        assert_eq!(p.payload(&p.members[1]), b"");
        assert_eq!(p.members[0].offset, 64);
        // the writer is deterministic: the members read back give the same bytes
        assert_eq!(write(&p.contents()).unwrap(), bytes);
    }

    #[test]
    fn rejects_damage() {
        let good = sample();
        let p = Pacbin::read(good.clone(), true).unwrap();

        let mut bad = good.clone();
        bad[p.members[0].offset as usize] ^= 1;
        let err = Pacbin::read(bad.clone(), true).unwrap_err().to_string();
        assert!(err.contains("member digest mismatch"), "{}", err);
        // the payload digests are only checked on request
        assert!(Pacbin::read(bad, false).is_ok());

        let mut bad = good.clone();
        bad[p.members[0].offset as usize + 100] = 7; // padding after the first payload
        let err = Pacbin::read(bad, true).unwrap_err().to_string();
        assert!(err.contains("padding must be zero"), "{}", err);

        let mut bad = good.clone();
        bad[p.index_offset as usize + INDEX_HEADER_SIZE + 4] = 6; // the first kind: unassigned
        let err = Pacbin::read(bad, true).unwrap_err().to_string();
        assert!(err.contains("unknown pacbin member kind: 6"), "{}", err);

        let mut bad = good.clone();
        let last = bad.len() - 1;
        bad[last] ^= 1; // the index digest
        let err = Pacbin::read(bad, true).unwrap_err().to_string();
        assert!(err.contains("index digest mismatch"), "{}", err);

        let err = Pacbin::read(good[..100].to_vec(), true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "{}", err);
        let err = Pacbin::read(b"\x7fELF".repeat(50), true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("wrong magic"), "{}", err);
    }

    #[test]
    fn paths() {
        assert_eq!(normalize_path("a/./b//c").unwrap(), "a/b/c");
        for bad in ["", "/a", "a/../b", "a\\b", "é", "./", "a\0"] {
            assert!(normalize_path(bad).is_err(), "{:?}", bad);
        }
        let x = b"x".to_vec();
        assert!(write(&[
            ("A/b".into(), Kind::SymjitApplication, &x[..]),
            ("a/B".into(), Kind::SymjitApplication, &x[..]),
        ])
        .is_err());
    }

    #[test]
    fn kinds() {
        for k in Kind::ALL {
            assert_eq!(Kind::parse(k.name()), Some(k));
            assert_eq!(Kind::parse(&k.code().to_string()), Some(k));
        }
        assert_eq!(Kind::from_code(6), None);
        assert_eq!(Kind::parse("kernel"), None);
    }
}
