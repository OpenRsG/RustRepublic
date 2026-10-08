use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const ANIMATION: u32 = 262342271;
pub const SKELETON: u32 = 615435132;
pub const MESH: u32 = 1096652136;
pub const BUILD_TABLE: u32 = 585940579;
const MAX_ENTRIES: u64 = 2_000_000;

#[derive(Debug)]
pub struct ForgeEntry {
    pub id: u64,
    pub offset: u64,
    pub size: u64,
    pub kind: u32,
    pub name: String,
}

#[derive(Debug)]
pub struct ForgeArchive {
    pub version: u32,
    pub entries: Vec<ForgeEntry>,
}

pub(crate) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn range(offset: u64, length: u64, file_size: u64) -> io::Result<()> {
    if offset.checked_add(length).is_none_or(|end| end > file_size) {
        return Err(invalid("Forge range outside file"));
    }
    Ok(())
}

fn at<const N: usize>(f: &mut (impl Read + Seek), offset: u64, size: u64) -> io::Result<[u8; N]> {
    range(offset, N as u64, size)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0; N];
    f.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

fn u64le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

fn position(bytes: &[u8]) -> io::Result<u64> {
    let value = i64::from_le_bytes(bytes[..8].try_into().unwrap());
    u64::try_from(value).map_err(|_| invalid("Negative Forge offset"))
}

impl ForgeArchive {
    pub fn open(path: &Path) -> io::Result<Self> {
        let f = File::open(path)?;
        let size = f.metadata()?.len();
        Self::read(&mut BufReader::new(f), size)
    }

    fn read(f: &mut (impl Read + Seek), size: u64) -> io::Result<Self> {
        let header = at::<21>(f, 0, size)?;
        if &header[..9] != b"scimitar\0" {
            return Err(invalid("Missing scimitar Forge signature"));
        }
        let version = u32le(&header[9..]);
        if version != 27 {
            return Err(invalid(format!(
                "Unsupported Forge version {version}; expected 27"
            )));
        }
        let header_size = position(&header[13..])?;
        if header_size < 21 {
            return Err(invalid("Forge header overlaps signature"));
        }
        let tables = at::<44>(f, header_size, size)?;
        let total = u32le(&tables) as u64;
        let sets = u32le(&tables[32..]) as u64;
        if total > MAX_ENTRIES || sets > 10_000 || (total > 0 && sets == 0) {
            return Err(invalid("Invalid or excessive Forge entry/fileset count"));
        }
        let mut next = i64::from_le_bytes(tables[36..44].try_into().unwrap());
        let mut visited = HashSet::new();
        let mut entries = Vec::with_capacity(total as usize);
        for _ in 0..sets {
            let pos =
                u64::try_from(next).map_err(|_| invalid("Premature Forge fileset terminator"))?;
            if !visited.insert(pos) {
                return Err(invalid("Cyclic Forge fileset chain"));
            }
            let set = at::<40>(f, pos, size)?;
            let count = u32le(&set) as u64;
            if count > total - entries.len() as u64 {
                return Err(invalid("Forge fileset exceeds declared total"));
            }
            let offsets = position(&set[8..])?;
            next = i64::from_le_bytes(set[16..24].try_into().unwrap());
            let info = position(&set[32..])?;
            range(offsets, count * 20, size)?;
            range(info, count * 192, size)?;
            f.seek(SeekFrom::Start(offsets))?;
            let mut records = vec![0; count as usize * 20];
            f.read_exact(&mut records)?;
            f.seek(SeekFrom::Start(info))?;
            for record in records.chunks_exact(20) {
                let offset = position(record)?;
                let length = i32::from_le_bytes(record[16..20].try_into().unwrap());
                let length =
                    u64::try_from(length).map_err(|_| invalid("Negative Forge entry length"))?;
                range(offset, length, size)?;
                let mut metadata = [0; 192];
                f.read_exact(&mut metadata)?;
                if u32le(&metadata) as u64 != length {
                    return Err(invalid("Forge offset/info lengths disagree"));
                }
                let name = &metadata[44..172];
                let end = name
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(name.len());
                entries.push(ForgeEntry {
                    id: u64le(&record[8..]),
                    offset,
                    size: length,
                    kind: u32le(&metadata[16..]),
                    name: name[..end].iter().map(|&byte| char::from(byte)).collect(),
                });
            }
        }
        if next != -1 || entries.len() as u64 != total {
            return Err(invalid("Forge fileset chain/count mismatch"));
        }
        Ok(Self { version, entries })
    }
}

pub fn read_entry(path: &Path, entry: &ForgeEntry) -> io::Result<Vec<u8>> {
    if entry.size > crate::data::MAX_DECODED as u64 {
        return Err(invalid("Entry exceeds 256 MiB extraction limit"));
    }
    let mut f = File::open(path)?;
    range(entry.offset, entry.size, f.metadata()?.len())?;
    f.seek(SeekFrom::Start(entry.offset))?;
    let mut bytes = vec![0; entry.size as usize];
    f.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub fn kind_name(kind: u32) -> &'static str {
    match kind {
        ANIMATION => "Animation",
        SKELETON => "Skeleton",
        MESH => "Mesh",
        BUILD_TABLE => "BuildTable",
        159662430 => "Entity",
        2729961751 => "TextureMap",
        2244483011 => "Material",
        1477804522 => "AirTricksSettings",
        2183262095 => "DBCollisionReactionGameplay",
        0 => "Sidecar",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture() -> Vec<u8> {
        let mut bytes = vec![0; 322];
        bytes[..9].copy_from_slice(b"scimitar\0");
        bytes[9..13].copy_from_slice(&27u32.to_le_bytes());
        bytes[13..21].copy_from_slice(&21u64.to_le_bytes());
        bytes[21..25].copy_from_slice(&1u32.to_le_bytes());
        bytes[53..57].copy_from_slice(&1u32.to_le_bytes());
        bytes[57..65].copy_from_slice(&65u64.to_le_bytes());
        bytes[65..69].copy_from_slice(&1u32.to_le_bytes());
        bytes[73..81].copy_from_slice(&105u64.to_le_bytes());
        bytes[81..89].copy_from_slice(&(-1i64).to_le_bytes());
        bytes[97..105].copy_from_slice(&125u64.to_le_bytes());
        bytes[105..113].copy_from_slice(&317u64.to_le_bytes());
        bytes[113..121].copy_from_slice(&42u64.to_le_bytes());
        bytes[121..125].copy_from_slice(&5u32.to_le_bytes());
        bytes[125..129].copy_from_slice(&5u32.to_le_bytes());
        bytes[141..145].copy_from_slice(&ANIMATION.to_le_bytes());
        bytes[169..174].copy_from_slice(b"Pedal");
        bytes[317..].copy_from_slice(b"asset");
        bytes
    }

    #[test]
    fn reads_identity_and_rejects_invalid_ranges_and_chains() {
        let bytes = fixture();
        let archive = ForgeArchive::read(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        let entry = &archive.entries[0];
        assert_eq!(
            (
                entry.id,
                entry.offset,
                entry.size,
                entry.kind,
                entry.name.as_str()
            ),
            (42, 317, 5, ANIMATION, "Pedal")
        );
        for (offset, replacement) in [
            (105, u64::MAX.to_le_bytes().to_vec()),
            (121, 6u32.to_le_bytes().to_vec()),
            (81, 65u64.to_le_bytes().to_vec()),
            (21, u32::MAX.to_le_bytes().to_vec()),
        ] {
            let mut bad = bytes.clone();
            bad[offset..offset + replacement.len()].copy_from_slice(&replacement);
            assert!(ForgeArchive::read(&mut Cursor::new(&bad), bad.len() as u64).is_err());
        }
        for end in [0, 20, 64, 104, 124, 316, 321] {
            assert!(ForgeArchive::read(&mut Cursor::new(&bytes[..end]), end as u64).is_err());
        }
    }
}
