use crate::forge::invalid;
use std::io;
use std::ops::Range;

pub const MAX_DECODED: usize = 256 * 1024 * 1024;
const MAGIC: u64 = 0x1004fa9957fbaa33;

#[derive(Debug)]
pub struct Resource {
    pub id: u64,
    pub kind: u32,
    pub name: String,
    pub record: Range<usize>,
    pub header: Range<usize>,
    pub payload: Range<usize>,
}

#[derive(Debug)]
pub struct DataContainer {
    pub metadata: Vec<u8>,
    pub files: Vec<u8>,
    pub resources: Vec<Resource>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, size: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(size)
            .ok_or_else(|| invalid("Resource size overflow"))?;
        let result = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| invalid("Truncated resource"))?;
        self.pos = end;
        Ok(result)
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn length(&mut self) -> io::Result<usize> {
        let value = self.u32()?;
        if value > i32::MAX as u32 {
            return Err(invalid("Negative resource length"));
        }
        Ok(value as usize)
    }
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (0u32, 0u32);
    for chunk in bytes.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

fn compressed(reader: &mut Reader<'_>, remaining_budget: usize) -> io::Result<Vec<u8>> {
    if reader.u64()? != MAGIC {
        return Err(invalid(
            "Missing compressed-data signature (sidecar or unsupported entry)",
        ));
    }
    let version = reader.u16()?;
    let algorithm = reader.take(1)?[0];
    reader.take(4)?;
    if version != 2 || algorithm != 5 {
        return Err(invalid(format!("Unsupported data compression version {version}, algorithm {algorithm}; expected Riders v2/Zstandard (5)")));
    }
    let count = reader.length()?;
    if count > remaining_budget / 8 {
        return Err(invalid("Excessive compression block count"));
    }
    let table = reader.take(
        count
            .checked_mul(8)
            .ok_or_else(|| invalid("Block table overflow"))?,
    )?;
    let mut total = 0usize;
    for info in table.chunks_exact(8) {
        let mut sizes = Reader {
            bytes: info,
            pos: 0,
        };
        let raw = sizes.length()?;
        let stored = sizes.length()?;
        if raw == 0 || stored == 0 {
            return Err(invalid("Empty compression block"));
        }
        total = total
            .checked_add(raw)
            .ok_or_else(|| invalid("Decoded length overflow"))?;
        if total > remaining_budget {
            return Err(invalid("Decoded data exceeds 256 MiB budget"));
        }
    }
    let mut output = vec![0; total];
    let mut decoder = zstd::bulk::Decompressor::new()?;
    decoder.set_parameter(zstd::zstd_safe::DParameter::WindowLogMax(23))?;
    let mut offset = 0;
    for info in table.chunks_exact(8) {
        let mut sizes = Reader {
            bytes: info,
            pos: 0,
        };
        let raw = sizes.length()?;
        let stored = sizes.length()?;
        let checksum = reader.u32()?;
        let block = reader.take(stored)?;
        if adler32(block) != checksum {
            return Err(invalid("Compressed block Adler-32 mismatch"));
        }
        let destination = &mut output[offset..offset + raw];
        if raw == stored {
            destination.copy_from_slice(block);
        } else if decoder.decompress_to_buffer(block, destination)? != raw {
            return Err(invalid("Zstandard decoded length mismatch"));
        }
        offset += raw;
    }
    Ok(output)
}

impl DataContainer {
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut reader = Reader { bytes, pos: 0 };
        let metadata = compressed(&mut reader, MAX_DECODED)?;
        let files = compressed(&mut reader, MAX_DECODED - metadata.len())?;
        if reader.pos != bytes.len() {
            return Err(invalid("Trailing compressed container bytes"));
        }
        let mut index = Reader {
            bytes: &metadata,
            pos: 0,
        };
        let count = index.u16()? as usize;
        if metadata.len() != 2 + count * 14 {
            return Err(invalid("Unsupported metadata record layout"));
        }
        let mut body = Reader {
            bytes: &files,
            pos: 0,
        };
        let mut resources = Vec::with_capacity(count);
        for _ in 0..count {
            let expected_id = index.u64()?;
            let expected_size = index.length()?;
            index.take(2)?;
            let record_start = body.pos;
            let kind = body.u32()?;
            let length = body.length()?;
            let name_length = body.length()?;
            let name = body.take(name_length)?;
            let name = name.strip_suffix(&[0]).unwrap_or(name);
            let name = name.iter().map(|&byte| char::from(byte)).collect();
            let header_start = body.pos;
            let flag = body.take(1)?[0];
            match flag {
                0 => (),
                1 => {
                    body.take(3)?;
                    let fields = body.length()?;
                    body.take(
                        fields
                            .checked_mul(12)
                            .ok_or_else(|| invalid("Extended header overflow"))?,
                    )?;
                }
                _ => return Err(invalid(format!("Unsupported file header flag {flag}"))),
            }
            let header_end = body.pos;
            let payload = body.take(length)?;
            if payload.len() < 12 {
                return Err(invalid("Resource payload lacks identity"));
            }
            let id = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let embedded_kind = u32::from_le_bytes(payload[8..12].try_into().unwrap());
            if id != expected_id
                || kind != embedded_kind
                || body.pos - record_start != expected_size
            {
                return Err(invalid(
                    "Resource identity/type/record length disagrees with metadata",
                ));
            }
            resources.push(Resource {
                id,
                kind,
                name,
                record: record_start..body.pos,
                header: header_start..header_end,
                payload: header_end..body.pos,
            });
        }
        if body.pos != files.len() {
            return Err(invalid("Unindexed resource payload bytes"));
        }
        Ok(Self {
            metadata,
            files,
            resources,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(bytes: &[u8], compress: bool) -> Vec<u8> {
        let encoded = if compress {
            zstd::bulk::compress(bytes, 3).unwrap()
        } else {
            bytes.to_vec()
        };
        let mut out = Vec::new();
        out.extend(MAGIC.to_le_bytes());
        out.extend(2u16.to_le_bytes());
        out.push(5);
        out.extend([0; 4]);
        out.extend(1u32.to_le_bytes());
        out.extend((bytes.len() as u32).to_le_bytes());
        out.extend((encoded.len() as u32).to_le_bytes());
        out.extend(adler32(&encoded).to_le_bytes());
        out.extend(encoded);
        out
    }

    fn fixture(compress: bool) -> Vec<u8> {
        let mut body = Vec::new();
        let mut meta = Vec::new();
        meta.extend(2u16.to_le_bytes());
        for (id, kind, name, extended) in [
            (42u64, 262342271u32, b"Pedal".as_slice(), false),
            (43, 615435132, b"BikeRig".as_slice(), true),
        ] {
            let start = body.len();
            body.extend(kind.to_le_bytes());
            body.extend(76u32.to_le_bytes());
            body.extend((name.len() as u32).to_le_bytes());
            body.extend(name);
            if extended {
                body.extend([1, 0, 0, 0]);
                body.extend(1u32.to_le_bytes());
                body.extend([0; 12]);
            } else {
                body.push(0);
            }
            body.extend(id.to_le_bytes());
            body.extend(kind.to_le_bytes());
            body.extend([7; 64]);
            meta.extend(id.to_le_bytes());
            meta.extend(((body.len() - start) as u32).to_le_bytes());
            meta.extend([0; 2]);
        }
        let mut bytes = stored(&meta, false);
        bytes.extend(stored(&body, compress));
        bytes
    }

    #[test]
    fn decodes_multiple_resources_without_losing_extended_header_or_identity() {
        for compress in [false, true] {
            let data = DataContainer::decode(&fixture(compress)).unwrap();
            assert_eq!(
                data.resources
                    .iter()
                    .map(|r| (r.id, r.name.as_str()))
                    .collect::<Vec<_>>(),
                [(42, "Pedal"), (43, "BikeRig")]
            );
            assert_eq!(data.resources[1].header.len(), 20);
            assert_eq!(
                &data.files[data.resources[0].payload.clone()][12..],
                &[7; 64]
            );
        }
    }

    #[test]
    fn rejects_corruption_truncation_and_excessive_decoded_length() {
        let bytes = fixture(true);
        let mut corrupt = bytes.clone();
        corrupt[31] ^= 1;
        assert!(DataContainer::decode(&corrupt)
            .unwrap_err()
            .to_string()
            .contains("Adler"));
        for end in [0, 18, 26, 30, bytes.len() - 1] {
            assert!(DataContainer::decode(&bytes[..end]).is_err());
        }
        let mut bomb = bytes.clone();
        bomb[19..23].copy_from_slice(&(MAX_DECODED as u32 + 1).to_le_bytes());
        assert!(DataContainer::decode(&bomb)
            .unwrap_err()
            .to_string()
            .contains("budget"));
        let mut wrong_identity = fixture(false);
        wrong_identity[33] ^= 1;
        let checksum = adler32(&wrong_identity[31..61]);
        wrong_identity[27..31].copy_from_slice(&checksum.to_le_bytes());
        assert!(DataContainer::decode(&wrong_identity)
            .unwrap_err()
            .to_string()
            .contains("identity"));
    }
}
