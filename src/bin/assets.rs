#[path = "../data.rs"]
mod data;
#[path = "../forge.rs"]
mod forge;

use data::DataContainer;
use forge::{ForgeArchive, ForgeEntry, invalid};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn game_root(explicit: Option<&str>) -> io::Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("RIDERS_REPUBLIC_DIR") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME")
        .ok_or_else(|| invalid("Set RIDERS_REPUBLIC_DIR or supply the game directory"))?;
    Ok(PathBuf::from(home).join(
        ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/RidersRepublic",
    ))
}

fn archives(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for file in fs::read_dir(root)? {
        let file = file?;
        let path = file.path();
        if file.file_type()?.is_file() && path.extension().is_some_and(|ext| ext == "forge") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(invalid(format!(
            "No Forge archives found in {}",
            root.display()
        )));
    }
    Ok(paths)
}

fn matches(entry: &ForgeEntry, term: &str) -> bool {
    term.is_empty()
        || entry
            .name
            .as_bytes()
            .windows(term.len())
            .any(|bytes| bytes.eq_ignore_ascii_case(term.as_bytes()))
}

fn print_entry(out: &mut impl Write, path: &Path, entry: &ForgeEntry) -> io::Result<()> {
    writeln!(
        out,
        "{}\t{}\t{:08x}\t{}\t{}\t{:?}",
        path.file_name().unwrap().to_string_lossy(),
        entry.id,
        entry.kind,
        forge::kind_name(entry.kind),
        entry.size,
        entry.name
    )
}

fn inspect(root: &Path, out: &mut impl Write) -> io::Result<()> {
    let paths = archives(root)?;
    let mut total = 0usize;
    writeln!(
        out,
        "archive\tversion\tentries\tanimations\tskeletons\tmeshes\tbike_name_matches"
    )?;
    for path in &paths {
        let archive =
            ForgeArchive::open(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let (mut animations, mut skeletons, mut meshes, mut bikes) = (0, 0, 0, 0);
        for entry in &archive.entries {
            match entry.kind {
                forge::ANIMATION => animations += 1,
                forge::SKELETON => skeletons += 1,
                forge::MESH => meshes += 1,
                _ => (),
            }
            if matches(entry, "bike") || matches(entry, "bmx") || matches(entry, "pedal") {
                bikes += 1;
            }
        }
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            path.file_name().unwrap().to_string_lossy(),
            archive.version,
            archive.entries.len(),
            animations,
            skeletons,
            meshes,
            bikes
        )?;
        total += archive.entries.len();
    }
    writeln!(
        out,
        "Loaded {} archive indexes, {total} entries (not a playable bike port).",
        paths.len()
    )
}

fn search(root: &Path, term: &str, check: bool, out: &mut impl Write) -> io::Result<()> {
    if term.is_empty() {
        return Err(invalid("Search term must not be empty"));
    }
    let mut found = 0;
    let mut failed = 0;
    writeln!(out, "archive\tid\ttype_id\ttype\tstored_bytes\tname")?;
    for path in archives(root)? {
        let archive = ForgeArchive::open(&path)?;
        for entry in archive.entries.iter().filter(|e| matches(e, term)) {
            found += 1;
            if check {
                let decoded =
                    forge::read_entry(&path, entry).and_then(|bytes| DataContainer::decode(&bytes));
                match decoded {
                    Ok(data) => {
                        if !data
                            .resources
                            .iter()
                            .any(|r| r.id == entry.id && r.kind == entry.kind)
                        {
                            failed += 1;
                            writeln!(
                                out,
                                "FAIL\t{}\t{}\tOuter resource identity missing",
                                path.display(),
                                entry.id
                            )?;
                            continue;
                        }
                        print_entry(out, &path, entry)?;
                        writeln!(
                            out,
                            "OK\t{} resources\t{} decoded bytes",
                            data.resources.len(),
                            data.files.len()
                        )?;
                    }
                    Err(e) => {
                        failed += 1;
                        writeln!(out, "FAIL\t{}\t{}\t{e}", path.display(), entry.id)?;
                    }
                }
            } else {
                print_entry(out, &path, entry)?;
            }
        }
    }
    writeln!(out, "Matched {found} entries; {failed} decode failures.")?;
    if found == 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No matching resources",
        ));
    }
    if failed > 0 {
        return Err(invalid(format!("{failed} resources could not be decoded")));
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)
}

fn extract(path: &Path, id: u64, out: &mut impl Write) -> io::Result<()> {
    let archive = ForgeArchive::open(path)?;
    let mut matching = archive.entries.iter().filter(|e| e.id == id);
    let entry = matching
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "ID absent from archive"))?;
    if matching.next().is_some() {
        return Err(invalid("Ambiguous duplicate ID in archive"));
    }
    let bytes = forge::read_entry(path, entry)?;
    let data = DataContainer::decode(&bytes)?;
    if !data
        .resources
        .iter()
        .any(|r| r.id == entry.id && r.kind == entry.kind)
    {
        return Err(invalid("Outer resource identity missing"));
    }
    let dir = PathBuf::from(".local/extracted").join(format!("{id:016x}"));
    fs::create_dir_all(dir.parent().unwrap())?;
    fs::create_dir(&dir)?;
    write_new(&dir.join("container.data"), &bytes)?;
    write_new(&dir.join("metadata.bin"), &data.metadata)?;
    write_new(&dir.join("files.bin"), &data.files)?;
    for (index, resource) in data.resources.iter().enumerate() {
        let base = format!("{index:04}_{:016x}_{:08x}", resource.id, resource.kind);
        write_new(
            &dir.join(format!("{base}.payload")),
            &data.files[resource.payload.clone()],
        )?;
        write_new(
            &dir.join(format!("{base}.header")),
            &data.files[resource.header.clone()],
        )?;
        write_new(
            &dir.join(format!("{base}.record")),
            &data.files[resource.record.clone()],
        )?;
        writeln!(
            out,
            "{}\t{}\t{}\t{:?}",
            resource.id,
            forge::kind_name(resource.kind),
            resource.payload.len(),
            resource.name
        )?;
    }
    writeln!(
        out,
        "Extracted to {}. Payloads retain native schemas; no keyframe/physics decoding is claimed.",
        dir.display()
    )
}

fn run(out: &mut impl Write) -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str).unwrap_or("inspect") {
        "inspect" if args.len() <= 2 => inspect(&game_root(args.get(1).map(String::as_str))?, out),
        "find" | "check" if (2..=3).contains(&args.len()) => search(
            &game_root(args.get(2).map(String::as_str))?,
            &args[1],
            args[0] == "check",
            out,
        ),
        "extract" if args.len() == 3 => {
            let id = if let Some(hex) = args[2].strip_prefix("0x") {
                u64::from_str_radix(hex, 16)
            } else {
                args[2].parse()
            }
            .map_err(|_| invalid("ID must be a decimal u64 or 0x-prefixed hexadecimal"))?;
            extract(Path::new(&args[1]), id, out)
        }
        "help" | "--help" | "-h" => writeln!(
            out,
            "Riders Republic read-only asset loader\n\n  inspect [GAME_DIR]\n  find TERM [GAME_DIR]\n  check TERM [GAME_DIR]       Decode and validate every match\n  extract ARCHIVE ID         Extract to ignored .local/extracted/\n\nGAME_DIR defaults to RIDERS_REPUBLIC_DIR, then Flatpak Steam under HOME.\nGamePort2Rust native analysis: bash scripts/gameport.sh query BINARY PROCEDURE\nThis loads actual retail resources; it does not implement playable bikes."
        ),
        _ => Err(invalid("Invalid command/arguments; run with --help")),
    }
}

fn main() {
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    if let Err(e) = run(&mut out).and_then(|()| out.flush()) {
        if e.kind() == io::ErrorKind::BrokenPipe {
            return;
        }
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
