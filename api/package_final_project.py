"""Package the course submission from versioned sources and the verified final release."""
import argparse
import hashlib
import subprocess
import tempfile
import zipfile
from pathlib import Path

from verify_final_project import FINAL, ROOT, verify


def package(destination):
    verify()
    files = subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode("utf-8").split("\0")
    # Ignore local runtime data, external datasets and the instructor's general course handout.
    files = {p for p in files if p and p.split("/")[0] in {"Baseline","Explore","api","local_model","results",
             "README.md","Project5_Implementation_Plan_zh.md","Project5_API_Plan_zh.md","environment.yml",
             "requirements-api.txt",".gitattributes",".gitignore"}}
    files.update(p.relative_to(ROOT).as_posix() for p in FINAL.rglob("*") if p.is_file() and p.suffix.lower() in
                 {".md",".json",".csv",".svg",".pdf",".html"})
    files.update("api/"+n for n in ("build_final_evidence.py","final_course_report.py","finish_local_searchqa.py",
                                    "package_final_project.py","render_final_project.py","verify_final_project.py"))
    blocked = {".env","__pycache__","runtime","models","runs","downloads",".git"}
    suffixes = {".gguf",".safetensors",".pt",".pth",".bin",".dll",".exe",".parquet",".zip",".pem",".key",".log",".pyc"}
    for name in files:
        p=Path(name)
        if blocked.intersection(part.lower() for part in p.parts) or any(part.lower().startswith(".env") for part in p.parts) or p.suffix.lower() in suffixes:
            raise ValueError("Private or runtime file in submission: "+name)
    destination.parent.mkdir(parents=True,exist_ok=True)
    start=b'<!doctype html><html lang="en"><meta charset="utf-8"><title>Project 5</title><meta http-equiv="refresh" content="0;url=results/final_project/index.html"><a href="results/final_project/index.html">Open the Project 5 results, paper and demonstration</a></html>'
    with tempfile.NamedTemporaryFile(dir=destination.parent,suffix=".tmp",delete=False) as temp:
        output=Path(temp.name)
    try:
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            checks=[]
            for name in sorted(files | {"START_HERE.html"}):
                raw=start if name=="START_HERE.html" else (ROOT/name).read_bytes()
                archive.writestr(name,raw)
                checks.append(f"{hashlib.sha256(raw).hexdigest()}  {name}")
            archive.writestr("SHA256SUMS.txt","\n".join(checks)+"\n")
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
            for line in archive.read("SHA256SUMS.txt").decode().splitlines():
                checksum,name=line.split("  ",1)
                if hashlib.sha256(archive.read(name)).hexdigest()!=checksum:
                    raise ValueError("Invalid member checksum: "+name)
        output.replace(destination)
    finally:
        output.unlink(missing_ok=True)
    print(f"Packaged {len(files)} project files, an offline entry point and SHA-256 manifest: {destination}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/Project5_submission_20261125.zip")
    package(parser.parse_args().output.resolve())
