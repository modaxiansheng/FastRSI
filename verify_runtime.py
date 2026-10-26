"""Verify the supplemental runtime payload; optionally extract with Python 3.12+."""
import argparse,hashlib,json,sys,tarfile,shutil,tempfile,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',default='.');p.add_argument('--extract');a=p.parse_args()
root=Path(a.directory);receipt=json.loads((root/'FINAL_RUNTIME_RECEIPT.json').read_text())
metadata=json.loads((root/'RUNTIME_METADATA_RECEIPT.json').read_text())
for r in receipt['files']+metadata['files']:
 f=root/r['path'];assert f.is_file() and f.stat().st_size==r['bytes'],r['path']
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==r['sha256'],r['path']
 print('OK',r['path'])
if a.extract:
 if sys.version_info<(3,12):raise SystemExit('Python 3.12+ required for safe extraction')
 destination=Path(a.extract)
 if destination.exists() and any(destination.iterdir()):raise SystemExit('Choose an empty output directory')
 destination.mkdir(parents=True,exist_ok=True)
 with tarfile.open(root/'final-runtime-dependencies-20261026.tar.gz') as archive:archive.extractall(destination,filter='data')
 modes=json.loads((root/'RUNTIME_FILE_MODES.json').read_text())
 for relative,mode in modes.items():
  relative_path=Path(relative)
  if relative_path.is_absolute() or '..' in relative_path.parts:raise ValueError('Unsafe mode path')
  file=destination/relative_path
  if not file.is_file():raise FileNotFoundError(file)
  if file.stat().st_nlink>1:
   with tempfile.NamedTemporaryFile(dir=file.parent,delete=False) as temp:
    with file.open('rb') as source:shutil.copyfileobj(source,temp)
    temporary=Path(temp.name)
   os.replace(temporary,file)
  file.chmod(mode & 0o777)
 print('Extracted to',destination.resolve())
