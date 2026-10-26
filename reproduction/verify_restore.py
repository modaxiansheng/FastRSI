"""Verify pinned backup downloads and optionally extract archives. Never runs an experiment."""
import argparse,hashlib,json,os,shutil,sys,tarfile,tempfile
from pathlib import Path

def verify_file(root,row):
 rel=Path(row['path'])
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
 file=root/rel
 if not file.is_file() or file.stat().st_size!=row['bytes']:raise ValueError('Missing or wrong size: '+str(file))
 with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 if digest!=row['sha256']:raise ValueError('SHA256 mismatch: '+str(file))

def extract_archive(archive,destination):
 if sys.version_info<(3,12):raise RuntimeError('Python 3.12+ required for extraction')
 if destination.exists():raise FileExistsError('Use a new extraction directory: '+str(destination))
 destination.mkdir(parents=True)
 with tarfile.open(archive) as stream:stream.extractall(destination,filter='data')

def restore_modes(root,modes):
 for rel,mode in modes.items():
  path=Path(rel)
  if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe permission path')
  file=root/path
  if not file.is_file() or file.is_symlink():raise ValueError('Missing regular runtime file: '+str(file))
  if file.stat().st_mode & 0o777==mode:continue
  if file.stat().st_nlink>1:
   with tempfile.NamedTemporaryFile(dir=file.parent,delete=False) as tmp:
    with file.open('rb') as source:shutil.copyfileobj(source,tmp)
    name=tmp.name
   os.replace(name,file)
  file.chmod(mode & 0o777)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--workspace',type=Path,required=True)
 parser.add_argument('--scope',choices=['all','artifacts','models'],default='all')
 parser.add_argument('--extract',action='store_true',help='Extract all five verified archives into new unpacked/<label> directories.')
 args=parser.parse_args();root=args.workspace.resolve();record=json.loads(Path(__file__).with_name('BACKUP_RECORD.json').read_text())
 if args.extract and args.scope=='models':parser.error('--extract requires artifacts or all')
 count=0
 if args.scope in {'all','artifacts'}:
  for r in record['artifact_files']:
   verify_file(root/'artifacts',r);count+=1;print('OK artifact',r['path'],flush=True)
 if args.scope in {'all','models'}:
  manifest=root/'models/FINAL_WEIGHT_MANIFEST.json'
  with manifest.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['model_manifest_sha256'],'Model manifest identity mismatch'
  for r in json.loads(manifest.read_text())['files']:
   verify_file(root/'models',{'path':r['repo_path'],'bytes':r['bytes'],'sha256':r['sha256']});count+=1;print('OK model',r['repo_path'],flush=True)
 if args.extract:
  if sys.version_info<(3,12):raise RuntimeError('Python 3.12+ required for extraction')
  destinations=[root/'unpacked'/r['unpack_directory'] for r in record['archives']]
  if any(p.exists() for p in destinations):raise FileExistsError('At least one output directory exists; choose a new workspace or move previous outputs aside.')
  required=record['statistics']['archive_unique_content_bytes_sum']+5_000_000_000
  if shutil.disk_usage(root).free<required:raise RuntimeError('Insufficient free space for extraction and metadata overhead')
  for r,dest in zip(record['archives'],destinations):
   extract_archive(root/'artifacts'/r['file'],dest);print('EXTRACTED',r['unpack_directory'],flush=True)
  restore_modes(root/'unpacked/runtime',json.loads((root/'artifacts/RUNTIME_FILE_MODES.json').read_text()))
 print('PASS:',count,'payload files verified. This is not a GPU/runtime readiness or reproduction result.')
if __name__=='__main__':main()
