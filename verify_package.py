import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for entry in manifest['files']:
    p=root/entry['path'];h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert p.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256'],str(p)
print('PASS:',len(manifest['files']),'files match the manifest')
