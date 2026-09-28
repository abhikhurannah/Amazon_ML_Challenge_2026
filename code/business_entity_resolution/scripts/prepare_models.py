"""Extract only regular model files within the local work directory."""
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=root/'work_hanuman_dev';dest.mkdir(exist_ok=True)
with tarfile.open(root/'artifacts/hanuman-Dev_models.tar.gz') as archive:
    for member in archive.getmembers():
        target=(dest/member.name).resolve()
        if not target.is_relative_to(dest.resolve()) or not (member.isdir() or member.isfile()):
            raise ValueError('Unsafe model archive entry')
    archive.extractall(dest,filter='data')
print('Models prepared:',dest)
