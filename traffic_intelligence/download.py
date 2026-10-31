"""Download data only, at an immutable upstream revision; do not run upstream code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import urllib.request

import certifi

from .settings import RAW, TML_REPOSITORY, TML_REVISION

TML_FILES = ['data/csv-traffic-bangalore.csv', 'data/csv-routes-bangalore.csv', 'data/SCHEMA.md']


def get_bytes(url: str, *, timeout=90) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'UrbanTrafficIntelligence/0.1 (research project)'})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def fetch_data(volume_path: str | None = None, revision: str = TML_REVISION) -> dict:
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Use an exact 40-character Git commit SHA for a reproducible snapshot.')
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / 'sources.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if volume_path:
        source = Path(volume_path).expanduser().resolve()
        target = RAW / 'bangalore_volume.csv'
        if source != target.resolve():
            shutil.copyfile(source, target)
        manifest['volume'] = {'file': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                              'source': 'User-supplied Bangalore City Traffic Dataset',
                              'provenance': 'Measurement origin and licence unverified; kept local.'}
    records = []
    for remote_path in TML_FILES:
        url = f'https://raw.githubusercontent.com/{TML_REPOSITORY}/{revision}/{remote_path}'
        payload = get_bytes(url)
        if not payload or payload.startswith(b'<html'):
            raise ValueError(f'Invalid dataset response for {remote_path}')
        name = Path(remote_path).name
        target = RAW / name
        temporary = target.with_suffix(target.suffix + '.part')
        temporary.write_bytes(payload)
        temporary.replace(target)
        records.append({'file': name, 'url': url, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
        print(f'Downloaded {name}: {len(payload):,} bytes', flush=True)
    manifest['journey'] = {'repository': f'https://github.com/{TML_REPOSITORY}', 'revision': revision,
                           'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'licence': 'CC BY 4.0',
                           'attribution': 'Traffic Monitor Lizard by Mahesh Shantaram / thecont1', 'files': records}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
