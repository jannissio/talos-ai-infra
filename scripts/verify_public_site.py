"""Verify an evidence-page export and, optionally, its unauthenticated HTTP bytes."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_public_site import ASSETS, ROOT
from simulation_lab.storage import require_space


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        self.references.extend(attrs[key] for key in ('src', 'href', 'poster') if attrs.get(key))


def verify(build, base_url=None):
    manifest = json.loads((build / 'build-manifest.json').read_text(encoding='utf-8'))
    sources = {name: ROOT / 'site' / name for name in ('index.html', 'style.css', 'app.js')}
    sources.update({'assets/' + name: ROOT / source for name, source in ASSETS.items()})
    if set(manifest) != set(sources):
        raise ValueError('The export does not match the explicit asset allowlist.')
    allowed = set(manifest) | {'build-manifest.json', '.nojekyll'}
    actual = {p.relative_to(build).as_posix() for p in build.rglob('*') if p.is_file()}
    if actual != allowed:
        raise ValueError('Unexpected or missing export files.')
    references = References()
    references.feed((build / 'index.html').read_text(encoding='utf-8'))
    references.references += re.findall(r"src:\s*'([^']+)'", (build / 'app.js').read_text(encoding='utf-8'))
    local = set()
    for ref in references.references:
        parsed = urlsplit(ref)
        if parsed.scheme or parsed.netloc:
            continue
        if not parsed.path:
            if parsed.fragment and parsed.fragment not in references.ids:
                raise ValueError('Missing in-page anchor: ' + ref)
            continue
        path = (build / unquote(parsed.path)).resolve()
        if not path.is_relative_to(build) or not path.is_file():
            raise ValueError('Missing or escaping local resource: ' + ref)
        local.add(path.relative_to(build).as_posix())
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError('Use an ordinary HTTP(S) base URL without credentials or query parameters.')
    rows = []
    for name, source in sources.items():
        payload = (build / name).read_bytes()
        if sha(payload) != manifest[name] or sha(source.read_bytes()) != manifest[name]:
            raise ValueError('Export differs from its manifest or source: ' + name)
        row = {'file': name, 'sha256': manifest[name], 'bytes': len(payload)}
        if base_url:
            with urlopen(urljoin(base_url.rstrip('/') + '/', name), timeout=15) as response:
                delivered = response.read(len(payload) + 1)
                if response.status != 200 or sha(delivered) != manifest[name]:
                    raise ValueError('HTTP delivery differs from the export: ' + name)
                row['http_status'] = response.status
                row['content_type'] = response.headers.get_content_type()
        rows.append(row)
    return {'schema': 'talos.evidence-site-verification.v1', 'passed': True,
            'files': rows, 'bytes': sum(r['bytes'] for r in rows), 'local_resource_references': sorted(local),
            'base_url': base_url, 'authentication_used': False,
            'scope': 'Static resources, source hashes and local anchors only. This does not verify robot physics, public visibility or the submission form.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--base-url')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Preserve previous verification reports; choose a new output.')
    report = verify(args.build.resolve(), args.base_url)
    if args.output:
        require_space(args.output, 1024**2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'local_resource_references')}, indent=2))
