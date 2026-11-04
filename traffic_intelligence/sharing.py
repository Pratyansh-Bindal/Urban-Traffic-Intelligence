"""Run a visitor-mode Streamlit instance behind a temporary Cloudflare tunnel."""
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
from urllib.parse import urlencode, urlparse
import urllib.request
import webbrowser

import certifi

from .download import get_bytes
from .settings import ROOT


CLOUDFLARED_VERSION = '2026.10.7'
# Official GitHub release-asset SHA256 digests; downloads are checked before use.
DOWNLOADS = {
    ('Darwin', 'arm64'): ('cloudflared-darwin-arm64.tgz', '3586ccf397e98e5f437e60e7c307ca6ac463e110580503af8bb8e6f9ebf831bd'),
    ('Darwin', 'amd64'): ('cloudflared-darwin-amd64.tgz', '02f3edfd87684dd5b705607688e4eb4b323f6e777b81e7b3a85af556aa356848'),
    ('Linux', 'arm64'): ('cloudflared-linux-arm64', '3df43aedc9ff291d1735b5cb576cb97e0b157488796db2e32b28aecabb445775'),
    ('Linux', 'amd64'): ('cloudflared-linux-amd64', '1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872'),
    ('Windows', 'amd64'): ('cloudflared-windows-amd64.exe', '2493dbc168923bb3e59e5e7716f71deced92992eea9acd6e9d74f84ac7751343'),
}
PUBLIC_URL = re.compile(r'https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com(?=[\s/|]|$)')


class SharingError(RuntimeError):
    """An actionable sharing/startup failure."""


def install_cloudflared(tools_dir, *, system=None, machine=None):
    """Keep only a checksum-verified executable and its integrity checksum."""
    system, machine = system or platform.system(), (machine or platform.machine()).lower()
    machine = {'aarch64': 'arm64', 'x86_64': 'amd64', 'x64': 'amd64'}.get(machine, machine)
    asset = DOWNLOADS.get((system, machine))
    if asset is None:
        raise SharingError('Install cloudflared for this platform, then use share --cloudflared /path/to/cloudflared.')
    name, expected = asset
    directory = Path(tools_dir) / f'cloudflared-{CLOUDFLARED_VERSION}-{system.lower()}-{machine}'
    binary = directory / ('cloudflared.exe' if system == 'Windows' else 'cloudflared')
    checksum = directory / 'binary.sha256'
    if binary.is_file() and checksum.is_file():
        if hashlib.sha256(binary.read_bytes()).hexdigest() == checksum.read_text().strip():
            return binary
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Downloading official cloudflared {CLOUDFLARED_VERSION} (first run only)…', flush=True)
    url = f'https://github.com/cloudflare/cloudflared/releases/download/{CLOUDFLARED_VERSION}/{name}'
    payload = get_bytes(url)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SharingError('Cloudflared download checksum did not match. Nothing was installed; try again.')
    # Read one regular member explicitly; never extract archive paths or links.
    if name.endswith('.tgz'):
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            members = [item for item in archive.getmembers() if item.name in ('cloudflared', './cloudflared')]
            if len(members) != 1 or not members[0].isfile() or members[0].size > 128 * 1024 * 1024:
                raise SharingError('The cloudflared archive does not contain the expected executable.')
            with archive.extractfile(members[0]) as source:
                payload = source.read()
    if not payload:
        raise SharingError('The cloudflared executable is empty.')
    with tempfile.TemporaryDirectory(prefix='install-', dir=directory) as temporary:
        staged = Path(temporary) / binary.name
        staged.write_bytes(payload)
        staged.chmod(0o755)
        staged.replace(binary)
    checksum.write_text(hashlib.sha256(payload).hexdigest() + '\n')
    return binary


def cloudflared_path(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SharingError(f'Cloudflared executable not found: {path}')
        return path
    installed = shutil.which('cloudflared')
    return Path(installed) if installed else install_cloudflared(ROOT / '.tools')


def check_port(port):
    """Never attach a public tunnel to somebody else's existing server."""
    if not 1 <= port <= 65535:
        raise SharingError('Choose a port between 1 and 65535.')
    try:
        with socket.socket() as listener:
            # Match the server's address reuse on POSIX so a recently closed
            # connection in TIME_WAIT does not block an immediate restart.
            option = socket.SO_EXCLUSIVEADDRUSE if os.name == 'nt' else socket.SO_REUSEADDR
            listener.setsockopt(socket.SOL_SOCKET, option, 1)
            listener.bind(('127.0.0.1', port))
    except OSError as error:
        raise SharingError(f'Cannot use port {port}. Stop the existing shared app or run share --port {port + 1 if port < 65535 else 8502}.') from error


def streamlit_command(url, port):
    hostname = urlparse(url).hostname
    command = [sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
            '--server.address', '127.0.0.1', '--server.port', str(port),
            '--server.headless', 'true', '--server.fileWatcherType', 'none',
            '--server.enableCORS', 'true', '--server.enableXsrfProtection', 'true',
            '--server.enableStaticServing', 'false', '--server.baseUrlPath', '',
            '--browser.serverAddress', hostname, '--browser.serverPort', '443',
            '--browser.gatherUsageStats', 'false', '--client.toolbarMode', 'viewer',
            '--client.showErrorDetails', 'none']
    for origin in [url, f'http://127.0.0.1:{port}', f'http://localhost:{port}']:
        command.extend(['--server.corsAllowedOrigins', origin])
    for host in [hostname, '127.0.0.1', 'localhost']:
        command.extend(['--server.allowedHosts', host])
    return command


def shared_environment(url):
    env = os.environ.copy()
    env.update({
        'TRAFFIC_PUBLIC_MODE': '1', 'TRAFFIC_PUBLIC_URL': url,
    })
    return env


@contextmanager
def handle_shutdown():
    """Handle terminal close/termination using the same cleanup as Ctrl+C."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {}
    for name in ('SIGTERM', 'SIGHUP'):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def check_process(process, name, log):
    if process.poll() is not None:
        details = log.read_text(errors='replace')[-4000:]
        raise SharingError(f'{name} stopped (exit {process.returncode}).\n{details}')


def wait_for_url(process, log, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_process(process, 'Cloudflare tunnel', log)
        match = PUBLIC_URL.search(log.read_text(errors='replace'))
        if match:
            return match.group(0)
        time.sleep(0.2)
    raise SharingError('Cloudflare did not create a public URL. Check the internet connection and try again.\n' + log.read_text(errors='replace')[-4000:])


def wait_for_health(url, processes, *, timeout=90):
    deadline = time.monotonic() + timeout
    last_error = 'No health response received'
    # Bypass proxy settings for loopback; retain normal HTTPS proxy support remotely.
    handlers = [urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))]
    if url.startswith('http://127.0.0.1:'):
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    while time.monotonic() < deadline:
        for process, name, log in processes:
            check_process(process, name, log)
        try:
            request = urllib.request.Request(url.rstrip('/') + '/_stcore/health', headers={'User-Agent': 'UrbanTrafficIntelligence/share'})
            with opener.open(request, timeout=3) as response:
                if response.status == 200 and response.read(16).strip() == b'ok':
                    return
                last_error = f'Unexpected health response (HTTP {response.status})'
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(0.3)
    raise SharingError(f'The app was not reachable at {url}. Check the internet connection/firewall and try again.\nLast check: {last_error}')


def wait_for_public_dns(url, processes, *, timeout=90):
    """Avoid caching NXDOMAIN locally while a brand-new tunnel name propagates.

    Ask Cloudflare's HTTPS DNS API first. Only then use the normal OS resolver
    for the public health check and browser. No DNS settings are changed.
    """
    query = urlencode({'name': urlparse(url).hostname, 'type': 'A'})
    request = urllib.request.Request('https://cloudflare-dns.com/dns-query?' + query,
                                     headers={'Accept': 'application/dns-json'})
    context = ssl.create_default_context(cafile=certifi.where())
    deadline = time.monotonic() + timeout
    last_error = 'The public hostname is not published yet'
    while time.monotonic() < deadline:
        for process, name, log in processes:
            check_process(process, name, log)
        try:
            with urllib.request.urlopen(request, context=context, timeout=3) as response:
                answer = json.loads(response.read(65536))
            if answer.get('Status') == 0 and any(row.get('type') == 1 and row.get('data') for row in answer.get('Answer', [])):
                return
        except (OSError, ValueError) as error:
            last_error = str(error)
        time.sleep(1)
    raise SharingError(f'Cloudflare has not published a usable public hostname. Try sharing again.\nLast check: {last_error}')


def share_app(*, port=8502, cloudflared=None, open_browser=True):
    """Own both subprocesses; every normal/error shutdown closes the public tunnel."""
    try:
        if importlib.util.find_spec('streamlit') is None:
            raise SharingError('Run this command with the project virtual environment after installing requirements.txt.')
        required = ['app.py', 'artifacts/estimator/model.joblib',
                    'artifacts/volume/history.parquet', 'artifacts/journey/history.parquet']
        if not all((ROOT / name).is_file() for name in required):
            raise SharingError('Prepare the data and models first: python -m traffic_intelligence train --dataset estimator')
        check_port(port)
        binary = cloudflared_path(cloudflared)
        with handle_shutdown(), tempfile.TemporaryDirectory(prefix='urban-traffic-share-') as temporary:
            directory = Path(temporary)
            # Explicit empty configuration prevents a personal named-tunnel config
            # from being picked up by this temporary tunnel.
            configuration = directory / 'config.yml'
            configuration.write_text('{}\n')
            tunnel_log, app_log = directory / 'tunnel.log', directory / 'app.log'
            tunnel = server = None
            with tunnel_log.open('w') as tunnel_output, app_log.open('w') as app_output:
                try:
                    print('Creating a temporary public link…', flush=True)
                    tunnel = subprocess.Popen([
                        str(binary), 'tunnel', '--config', str(configuration), '--no-autoupdate',
                        '--protocol', 'http2', '--metrics', '127.0.0.1:0',
                        '--url', f'http://127.0.0.1:{port}',
                    ], cwd=ROOT, stdout=tunnel_output, stderr=subprocess.STDOUT)
                    url = wait_for_url(tunnel, tunnel_log)
                    # Recheck in case another local app started while waiting.
                    check_port(port)
                    server = subprocess.Popen(streamlit_command(url, port), cwd=ROOT,
                                              env=shared_environment(url),
                                              stdout=app_output, stderr=subprocess.STDOUT)
                    processes = [(server, 'Traffic app', app_log), (tunnel, 'Cloudflare tunnel', tunnel_log)]
                    print('Starting the traffic app and checking the public connection…', flush=True)
                    wait_for_health(f'http://127.0.0.1:{port}', processes)
                    wait_for_public_dns(url, processes)
                    wait_for_health(url, processes, timeout=120)
                    print(f'\nPUBLIC LINK: {url}\nLocal shared app: http://127.0.0.1:{port}\n'
                          'Send the PUBLIC LINK to your visitors. Anyone with the link can open the app.\n'
                          'Keep this terminal open and your laptop awake and online.\n'
                          'Press Ctrl+C to stop sharing. A new launch creates a new link.\n', flush=True)
                    if open_browser:
                        try:
                            webbrowser.open(url)
                        except (OSError, webbrowser.Error):
                            print('Open the PUBLIC LINK manually in your browser.', flush=True)
                    while True:
                        for process, name, log in processes:
                            check_process(process, name, log)
                        time.sleep(0.5)
                finally:
                    try:
                        stop_process(tunnel)
                    finally:
                        stop_process(server)
    except KeyboardInterrupt:
        print('\nSharing stopped. The public link is closed.', flush=True)
        return 0
    except (SharingError, OSError, tarfile.TarError, subprocess.SubprocessError) as error:
        print(f'Could not start/continue sharing: {error}', file=sys.stderr, flush=True)
        return 1
