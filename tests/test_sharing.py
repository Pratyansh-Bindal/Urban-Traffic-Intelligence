import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from traffic_intelligence import sharing


URL = 'https://traffic-demo.trycloudflare.com'


def test_download_checks_integrity_reuses_binary_and_repairs_corruption(tmp_path, monkeypatch):
    payload = b'verified executable fixture'
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(sharing, 'DOWNLOADS', {('Linux', 'amd64'): ('cloudflared-linux-amd64', digest)})
    requests = []
    monkeypatch.setattr(sharing, 'get_bytes', lambda url: requests.append(url) or payload)
    binary = sharing.install_cloudflared(tmp_path, system='Linux', machine='x86_64')
    assert binary.read_bytes() == payload and binary.stat().st_mode & 0o111
    assert sharing.install_cloudflared(tmp_path, system='Linux', machine='x86_64') == binary
    assert len(requests) == 1
    binary.write_bytes(b'corrupt')
    sharing.install_cloudflared(tmp_path, system='Linux', machine='x86_64')
    assert binary.read_bytes() == payload and len(requests) == 2
    assert not list(tmp_path.rglob('install-*'))


def test_failed_checksum_never_installs_an_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(sharing, 'get_bytes', lambda url: b'wrong download')
    with pytest.raises(sharing.SharingError, match='checksum'):
        sharing.install_cloudflared(tmp_path, system='Linux', machine='amd64')
    assert not list(tmp_path.rglob('cloudflared'))


def archive_bytes(*, symlink=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        member = tarfile.TarInfo('cloudflared')
        if symlink:
            member.type, member.linkname = tarfile.SYMTYPE, '../../outside'
            archive.addfile(member)
        else:
            member.size = 10
            archive.addfile(member, io.BytesIO(b'executable'))
        member = tarfile.TarInfo('../../outside')
        member.size = 4
        archive.addfile(member, io.BytesIO(b'evil'))
    return stream.getvalue()


@pytest.mark.parametrize('symlink', [False, True])
def test_archive_only_reads_the_expected_regular_binary(tmp_path, monkeypatch, symlink):
    payload = archive_bytes(symlink=symlink)
    monkeypatch.setattr(sharing, 'DOWNLOADS', {('Darwin', 'arm64'): ('cloudflared.tgz', hashlib.sha256(payload).hexdigest())})
    monkeypatch.setattr(sharing, 'get_bytes', lambda url: payload)
    if symlink:
        with pytest.raises(sharing.SharingError, match='expected executable'):
            sharing.install_cloudflared(tmp_path, system='Darwin', machine='arm64')
    else:
        binary = sharing.install_cloudflared(tmp_path, system='Darwin', machine='arm64')
        assert binary.read_bytes() == b'executable'
    assert not (tmp_path / 'outside').exists()


@pytest.mark.parametrize('url', ['http://demo.trycloudflare.com', 'https://demo.trycloudflare.com.evil.org',
                                 'https://demo.trycloudflare.com@evil.org', 'https://trycloudflare.com'])
def test_public_url_does_not_accept_a_different_origin(url):
    assert sharing.PUBLIC_URL.search(url) is None


class Process:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout):
        return self.returncode


def prepare_launcher(tmp_path, monkeypatch):
    for name in ['app.py', 'artifacts/estimator/model.joblib', 'artifacts/volume/history.parquet',
                 'artifacts/journey/history.parquet']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(sharing, 'ROOT', tmp_path)
    monkeypatch.setattr(sharing, 'check_port', lambda port: None)
    monkeypatch.setattr(sharing, 'cloudflared_path', lambda explicit: Path('/example/cloudflared'))
    monkeypatch.setattr(sharing, 'wait_for_url', lambda process, log: URL)
    monkeypatch.setattr(sharing, 'wait_for_health', lambda *args, **kwargs: None)
    monkeypatch.setattr(sharing, 'wait_for_public_dns', lambda *args, **kwargs: None)
    created = []
    def start(command, **kwargs):
        process = Process()
        created.append((process, command, kwargs))
        return process
    monkeypatch.setattr(sharing.subprocess, 'Popen', start)
    return created


def test_launcher_enables_visitor_mode_and_stops_both_processes_on_ctrl_c(tmp_path, monkeypatch, capsys):
    created = prepare_launcher(tmp_path, monkeypatch)
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(sharing.time, 'sleep', interrupt)
    assert sharing.share_app(open_browser=False) == 0
    assert len(created) == 2 and all(process.terminated for process, _, _ in created)
    tunnel, server = created
    assert tunnel[1][-1] == 'http://127.0.0.1:8502'
    configuration = Path(tunnel[1][tunnel[1].index('--config') + 1])
    assert not configuration.parent.exists()  # Logs/configuration are temporary.
    env = server[2]['env']
    assert env['TRAFFIC_PUBLIC_MODE'] == '1' and env['TRAFFIC_PUBLIC_URL'] == URL
    command = server[1]
    for flag in ['--server.enableCORS', '--server.enableXsrfProtection']:
        assert command[command.index(flag) + 1] == 'true'
    assert command[command.index('--server.address') + 1] == '127.0.0.1'
    assert f'PUBLIC LINK: {URL}' in capsys.readouterr().out


def test_real_streamlit_cli_receives_the_exact_host_and_origin_lists(monkeypatch):
    from click.testing import CliRunner
    from streamlit.web import cli
    captured = {}
    monkeypatch.setattr(cli, '_main_run', lambda *args, **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(cli.main, sharing.streamlit_command(URL, 8502)[3:],
                                env={'STREAMLIT_SERVER_ALLOWED_HOSTS': '*',
                                     'STREAMLIT_SERVER_CORS_ALLOWED_ORIGINS': 'https://other.example'})
    assert result.exit_code == 0, result.output
    options = captured['flag_options']
    assert options['server_allowedHosts'] == ('traffic-demo.trycloudflare.com', '127.0.0.1', 'localhost')
    assert options['server_corsAllowedOrigins'] == (URL, 'http://127.0.0.1:8502', 'http://localhost:8502')
    assert options['server_enableCORS'] is True and options['server_enableXsrfProtection'] is True


def test_unreachable_public_app_closes_processes_without_claiming_ready(tmp_path, monkeypatch, capsys):
    created = prepare_launcher(tmp_path, monkeypatch)
    def unavailable(url, processes, **kwargs):
        if url == URL:
            raise sharing.SharingError('Public connection unavailable')
    monkeypatch.setattr(sharing, 'wait_for_health', unavailable)
    assert sharing.share_app(open_browser=False) == 1
    assert all(process.terminated for process, _, _ in created)
    output = capsys.readouterr()
    assert 'PUBLIC LINK:' not in output.out and 'Public connection unavailable' in output.err


def test_interrupt_before_hostname_stops_tunnel(tmp_path, monkeypatch):
    created = prepare_launcher(tmp_path, monkeypatch)
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(sharing, 'wait_for_url', interrupt)
    assert sharing.share_app(open_browser=False) == 0
    assert len(created) == 1 and created[0][0].terminated


def test_startup_failure_after_tunnel_stops_tunnel(tmp_path, monkeypatch, capsys):
    created = prepare_launcher(tmp_path, monkeypatch)
    start = sharing.subprocess.Popen
    def failed_server(command, **kwargs):
        if created:
            raise OSError('cannot start server')
        return start(command, **kwargs)
    monkeypatch.setattr(sharing.subprocess, 'Popen', failed_server)
    assert sharing.share_app(open_browser=False) == 1
    assert len(created) == 1 and created[0][0].terminated
    assert 'cannot start server' in capsys.readouterr().err


def test_browser_open_failure_does_not_end_sharing(tmp_path, monkeypatch, capsys):
    created = prepare_launcher(tmp_path, monkeypatch)
    def no_browser(url):
        raise sharing.webbrowser.Error('no browser')
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(sharing.webbrowser, 'open', no_browser)
    monkeypatch.setattr(sharing.time, 'sleep', interrupt)
    assert sharing.share_app() == 0
    assert all(process.terminated for process, _, _ in created)
    assert 'Open the PUBLIC LINK manually' in capsys.readouterr().out


def test_new_hostname_is_published_before_the_os_resolver_is_used(tmp_path, monkeypatch):
    prepare_launcher(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(sharing, 'wait_for_public_dns', lambda url, processes: calls.append('dns'))
    monkeypatch.setattr(sharing, 'wait_for_health', lambda url, processes, **kwargs: calls.append(url))
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(sharing.time, 'sleep', interrupt)
    assert sharing.share_app(open_browser=False) == 0
    assert calls == ['http://127.0.0.1:8502', 'dns', URL]


def test_public_dns_waits_for_a_positive_address_record(monkeypatch):
    responses = iter([{'Status': 3}, {'Status': 0, 'Answer': [{'type': 1, 'data': '104.16.230.132'}]}])
    requests = []
    def provider(request, **kwargs):
        requests.append(request.full_url)
        return io.BytesIO(json.dumps(next(responses)).encode())
    monkeypatch.setattr(sharing.urllib.request, 'urlopen', provider)
    monkeypatch.setattr(sharing.time, 'sleep', lambda seconds: None)
    sharing.wait_for_public_dns(URL, [])
    assert requests == ['https://cloudflare-dns.com/dns-query?name=traffic-demo.trycloudflare.com&type=A'] * 2


def test_occupied_port_is_rejected(tmp_path, monkeypatch):
    class Occupied:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def setsockopt(self, *args):
            pass
        def bind(self, address):
            raise OSError('already in use')
    monkeypatch.setattr(sharing.socket, 'socket', Occupied)
    with pytest.raises(sharing.SharingError, match='share --port 8503'):
        sharing.check_port(8502)
