#!/usr/bin/env python3
"""Minimal stdio MCP server giving Copilot agents public web page access.

Copilot's sandbox keeps agent shell commands offline. Its built-in web_fetch
obeys the same policy, and on Windows enabling outbound access also exposes
localhost and the private network. This server runs outside the sandbox and
fetches only public http(s) addresses, so the shell stays offline while the
agent can still read papers and pages found by web search.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import sys
import tempfile
from urllib.parse import urljoin, urlsplit


SERVER_NAME = "loose-ends-fetch"
TOOL_NAME = "fetch_url"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_LENGTH = 20_000
MAX_LENGTH_LIMIT = 100_000
TIMEOUT_SECONDS = 30
USER_AGENT = "loose-ends-fetch/1.0 (research agent; +https://github.com/tomlinsonk/loose-ends-copilot)"
TEXT_TYPES = re.compile(
    r"^(text/|application/(json|xml|[\w.+-]+\+xml|[\w.+-]+\+json|x-tex|x-latex|javascript))",
    re.IGNORECASE,
)
SKIPPED_TAGS = {"script", "style", "noscript", "svg", "template", "head"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "nav",
    "li", "ul", "ol", "table", "tr", "br", "hr", "pre", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "figure", "figcaption",
}

TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Fetch a public http(s) URL (for example an arXiv abstract, HTML paper, "
        "or journal page) and return its text. Shell commands have no network "
        "access, so use this tool to read pages found by web search. Binary "
        "content such as PDFs or arXiv source archives (https://arxiv.org/e-print/ID) "
        "can be stored in the workspace with save_as and then processed with "
        "shell tools. Local and private network addresses are refused."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Public http or https URL."},
            "max_length": {
                "type": "integer",
                "description": f"Maximum characters to return (default {DEFAULT_MAX_LENGTH}, at most {MAX_LENGTH_LIMIT}).",
            },
            "start_index": {
                "type": "integer",
                "description": "Character offset for continuing a truncated result.",
            },
            "save_as": {
                "type": "string",
                "description": "Optional workspace-relative path where the raw response body is saved.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}


class FetchError(Exception):
    pass


def public_address(host: str, port: int) -> str:
    """Resolve host and return an address only if every result is public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve {host}: {exc}") from exc
    addresses = []
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        mapped = getattr(address, "ipv4_mapped", None)
        if not (mapped or address).is_global:
            raise FetchError(f"refusing non-public address {address} for {host}")
        addresses.append(str(address))
    if not addresses:
        raise FetchError(f"could not resolve {host}")
    return addresses[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, address, port, timeout):
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self):
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, address, port, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self):
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def fetch(url: str) -> tuple[str, str, bytes]:
    """Return (final URL, content type, body), re-vetting every redirect."""
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            raise FetchError("only http and https URLs are supported")
        if not parts.hostname or parts.username or parts.password:
            raise FetchError("URL must name a host and must not contain credentials")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        address = public_address(parts.hostname, port)
        connection_type = (
            _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
        )
        connection = connection_type(parts.hostname, address, port, TIMEOUT_SECONDS)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,text/plain,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                },
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise FetchError(f"HTTP {response.status} without a Location header")
                url = urljoin(url, location)
                continue
            if response.status >= 400:
                raise FetchError(f"HTTP {response.status} {response.reason} for {url}")
            body = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(body) > MAX_DOWNLOAD_BYTES:
                raise FetchError(f"response exceeds {MAX_DOWNLOAD_BYTES} bytes")
            return url, response.getheader("Content-Type") or "", body
        except (OSError, http.client.HTTPException) as exc:
            raise FetchError(f"request to {url} failed: {exc}") from exc
        finally:
            connection.close()
    raise FetchError(f"more than {MAX_REDIRECTS} redirects")


class _TextExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.skip_depth = 0
        self.links: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIPPED_TAGS:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            self.links.append(urljoin(self.base_url, href) if href else None)

    def handle_endtag(self, tag):
        if tag in SKIPPED_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a" and self.links:
            href = self.links.pop()
            if href and not self.skip_depth and href.startswith(("http://", "https://")):
                self.parts.append(f" <{href}>")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def _charset(content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    return match.group(1) if match else "utf-8"


def body_text(url: str, content_type: str, body: bytes) -> str | None:
    media = content_type.split(";", 1)[0].strip().lower()
    sniff = body[:512].lstrip().lower()
    is_html = media in {"text/html", "application/xhtml+xml"} or (
        not media and sniff.startswith((b"<!doctype html", b"<html"))
    )
    if not (is_html or TEXT_TYPES.match(media) or (not media and b"\0" not in body[:1024])):
        return None
    try:
        text = body.decode(_charset(content_type), errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    if is_html:
        parser = _TextExtractor(url)
        parser.feed(text)
        parser.close()
        return parser.text()
    return text


def save_body(workspace: Path, relative: str, body: bytes) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive:
        raise FetchError("save_as must be a workspace-relative path")
    root = workspace.resolve()
    target = (root / candidate).resolve()
    if target == root or root not in target.parents:
        raise FetchError("save_as must stay inside the workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".fetch-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(body)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def _integer(arguments: dict, name: str, default: int, minimum: int, maximum: int) -> int:
    value = arguments.get(name, default)
    if type(value) is not int:
        raise FetchError(f"{name} must be an integer")
    return max(minimum, min(maximum, value))


def call_fetch(arguments: dict, workspace: Path | None) -> str:
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        raise FetchError("url is required")
    max_length = _integer(arguments, "max_length", DEFAULT_MAX_LENGTH, 1, MAX_LENGTH_LIMIT)
    start = _integer(arguments, "start_index", 0, 0, sys.maxsize)
    final_url, content_type, body = fetch(url.strip())
    lines = [f"URL: {final_url}", f"Content-Type: {content_type or 'unknown'}", f"Bytes: {len(body)}"]
    save_as = arguments.get("save_as")
    if save_as is not None:
        if not isinstance(save_as, str) or not save_as.strip():
            raise FetchError("save_as must be a non-empty path")
        if workspace is None:
            raise FetchError("saving is unavailable without a workspace")
        saved = save_body(workspace, save_as.strip(), body)
        lines.append(f"Saved: {saved.relative_to(workspace.resolve()).as_posix()}")
    text = body_text(final_url, content_type, body)
    if text is None:
        if save_as is None:
            lines.append(
                "Binary content was not shown. Fetch again with save_as to store it "
                "in the workspace, or fetch an HTML version of the page."
            )
        return "\n".join(lines)
    chunk = text[start:start + max_length]
    lines.append(f"Characters: {start}-{start + len(chunk)} of {len(text)}")
    if start + len(chunk) < len(text):
        lines.append(f"Truncated: continue with start_index={start + len(chunk)}")
    return "\n".join(lines) + "\n\n" + chunk


def handle(message: dict, workspace: Path | None) -> dict | None:
    method = message.get("method")
    identifier = message.get("id")
    if identifier is None:
        return None  # Notifications need no response.
    if method == "initialize":
        params = message.get("params") or {}
        result = {
            "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": "1.0"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != TOOL_NAME:
            return {"jsonrpc": "2.0", "id": identifier,
                    "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"}}
        try:
            text, is_error = call_fetch(params.get("arguments") or {}, workspace), False
        except FetchError as exc:
            text, is_error = f"Fetch failed: {exc}", True
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": identifier,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": identifier, "result": result}


def serve(workspace: Path | None, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(message, workspace) if isinstance(message, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="directory that save_as writes into")
    args = parser.parse_args(argv)
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    serve(args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
