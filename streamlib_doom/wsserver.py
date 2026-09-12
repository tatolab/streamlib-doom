"""A WebSocket server small enough to live inside a processor: the RFC 6455
handshake and framing over the standard library, nothing else.

Two processors host one each — the frame sender over its HTTP server, the game
over a bare socket — so the browser is just two links into the graph.
"""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + WEBSOCKET_GUID).encode()).digest()).decode()


def handshake_response(client_key: str) -> bytes:
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(client_key)}\r\n\r\n"
    ).encode()


def encode_frame(payload: bytes, opcode: int) -> bytes:
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", 0x80 | opcode, length)
    elif length < 65536:
        header = struct.pack("!BBH", 0x80 | opcode, 126, length)
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 127, length)
    return header + payload


def text_frame(text: str) -> bytes:
    return encode_frame(text.encode(), 0x1)


def binary_frame(payload: bytes) -> bytes:
    return encode_frame(payload, 0x2)


def _read_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n > 0:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    """(opcode, payload) of the next client frame; client frames are always masked."""
    first, second = struct.unpack("!BB", _read_exactly(sock, 2))
    opcode = first & 0x0F
    masked = second & 0x80
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exactly(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exactly(sock, 8))[0]
    mask = _read_exactly(sock, 4) if masked else b"\0\0\0\0"
    payload = bytearray(_read_exactly(sock, length))
    if masked:
        for i in range(length):
            payload[i] ^= mask[i & 3]
    return opcode, bytes(payload)


class Broadcaster:
    """Every connected browser gets every frame; a slow one is dropped, never waited on."""

    def __init__(self) -> None:
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()

    def add(self, sock: socket.socket) -> None:
        with self._lock:
            self._clients.add(sock)

    def remove(self, sock: socket.socket) -> None:
        with self._lock:
            self._clients.discard(sock)

    def count(self) -> int:
        with self._lock:
            return len(self._clients)

    def send(self, frame: bytes) -> None:
        with self._lock:
            clients = list(self._clients)
        for sock in clients:
            try:
                sock.sendall(frame)
            except OSError:
                self.remove(sock)
                try:
                    sock.close()
                except OSError:
                    pass


def serve_controls(port: int, on_message, on_client_change=None) -> threading.Thread:
    """A bare WebSocket listener: every text message becomes on_message(dict)."""

    def client(sock: socket.socket) -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = sock.recv(4096)
                if not chunk:
                    return
                request += chunk
            headers = {}
            for line in request.split(b"\r\n")[1:]:
                if b":" in line:
                    name, value = line.split(b":", 1)
                    headers[name.strip().lower().decode()] = value.strip().decode()
            key = headers.get("sec-websocket-key")
            if not key:
                sock.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                return
            sock.sendall(handshake_response(key))
            if on_client_change:
                on_client_change(+1)
            while True:
                opcode, payload = read_frame(sock)
                if opcode == 0x8:
                    return
                if opcode == 0x9:
                    sock.sendall(encode_frame(payload, 0xA))
                elif opcode == 0x1:
                    try:
                        on_message(json.loads(payload.decode()))
                    except ValueError:
                        pass
        except (OSError, ConnectionError):
            return
        finally:
            if on_client_change:
                on_client_change(-1)
            try:
                sock.close()
            except OSError:
                pass

    def listen() -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(8)
        while True:
            sock, _ = server.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=client, args=(sock,), daemon=True).start()

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return thread
