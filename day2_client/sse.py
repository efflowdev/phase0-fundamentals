class Parser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._index = 0
        self._line_buffer = b""
        self._event_lines: list[bytes] = []
        self._pending_cr = False
        self._emitted_payloads: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        if chunk:
            self._buffer.extend(chunk)
        self._process_buffer()
        return self._drain_emitted_payloads()

    def eof(self) -> list[str]:
        if self._pending_cr:
            self._pending_cr = False
            self._finish_line()
        if self._line_buffer:
            self._finish_line()
        if self._event_lines:
            self._emit_current_event()
        return self._drain_emitted_payloads()

    def _process_buffer(self) -> None:
        while self._index < len(self._buffer):
            if self._pending_cr:
                if self._buffer[self._index : self._index + 1] == b"\n":
                    self._index += 1
                    self._finish_line()
                    self._pending_cr = False
                    continue

                self._pending_cr = False
                self._finish_line()
                continue

            byte = bytes(self._buffer[self._index : self._index + 1])
            self._index += 1

            if byte == b"\r":
                self._pending_cr = True
            elif byte == b"\n":
                self._finish_line()
            else:
                self._line_buffer += byte

        self._compact_buffer()

    def _compact_buffer(self) -> None:
        if self._index > 0:
            del self._buffer[: self._index]
            self._index = 0

    def _finish_line(self) -> None:
        if self._line_buffer:
            self._event_lines.append(self._line_buffer)
            self._line_buffer = b""
            return

        self._emit_current_event()

    def _emit_current_event(self) -> None:
        if self._event_lines:
            payload = self._parse_event(self._event_lines)
            if payload is not None:
                self._emitted_payloads.append(payload)
            self._event_lines = []

    def _parse_event(self, lines: list[bytes]) -> str | None:
        data_lines: list[str] = []
        for line in lines:
            if line.startswith(b":"):
                continue
            if line.startswith(b"data:"):
                payload = line[len(b"data:") :]
                if payload.startswith(b" "):
                    payload = payload[1:]
                data_lines.append(payload.decode("utf-8"))

        if data_lines:
            return "\n".join(data_lines)
        return None

    def _drain_emitted_payloads(self) -> list[str]:
        payloads = self._emitted_payloads
        self._emitted_payloads = []
        return payloads
