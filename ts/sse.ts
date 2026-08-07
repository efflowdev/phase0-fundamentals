/**
 * The same SSE state machine as `day2_client/sse.py`, line for line, so the two
 * can be compared honestly.
 *
 * One real difference. Python buffers *bytes* and decodes each data line once
 * it is whole. Here the decoder runs first, so it must be told the input is a
 * stream:
 *
 *   new TextDecoder().decode(chunk)                  // corrupts split UTF-8
 *   new TextDecoder().decode(chunk, {stream: true})  // holds the partial bytes
 *
 * A network chunk can end in the middle of a multi-byte character just as
 * easily as in the middle of an event. Without `{stream: true}` you get U+FFFD
 * in the transcript, which — like the `\n\n` bug — will not show up in English
 * test data and will show up the first time a user types an emoji.
 */
export class SseParser {
  #lineBuffer = "";
  #eventLines: string[] = [];
  #pendingCr = false;
  #decoder = new TextDecoder("utf-8");

  feed(chunk: Uint8Array): string[] {
    return this.#consume(this.#decoder.decode(chunk, { stream: true }));
  }

  eof(): string[] {
    const payloads = this.#consume(this.#decoder.decode());
    if (this.#pendingCr) {
      this.#pendingCr = false;
      this.#finishLine(payloads);
    }
    if (this.#lineBuffer) this.#finishLine(payloads);
    if (this.#eventLines.length) this.#emit(payloads);
    return payloads;
  }

  #consume(text: string): string[] {
    const payloads: string[] = [];

    for (const char of text) {
      // A lone \r is a valid line terminator, so it can only be resolved once
      // the next character arrives — which may be in the next network chunk.
      if (this.#pendingCr) {
        this.#pendingCr = false;
        this.#finishLine(payloads);
        if (char === "\n") continue;
      }

      if (char === "\r") this.#pendingCr = true;
      else if (char === "\n") this.#finishLine(payloads);
      else this.#lineBuffer += char;
    }

    return payloads;
  }

  #finishLine(payloads: string[]): void {
    if (this.#lineBuffer) {
      this.#eventLines.push(this.#lineBuffer);
      this.#lineBuffer = "";
      return;
    }
    // An empty line is the event terminator, not an empty line.
    this.#emit(payloads);
  }

  #emit(payloads: string[]): void {
    if (!this.#eventLines.length) return;

    const data: string[] = [];
    for (const line of this.#eventLines) {
      if (line.startsWith(":")) continue;
      if (!line.startsWith("data:")) continue;
      let value = line.slice("data:".length);
      if (value.startsWith(" ")) value = value.slice(1);
      data.push(value);
    }

    this.#eventLines = [];
    if (data.length) payloads.push(data.join("\n"));
  }
}

/** Pull assistant text out of one OpenAI-compatible chunk payload. */
export function contentOf(payload: string): string | undefined {
  if (payload === "[DONE]") return undefined;
  try {
    const chunk = JSON.parse(payload);
    const content = chunk?.choices?.[0]?.delta?.content;
    return typeof content === "string" && content ? content : undefined;
  } catch {
    return undefined;
  }
}
