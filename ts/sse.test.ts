import assert from "node:assert/strict";
import test from "node:test";

import { SseParser, contentOf } from "./sse.ts";
import { streamGroq } from "./client.ts";

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);

test("multi-line data and comments are combined", () => {
  const parser = new SseParser();
  const raw = "data: line one\r\ndata: line two\r\n\r\n: ping\r\ndata: final\r\n\r\n";

  assert.deepEqual(parser.feed(bytes(raw)), ["line one\nline two", "final"]);
});

test("chunking is invariant", () => {
  for (const separator of ["\n", "\r\n"]) {
    for (const size of [1, 3, 7, 13, 64]) {
      const raw = bytes(`data: a${separator}data: b${separator}${separator}`);
      const parser = new SseParser();
      const seen: string[] = [];
      for (let start = 0; start < raw.length; start += size) {
        seen.push(...parser.feed(raw.slice(start, start + size)));
      }
      assert.deepEqual(seen, ["a\nb"], `separator=${JSON.stringify(separator)} size=${size}`);
    }
  }
});

test("a truncated \\r is resolved by the next chunk", () => {
  const parser = new SseParser();

  assert.deepEqual(parser.feed(bytes("data: a\r")), []);
  assert.deepEqual(parser.feed(bytes("\ndata: b\r\n\r\n")), ["a\nb"]);
});

test("eof flushes a pending \\r", () => {
  const parser = new SseParser();

  assert.deepEqual(parser.feed(bytes("data: a\rdata: b\r")), []);
  assert.deepEqual(parser.eof(), ["a\nb"]);
});

test("an empty data value is kept", () => {
  assert.deepEqual(new SseParser().feed(bytes("data:\n\n")), [""]);
});

test("[DONE] is returned as a payload", () => {
  assert.deepEqual(new SseParser().feed(bytes("data: [DONE]\n\n")), ["[DONE]"]);
  assert.equal(contentOf("[DONE]"), undefined);
});

test("a multi-byte character split across chunks survives", () => {
  // The bug `{stream: true}` prevents: "€" is three bytes, cut here between the
  // first and the second.
  const raw = bytes("data: 10€\n\n");
  const parser = new SseParser();
  const cut = raw.indexOf(0xe2) + 1;

  const seen = [...parser.feed(raw.slice(0, cut)), ...parser.feed(raw.slice(cut))];

  assert.deepEqual(seen, ["10€"]);
});

/** A response body we can watch, mirroring `_Stream` on the Python side. */
function watchedFetch(chunks: string[]): { calls: { cancelled: boolean } } {
  const state = { cancelled: false };

  globalThis.fetch = (async (_url: string, init: RequestInit = {}) => {
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const next = chunks.shift();
        if (next === undefined) {
          controller.close();
          return;
        }
        if (init.signal?.aborted) {
          controller.error(new DOMException("aborted", "AbortError"));
          return;
        }
        controller.enqueue(bytes(next));
      },
      cancel() {
        state.cancelled = true;
      },
    });
    return new Response(body, { status: 200 });
  }) as typeof fetch;

  return { calls: state };
}

const textChunk = (content: string) =>
  `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`;

test("breaking out of for-await closes the body — no aclosing needed", async () => {
  process.env.GROQ_API_KEY = "test-key";
  const { calls } = watchedFetch(
    Array.from({ length: 50 }, (_, n) => textChunk(String(n))),
  );

  let seen = 0;
  for await (const _text of streamGroq([{ role: "user", content: "hi" }])) {
    seen += 1;
    if (seen === 3) break;
  }

  assert.equal(seen, 3);
  // The whole contrast with Python in one assertion: `for await` called
  // .return() on the generator, the finally ran, the socket is shut.
  assert.equal(calls.cancelled, true);
});

test("an external abort stops the stream, but not on the exact token", async () => {
  process.env.GROQ_API_KEY = "test-key";
  watchedFetch(Array.from({ length: 50 }, (_, n) => textChunk(String(n))));
  const controller = new AbortController();

  let seen = 0;
  await assert.rejects(async () => {
    for await (const _text of streamGroq([{ role: "user", content: "hi" }], {
      signal: controller.signal,
    })) {
      seen += 1;
      if (seen === 3) controller.abort();
    }
  }, /abort/i);

  // Aborting at 3 delivers 4. A ReadableStream pulls ahead of the consumer
  // (highWaterMark 1), so the next chunk was already buffered when abort fired
  // — and real fetch behaves the same way with bytes already in the socket
  // buffer. Abort bounds the stream; it does not cut it at a chosen token.
  assert.ok(seen > 3, `expected read-ahead past the abort, saw ${seen}`);
  assert.ok(seen < 10, `expected the stream to stop early, saw ${seen}`);
});
