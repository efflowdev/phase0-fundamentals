import { SseParser, contentOf } from "./sse.ts";

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL = "llama-3.3-70b-versatile";

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * Text-only counterpart to `stream_groq`. No retry ladder, no tool calls — the
 * point of this file is the protocol handling and the cancellation contract,
 * and both are already visible without them.
 *
 * Cancellation, unlike in Python, is safe by default. `for await...of` calls
 * `.return()` on the iterator when the body `break`s or throws, which resumes
 * this generator at the `yield` with a return and runs the `finally`. Python's
 * `async for` does not do the equivalent, which is why the Python client has to
 * document `aclosing`.
 *
 * The AbortSignal is still worth having, for the case `break` cannot reach: an
 * abort decided somewhere other than the consuming loop. That is the same gap
 * `Task.cancel()` fills in Python — JS needs an explicit object for it because
 * the fetch Promise underneath is not cancellable.
 */
export async function* streamGroq(
  messages: Message[],
  { signal }: { signal?: AbortSignal } = {},
): AsyncGenerator<string> {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error("GROQ_API_KEY is not set");

  const response = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: GROQ_MODEL, messages, stream: true }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
  if (!response.body) {
    throw new Error("response had no body");
  }

  const reader = response.body.getReader();
  const parser = new SseParser();

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const payload of parser.feed(value)) {
        const text = contentOf(payload);
        if (text) yield text;
      }
    }
    for (const payload of parser.eof()) {
      const text = contentOf(payload);
      if (text) yield text;
    }
  } finally {
    // Reached on normal completion, on `break` in the caller, and on abort.
    // Releasing the reader is what actually closes the socket.
    await reader.cancel().catch(() => {});
  }
}

async function main(): Promise<void> {
  const controller = new AbortController();
  // Abort from outside the loop, to exercise the path `break` cannot reach.
  const timer = setTimeout(() => controller.abort(), 800);

  try {
    const messages: Message[] = [
      { role: "user", content: "Count slowly from 1 to 200." },
    ];
    let deltas = 0;
    for await (const text of streamGroq(messages, { signal: controller.signal })) {
      process.stdout.write(text);
      deltas += 1;
    }
    console.log(`\n── finished after ${deltas} deltas`);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      console.log("\n── aborted at 800ms");
    } else {
      throw error;
    }
  } finally {
    clearTimeout(timer);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
