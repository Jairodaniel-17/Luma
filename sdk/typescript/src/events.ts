import { HttpClient } from "./http.js";
import type { EventStreamOptions, LumaEvent } from "./types.js";

/**
 * EventsClient — Server-Sent Events stream from the Luma pub/sub bus.
 *
 * Usage:
 *   for await (const event of client.events.stream()) {
 *     console.log(event);
 *   }
 */
export class EventsClient {
  constructor(private readonly http: HttpClient) {}

  /**
   * Returns an async iterable that yields parsed `LumaEvent` objects.
   *
   * In Node.js 18+ the native `fetch` API is used to consume the SSE stream
   * via `ReadableStream`. In environments that expose the browser `EventSource`
   * API it falls back to that implementation automatically.
   *
   * The stream terminates when the caller breaks out of the `for await` loop
   * or if the connection is closed by the server.
   */
  stream(opts: EventStreamOptions = {}): AsyncIterable<LumaEvent> {
    const params: Record<string, string | number | boolean | undefined> = {
      since: opts.since ?? 0,
    };
    if (opts.types !== undefined) params["types"] = opts.types;
    if (opts.key_prefix !== undefined) params["key_prefix"] = opts.key_prefix;
    if (opts.collection !== undefined) params["collection"] = opts.collection;

    const http = this.http;

    return {
      [Symbol.asyncIterator](): AsyncIterator<LumaEvent> {
        let done = false;
        let responsePromise: Promise<Response> | null = null;
        let readerPromise: Promise<ReadableStreamDefaultReader<Uint8Array>> | null = null;
        let lineBuffer = "";
        let dataLines: string[] = [];

        async function getReader(): Promise<ReadableStreamDefaultReader<Uint8Array>> {
          if (!responsePromise) {
            responsePromise = http.stream("/v1/events", params);
          }
          const res = await responsePromise;
          if (!res.body) {
            throw new Error("Response body is null — SSE not supported");
          }
          return res.body.getReader();
        }

        async function nextEvent(): Promise<IteratorResult<LumaEvent>> {
          if (done) return { value: undefined as unknown as LumaEvent, done: true };

          if (!readerPromise) {
            readerPromise = getReader();
          }
          const reader = await readerPromise;
          const decoder = new TextDecoder();

          while (true) {
            const { value, done: streamDone } = await reader.read();
            if (streamDone) {
              done = true;
              return { value: undefined as unknown as LumaEvent, done: true };
            }

            lineBuffer += decoder.decode(value, { stream: true });
            const lines = lineBuffer.split("\n");
            // Keep the last (potentially incomplete) line in the buffer.
            lineBuffer = lines.pop() ?? "";

            for (const line of lines) {
              const trimmed = line.trimEnd();
              if (trimmed === "") {
                // Blank line = end of event block
                if (dataLines.length > 0) {
                  const payload = dataLines.join("\n");
                  dataLines = [];
                  try {
                    const event = JSON.parse(payload) as LumaEvent;
                    return { value: event, done: false };
                  } catch {
                    // Non-JSON SSE payload (e.g. keepalive comment), skip.
                  }
                }
              } else if (trimmed.startsWith("data: ")) {
                dataLines.push(trimmed.slice(6));
              } else if (trimmed.startsWith(":")) {
                // SSE comment / keepalive — ignore.
              }
            }
          }
        }

        return { next: nextEvent };
      },
    };
  }
}
