import { HttpClient } from "./http.js";
import type { QueueAck, QueueEnqueueResult, QueueReceiveResult, QueueStats } from "./types.js";

/**
 * Durable, disk-backed queues (`/v1/queue`).
 *
 * Delivery is **at-least-once**: `receive` leases messages for a visibility
 * window, and a message not acked before the window expires becomes visible
 * again with its `attempts` counter incremented. Consumers must therefore be
 * idempotent — the same message can legitimately arrive twice.
 *
 * ```typescript
 * const { messages } = await client.queue.receive("jobs", { max: 10, visibilitySecs: 60 });
 * for (const msg of messages) {
 *   await handle(msg.body);
 *   await client.queue.ack("jobs", msg.id);
 * }
 * ```
 */
export class QueueClient {
  constructor(private readonly http: HttpClient) {}

  /** Append a message. `body` is arbitrary JSON, returned verbatim on receive. */
  enqueue(queue: string, body: unknown, delaySecs?: number): Promise<QueueEnqueueResult> {
    const payload: Record<string, unknown> = { body };
    if (delaySecs !== undefined) payload.delay_secs = delaySecs;
    return this.http.post(`/v1/queue/${encodeURIComponent(queue)}`, payload);
  }

  /**
   * Lease up to `max` messages for `visibilitySecs`.
   *
   * Returns a possibly empty list. Each message carries `attempts`; a value
   * above 1 means a previous lease expired without an ack, which is the signal
   * to watch for a poison message.
   */
  receive(
    queue: string,
    options: { max?: number; visibilitySecs?: number } = {},
  ): Promise<QueueReceiveResult> {
    const payload: Record<string, unknown> = {};
    if (options.max !== undefined) payload.max = options.max;
    if (options.visibilitySecs !== undefined) payload.visibility_secs = options.visibilitySecs;
    return this.http.post(`/v1/queue/${encodeURIComponent(queue)}/receive`, payload);
  }

  /** Delete a leased message. Without this it is redelivered. */
  ack(queue: string, id: string): Promise<QueueAck> {
    return this.http.delete(
      `/v1/queue/${encodeURIComponent(queue)}/${encodeURIComponent(id)}`,
    );
  }

  /** Queue depth (all messages) and visible count (available to receive). */
  stats(queue: string): Promise<QueueStats> {
    return this.http.get(`/v1/queue/${encodeURIComponent(queue)}`);
  }
}
