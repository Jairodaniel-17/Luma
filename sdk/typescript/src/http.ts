import { LumaAuthError, LumaConflictError, LumaError, LumaForbiddenError, LumaNotFoundError } from "./errors.js";

const STATUS_MAP: Record<number, new (status: number, body: unknown) => LumaError> = {
  401: LumaAuthError,
  403: LumaForbiddenError,
  404: LumaNotFoundError,
  409: LumaConflictError,
};

async function throwForStatus(res: Response): Promise<void> {
  if (res.ok) return;
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = await res.text().catch(() => res.statusText);
  }
  const Ctor = STATUS_MAP[res.status];
  if (Ctor) {
    throw new Ctor(res.status, body);
  }
  throw new LumaError(res.status, body);
}

async function decodeResponse<T>(res: Response): Promise<T> {
  const ct = res.headers.get("content-type") ?? "";
  if (!res.body && res.status === 204) {
    return undefined as unknown as T;
  }
  const text = await res.text();
  if (!text) return undefined as unknown as T;
  if (ct.includes("application/json")) {
    return JSON.parse(text) as T;
  }
  return text as unknown as T;
}

export interface HttpClientOptions {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
}

export class HttpClient {
  readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;

  constructor(options: HttpClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.timeout = options.timeout ?? 30_000;
    this.headers = {
      Authorization: `Bearer ${options.apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    };
  }

  private url(path: string, params?: Record<string, string | number | boolean | undefined>): string {
    const base = `${this.baseUrl}${path}`;
    if (!params) return base;
    const qs = Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
      .join("&");
    return qs ? `${base}?${qs}` : base;
  }

  private signal(): AbortSignal {
    return AbortSignal.timeout(this.timeout);
  }

  async get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    const res = await fetch(this.url(path, params), {
      method: "GET",
      headers: this.headers,
      signal: this.signal(),
    });
    await throwForStatus(res);
    return decodeResponse<T>(res);
  }

  async post<T>(path: string, body?: unknown): Promise<T> {
    const res = await fetch(this.url(path), {
      method: "POST",
      headers: this.headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: this.signal(),
    });
    await throwForStatus(res);
    return decodeResponse<T>(res);
  }

  async put<T>(path: string, body?: unknown): Promise<T> {
    const res = await fetch(this.url(path), {
      method: "PUT",
      headers: this.headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: this.signal(),
    });
    await throwForStatus(res);
    return decodeResponse<T>(res);
  }

  async delete<T>(path: string): Promise<T> {
    const res = await fetch(this.url(path), {
      method: "DELETE",
      headers: this.headers,
      signal: this.signal(),
    });
    await throwForStatus(res);
    return decodeResponse<T>(res);
  }

  /**
   * Returns a raw Response suitable for streaming (SSE). The caller is
   * responsible for consuming the body.
   */
  async stream(
    path: string,
    params?: Record<string, string | number | boolean | undefined>,
  ): Promise<Response> {
    const res = await fetch(this.url(path, params), {
      method: "GET",
      headers: { ...this.headers, Accept: "text/event-stream" },
      // No AbortSignal here — the caller controls the stream lifetime.
    });
    await throwForStatus(res);
    return res;
  }
}
