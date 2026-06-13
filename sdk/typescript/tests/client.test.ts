import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { LumaClient } from "../src/client.js";
import { HttpClient } from "../src/http.js";
import {
  LumaAuthError,
  LumaConflictError,
  LumaError,
  LumaForbiddenError,
  LumaNotFoundError,
} from "../src/errors.js";

// ─── Fetch mock helpers ────────────────────────────────────────────────────────

function mockFetch(status: number, body: unknown, contentType = "application/json") {
  const responseBody = typeof body === "string" ? body : JSON.stringify(body);
  const mockResponse = {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": contentType }),
    text: () => Promise.resolve(responseBody),
    json: () => Promise.resolve(body),
    body: null,
  } as unknown as Response;

  return jest.fn<typeof fetch>().mockResolvedValue(mockResponse);
}

// ─── LumaClient construction ─────────────────────────────────────────────────

describe("LumaClient", () => {
  it("exposes all sub-client properties", () => {
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "test" });
    expect(client.vector).toBeDefined();
    expect(client.state).toBeDefined();
    expect(client.doc).toBeDefined();
    expect(client.sql).toBeDefined();
    expect(client.events).toBeDefined();
    expect(client.auth).toBeDefined();
    expect(client.admin).toBeDefined();
  });

  it("creates HubClient via db()", () => {
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "test" });
    const hub = client.db("my-namespace");
    expect(hub).toBeDefined();
    expect(typeof hub.ingest).toBe("function");
    expect(typeof hub.search).toBe("function");
  });

  it("creates MemoryClient via memory()", () => {
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "test" });
    const mem = client.memory("agent-1");
    expect(mem).toBeDefined();
    expect(typeof mem.ingestEvent).toBe("function");
    expect(typeof mem.upsertFact).toBe("function");
    expect(typeof mem.query).toBe("function");
  });

  it("creates DiskAnnClient via diskann()", () => {
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "test" });
    const da = client.diskann("my-collection");
    expect(da).toBeDefined();
    expect(typeof da.build).toBe("function");
    expect(typeof da.status).toBe("function");
  });

  it("strips trailing slash from baseUrl", () => {
    const client = new LumaClient({ baseUrl: "http://localhost:1234/", apiKey: "test" });
    const http = (client as unknown as { http: HttpClient }).http;
    expect(http.baseUrl).toBe("http://localhost:1234");
  });
});

// ─── HttpClient — success paths ───────────────────────────────────────────────

describe("HttpClient — success", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("GET request sends Authorization header", async () => {
    const payload = { collections: [] };
    const mock = mockFetch(200, payload);
    globalThis.fetch = mock;

    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "secret" });
    const result = await http.get("/v1/vector");

    expect(result).toEqual(payload);
    expect(mock).toHaveBeenCalledTimes(1);
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/vector");
    const headers = init.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer secret");
  });

  it("POST request sends JSON body and Content-Type", async () => {
    const payload = { ok: true };
    const mock = mockFetch(200, payload);
    globalThis.fetch = mock;

    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    const result = await http.post("/v1/vector/test", { dim: 384, metric: "cosine" });

    expect(result).toEqual(payload);
    const [, init] = mock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ dim: 384, metric: "cosine" }));
  });

  it("GET with query params appends them to URL", async () => {
    const mock = mockFetch(200, []);
    globalThis.fetch = mock;

    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    await http.get("/v1/state", { prefix: "user:", limit: 50 });

    const [url] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("prefix=user%3A");
    expect(url).toContain("limit=50");
  });

  it("DELETE request uses DELETE method", async () => {
    const mock = mockFetch(200, null);
    globalThis.fetch = mock;

    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    await http.delete("/v1/state/mykey");

    const [, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe("DELETE");
  });

  it("returns undefined for empty 204 responses", async () => {
    const emptyResponse = {
      ok: true,
      status: 204,
      headers: new Headers({ "content-type": "application/json" }),
      text: () => Promise.resolve(""),
      json: () => Promise.resolve(null),
      body: null,
    } as unknown as Response;

    globalThis.fetch = jest.fn<typeof fetch>().mockResolvedValue(emptyResponse);
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    const result = await http.delete("/v1/state/mykey");
    expect(result).toBeUndefined();
  });
});

// ─── HttpClient — error mapping ───────────────────────────────────────────────

describe("HttpClient — error handling", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("throws LumaAuthError on 401", async () => {
    globalThis.fetch = mockFetch(401, { message: "Unauthorized" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "bad" });
    await expect(http.get("/v1/vector")).rejects.toBeInstanceOf(LumaAuthError);
  });

  it("throws LumaForbiddenError on 403", async () => {
    globalThis.fetch = mockFetch(403, { message: "Forbidden" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "user-key" });
    await expect(http.get("/v1/auth/keys")).rejects.toBeInstanceOf(LumaForbiddenError);
  });

  it("throws LumaNotFoundError on 404", async () => {
    globalThis.fetch = mockFetch(404, { message: "Not found" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    await expect(http.get("/v1/state/missing")).rejects.toBeInstanceOf(LumaNotFoundError);
  });

  it("throws LumaConflictError on 409", async () => {
    globalThis.fetch = mockFetch(409, { message: "Conflict" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    await expect(http.put("/v1/state/k", { value: 1, if_revision: 0 })).rejects.toBeInstanceOf(
      LumaConflictError,
    );
  });

  it("throws base LumaError on 500", async () => {
    globalThis.fetch = mockFetch(500, { message: "Internal server error" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    const err = await http.get("/v1/vector").catch((e: unknown) => e) as LumaError;
    expect(err).toBeInstanceOf(LumaError);
    expect(err.status).toBe(500);
    expect(err).not.toBeInstanceOf(LumaAuthError);
  });

  it("LumaError carries .status and .body", async () => {
    globalThis.fetch = mockFetch(400, { message: "bad input", code: "E_INVALID" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "key" });
    const err = await http.post("/v1/vector/c", {}).catch((e: unknown) => e) as LumaError;
    expect(err.status).toBe(400);
    expect((err.body as Record<string, unknown>)["code"]).toBe("E_INVALID");
  });

  it("error subclasses are instanceof LumaError", async () => {
    globalThis.fetch = mockFetch(401, { message: "no" });
    const http = new HttpClient({ baseUrl: "http://localhost:1234", apiKey: "bad" });
    const err = await http.get("/").catch((e) => e);
    expect(err).toBeInstanceOf(LumaError);
    expect(err).toBeInstanceOf(LumaAuthError);
  });
});

// ─── VectorClient ─────────────────────────────────────────────────────────────

describe("VectorClient", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("list() calls GET /v1/vector", async () => {
    const mock = mockFetch(200, { collections: [] });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    const result = await client.vector.list();
    expect(result).toEqual({ collections: [] });
    const [url] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/vector");
  });

  it("upsert() sends correct body", async () => {
    const mock = mockFetch(200, { ok: true });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.vector.upsert("docs", "item-1", [0.1, 0.2, 0.3], { tag: "test" });
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/vector/docs/upsert");
    const body = JSON.parse(init.body as string);
    expect(body.id).toBe("item-1");
    expect(body.vector).toEqual([0.1, 0.2, 0.3]);
    expect(body.meta).toEqual({ tag: "test" });
  });

  it("search() sends POST with vector and k", async () => {
    const mock = mockFetch(200, { hits: [] });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.vector.search("docs", { vector: [0.1, 0.2], k: 5 });
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/vector/docs/search");
    const body = JSON.parse(init.body as string);
    expect(body.k).toBe(5);
  });
});

// ─── HubClient ────────────────────────────────────────────────────────────────

describe("HubClient", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("ingest() sends text and optional metadata", async () => {
    const mock = mockFetch(200, { ok: true });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.db("app").ingest("Hello world", { id: "doc-1", metadata: { src: "web" } });
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/db/app/ingest");
    const body = JSON.parse(init.body as string);
    expect(body.text).toBe("Hello world");
    expect(body.metadata).toEqual({ src: "web" });
  });

  it("search() sends query and limit", async () => {
    const mock = mockFetch(200, { hits: [] });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.db("app").search("what is this?", { limit: 5 });
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/db/app/search");
    const body = JSON.parse(init.body as string);
    expect(body.query).toBe("what is this?");
    expect(body.limit).toBe(5);
  });
});

// ─── MemoryClient ─────────────────────────────────────────────────────────────

describe("MemoryClient", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("ingestEvent() posts to correct namespace path", async () => {
    const mock = mockFetch(200, { ok: true });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.memory("agent-1").ingestEvent("user asked about pricing");
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/memory/agent-1/ingest_event");
    const body = JSON.parse(init.body as string);
    expect(body.text).toBe("user asked about pricing");
  });

  it("upsertFact() sends content and fact_key", async () => {
    const mock = mockFetch(200, { ok: true });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.memory("agent-1").upsertFact("enterprise plan", { fact_key: "user_plan" });
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/memory/agent-1/upsert_fact");
    const body = JSON.parse(init.body as string);
    expect(body.content).toBe("enterprise plan");
    expect(body.fact_key).toBe("user_plan");
  });

  it("query() posts query text", async () => {
    const mock = mockFetch(200, { records: [] });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.memory("agent-1").query("pricing preferences");
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/memory/agent-1/query");
    const body = JSON.parse(init.body as string);
    expect(body.query).toBe("pricing preferences");
  });

  it("optional fields are omitted from request body", async () => {
    const mock = mockFetch(200, { ok: true });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.memory("ns").ingestEvent("some event");
    const [, init] = mock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    // Only 'text' should be present — no undefined fields
    expect(Object.keys(body)).toEqual(["text"]);
  });
});

// ─── StateClient ──────────────────────────────────────────────────────────────

describe("StateClient", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("put() includes ttl_ms and if_revision when provided", async () => {
    const mock = mockFetch(200, { key: "k", value: 1, revision: 2 });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.state.put("mykey", 42, { ttl_ms: 5000, if_revision: 1 });
    const [, init] = mock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.value).toBe(42);
    expect(body.ttl_ms).toBe(5000);
    expect(body.if_revision).toBe(1);
  });

  it("put() omits optional fields when not given", async () => {
    const mock = mockFetch(200, { key: "k", value: 1, revision: 1 });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "k" });
    await client.state.put("mykey", "hello");
    const [, init] = mock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body).not.toHaveProperty("ttl_ms");
    expect(body).not.toHaveProperty("if_revision");
  });
});

// ─── AuthClient ───────────────────────────────────────────────────────────────

describe("AuthClient", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("createKey() sends name and role", async () => {
    const mock = mockFetch(200, { id: "abc", name: "ci", role: "user", key: "sk-xxx", created_at_ms: 0 });
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "admin" });
    const result = await client.auth.createKey("ci", "user");
    expect(result.name).toBe("ci");
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/auth/keys");
    const body = JSON.parse(init.body as string);
    expect(body).toEqual({ name: "ci", role: "user" });
  });

  it("revokeKey() sends DELETE", async () => {
    const mock = mockFetch(200, null);
    globalThis.fetch = mock;
    const client = new LumaClient({ baseUrl: "http://localhost:1234", apiKey: "admin" });
    await client.auth.revokeKey("key-id-123");
    const [url, init] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:1234/v1/auth/keys/key-id-123");
    expect(init.method).toBe("DELETE");
  });
});
