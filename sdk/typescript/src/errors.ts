/**
 * Base error thrown for all non-2xx responses from the Luma server.
 */
export class LumaError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown) {
    const message = extractMessage(body) ?? `HTTP ${status}`;
    super(message);
    this.name = "LumaError";
    this.status = status;
    this.body = body;
    // Maintains proper prototype chain in transpiled ES5.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** 401 Unauthorized — missing or invalid API key. */
export class LumaAuthError extends LumaError {
  constructor(status: number, body: unknown) {
    super(status, body);
    this.name = "LumaAuthError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** 403 Forbidden — authenticated but insufficient role. */
export class LumaForbiddenError extends LumaError {
  constructor(status: number, body: unknown) {
    super(status, body);
    this.name = "LumaForbiddenError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** 404 Not Found. */
export class LumaNotFoundError extends LumaError {
  constructor(status: number, body: unknown) {
    super(status, body);
    this.name = "LumaNotFoundError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** 409 Conflict — e.g. optimistic locking CAS failure. */
export class LumaConflictError extends LumaError {
  constructor(status: number, body: unknown) {
    super(status, body);
    this.name = "LumaConflictError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

function extractMessage(body: unknown): string | undefined {
  if (typeof body === "string") return body || undefined;
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    if (typeof b["message"] === "string") return b["message"];
    if (typeof b["error"] === "string") return b["error"];
  }
  return undefined;
}
