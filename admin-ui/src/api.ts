// Minimal typed API client. All requests use RELATIVE paths (/v1/...) so the
// same build works locally and behind any reverse proxy / TLS terminator.

const TOKEN_KEY = "luma_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const resp = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    let message = `${resp.status} ${resp.statusText}`;
    try {
      const j = await resp.json();
      if (j && j.message) message = j.message;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export const api = {
  login: (email: string, password: string) =>
    request<{ token: string; role: string; org_id: string }>(
      "POST",
      "/v1/auth/login",
      { email, password },
    ),
  register: (org_name: string, email: string, password: string) =>
    request<{ org_id: string }>("POST", "/v1/auth/register", {
      org_name,
      email,
      password,
    }),
  logout: () => request<void>("POST", "/v1/auth/logout"),
  stats: () => request<Record<string, number>>("GET", "/v1/admin/stats"),
  health: () => request<Record<string, unknown>>("GET", "/v1/health"),
  listUsers: () =>
    request<{ users: UserRow[] }>("GET", "/v1/admin/users"),
  createUser: (email: string, password: string, role: string, org_id?: string) =>
    request<UserRow>(
      "POST",
      "/v1/admin/users",
      org_id ? { email, password, role, org_id } : { email, password, role },
    ),
  deleteUser: (id: string) => request<void>("DELETE", `/v1/admin/users/${id}`),
  listOrgs: () => request<{ orgs: OrgRow[] }>("GET", "/v1/admin/orgs"),
  createOrg: (name: string) =>
    request<OrgRow>("POST", "/v1/admin/orgs", { name }),
  deleteOrg: (id: string) => request<void>("DELETE", `/v1/admin/orgs/${id}`),
  listOrgMembers: (org: string) =>
    request<{ members: MemberRow[] }>(
      "GET",
      `/v1/admin/orgs/${encodeURIComponent(org)}/members`,
    ),
  addOrgMember: (org: string, email: string, role: string) =>
    request<{ user_id: string; email: string; role: string }>(
      "POST",
      `/v1/admin/orgs/${encodeURIComponent(org)}/members`,
      { email, role },
    ),
  updateOrgMemberRole: (org: string, userId: string, role: string) =>
    request<void>(
      "PUT",
      `/v1/admin/orgs/${encodeURIComponent(org)}/members/${encodeURIComponent(userId)}`,
      { role },
    ),
  removeOrgMember: (org: string, userId: string) =>
    request<void>(
      "DELETE",
      `/v1/admin/orgs/${encodeURIComponent(org)}/members/${encodeURIComponent(userId)}`,
    ),
  myOrgs: () => request<{ orgs: UserOrgRow[] }>("GET", "/v1/auth/my-orgs"),
  switchOrg: (org_id: string) =>
    request<{ token: string; org_id: string; role: string }>(
      "POST",
      "/v1/auth/switch-org",
      { org_id },
    ),
  listKeys: () => request<KeyRow[]>("GET", "/v1/auth/keys"),
  createKey: (name: string, role: string) =>
    request<{ id: string; key: string }>("POST", "/v1/auth/keys", {
      name,
      role,
    }),
  revokeKey: (id: string) => request<void>("DELETE", `/v1/auth/keys/${id}`),
  auditEvents: () =>
    request<{ events: AuditRow[] }>("GET", "/v1/admin/audit-events?limit=100"),
  getAccessPolicy: () =>
    request<AccessPolicy>("GET", "/v1/auth/access-policy"),
  setAccessPolicy: (policy: AccessPolicy) =>
    request<AccessPolicy>("PUT", "/v1/auth/access-policy", policy),
  listCollections: () =>
    request<{ collections: CollectionRow[] }>("GET", "/v1/vector"),
  listState: () => request<unknown[]>("GET", "/v1/state"),
  config: () => request<Record<string, unknown>>("GET", "/v1/config"),
  updateConfig: (cfg: Record<string, unknown>) =>
    request<{ status: string; message: string }>("PUT", "/v1/config", cfg),
  probeEmbedding: (body: {
    provider: string;
    url: string;
    api_key: string;
    model: string;
  }) =>
    request<{ ok: boolean; dim?: number; error?: string }>(
      "POST",
      "/v1/config/embedding/probe",
      body,
    ),
  hubIngest: (namespace: string, text: string, metadata?: unknown) =>
    request<{ status: string; doc_id: string }>(
      "POST",
      `/v1/db/${encodeURIComponent(namespace)}/ingest`,
      metadata === undefined ? { text } : { text, metadata },
    ),
  hubSearch: (
    namespace: string,
    query: string,
    limit: number,
    sql_filter?: string,
  ) =>
    request<{ results: HubResult[] }>(
      "POST",
      `/v1/db/${encodeURIComponent(namespace)}/search`,
      sql_filter ? { query, limit, sql_filter } : { query, limit },
    ),
};

export type HubResult = Record<string, unknown>;

export interface CollectionRow {
  collection: string;
  dim: number | null;
  metric: string | null;
  count: number | null;
}

export interface AccessPolicy {
  domains: string[];
  emails: string[];
}

export interface UserRow {
  id: string;
  email: string;
  role: string;
  status: string;
  org_id: string;
}
export interface OrgRow {
  id: string;
  name: string;
  created_at_ms: number;
}
export interface MemberRow {
  user_id: string;
  email: string;
  role: string;
  created_at_ms: number;
}
export interface UserOrgRow {
  org_id: string;
  name: string;
  role: string;
  created_at_ms: number;
}
export interface KeyRow {
  id: string;
  name: string;
  role: string;
}
export interface AuditRow {
  id: number;
  ts_ms: number;
  action: string;
  resource: string | null;
  user_id: string | null;
  ip: string | null;
  detail: string | null;
}
