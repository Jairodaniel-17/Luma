import { HttpClient } from "./http.js";
import type {
  AccessPolicy,
  AuditEventList,
  DomainOrgList,
  InviteResult,
  MemberList,
  OrgList,
  Organization,
  RefreshResult,
  SessionList,
  SessionResult,
  SwitchOrgResult,
  User,
  UserList,
} from "./types.js";

/**
 * Enterprise account layer: sessions, organizations, members, users.
 *
 * `login` returns an opaque `lums_…` token. It is a credential interchangeable
 * with an API key in the `Authorization` header, so this client does not store
 * it for you — construct a new `LumaClient` with it when you want later calls
 * to use it.
 *
 * Two behaviours to build around:
 * - `refresh` and `switchOrg` revoke the token you presented and return a new
 *   one, so replace it atomically; retrying with the old one fails.
 * - `removeMember` revokes that user's sessions bound to the org, so access
 *   stops immediately rather than at token expiry.
 */
export class AccountsClient {
  constructor(private readonly http: HttpClient) {}

  // ─── registration and sessions (public routes) ──────────────────────────────

  /**
   * Create an organization and its first user.
   *
   * If the email's domain is mapped via `setDomainOrg`, the user joins that
   * organization instead of creating a new one. Passwords need 8+ characters.
   */
  register(orgName: string, email: string, password: string): Promise<Record<string, string>> {
    return this.http.post("/v1/auth/register", { org_name: orgName, email, password });
  }

  /** Exchange credentials for a session token (`lums_…`, 7-day TTL). */
  login(email: string, password: string): Promise<SessionResult> {
    return this.http.post("/v1/auth/login", { email, password });
  }

  /** Revoke the token currently configured on this client. */
  logout(): Promise<void> {
    return this.http.post("/v1/auth/logout");
  }

  /** Rotate the session token. The presented one is revoked. */
  refresh(): Promise<RefreshResult> {
    return this.http.post("/v1/auth/refresh");
  }

  // ─── session management ─────────────────────────────────────────────────────

  /** List the caller's active sessions. Tokens are never returned. */
  sessions(): Promise<SessionList> {
    return this.http.get("/v1/auth/sessions");
  }

  /** Sign out of every device except the current one. */
  revokeOtherSessions(): Promise<{ revoked: number }> {
    return this.http.post("/v1/auth/sessions/revoke-all");
  }

  // ─── multi-org ──────────────────────────────────────────────────────────────

  /** Organizations the caller is a member of. */
  myOrgs(): Promise<OrgList> {
    return this.http.get("/v1/auth/my-orgs");
  }

  /**
   * Rebind the session to another organization. Rotates the token: the old one
   * is revoked and the new one carries the role held in the target org.
   */
  switchOrg(orgId: string): Promise<SwitchOrgResult> {
    return this.http.post("/v1/auth/switch-org", { org_id: orgId });
  }

  // ─── organizations (admin) ──────────────────────────────────────────────────

  /** List organizations. A tenant-bound admin sees only their own. */
  listOrgs(): Promise<OrgList> {
    return this.http.get("/v1/admin/orgs");
  }

  /** Create an organization. Platform admin only. */
  createOrg(name: string): Promise<Organization> {
    return this.http.post("/v1/admin/orgs", { name });
  }

  /**
   * Delete an organization.
   *
   * Throws `LumaNotFoundError` — not a forbidden error — for an org the caller
   * may not touch: the server hides the existence of other organizations
   * rather than confirming it with a 403.
   */
  deleteOrg(orgId: string): Promise<void> {
    return this.http.delete(`/v1/admin/orgs/${encodeURIComponent(orgId)}`);
  }

  // ─── members ────────────────────────────────────────────────────────────────

  members(orgId: string): Promise<MemberList> {
    return this.http.get(`/v1/admin/orgs/${encodeURIComponent(orgId)}/members`);
  }

  /**
   * Add an **existing** account to the org. Use `invite` when the account may
   * not exist yet. A caller cannot grant a role at or above their own.
   */
  addMember(orgId: string, email: string, role: string): Promise<unknown> {
    return this.http.post(`/v1/admin/orgs/${encodeURIComponent(orgId)}/members`, { email, role });
  }

  /**
   * Add or create-and-add a user in one step.
   *
   * When the account is new and no password is given, the server generates a
   * temporary one and returns it as `temp_password`. That value is returned
   * **once** and cannot be read back later.
   */
  invite(orgId: string, email: string, role: string, password?: string): Promise<InviteResult> {
    const body: Record<string, string> = { email, role };
    if (password !== undefined) body.password = password;
    return this.http.post(`/v1/admin/orgs/${encodeURIComponent(orgId)}/invite`, body);
  }

  setMemberRole(orgId: string, userId: string, role: string): Promise<void> {
    return this.http.put(
      `/v1/admin/orgs/${encodeURIComponent(orgId)}/members/${encodeURIComponent(userId)}`,
      { role },
    );
  }

  /** Remove a membership. Revokes that user's sessions bound to the org. */
  removeMember(orgId: string, userId: string): Promise<void> {
    return this.http.delete(
      `/v1/admin/orgs/${encodeURIComponent(orgId)}/members/${encodeURIComponent(userId)}`,
    );
  }

  // ─── users (admin) ──────────────────────────────────────────────────────────

  /** List users, scoped to the caller's org unless they are a platform admin. */
  listUsers(): Promise<UserList> {
    return this.http.get("/v1/admin/users");
  }

  /** Create a user. `orgId` is honored only for platform admins. */
  createUser(email: string, password: string, role: string, orgId?: string): Promise<User> {
    const body: Record<string, string> = { email, password, role };
    if (orgId !== undefined) body.org_id = orgId;
    return this.http.post("/v1/admin/users", body);
  }

  setUserRole(userId: string, role: string): Promise<void> {
    return this.http.put(`/v1/admin/users/${encodeURIComponent(userId)}/role`, { role });
  }

  deleteUser(userId: string): Promise<void> {
    return this.http.delete(`/v1/admin/users/${encodeURIComponent(userId)}`);
  }

  userOrgs(userId: string): Promise<OrgList> {
    return this.http.get(`/v1/admin/users/${encodeURIComponent(userId)}/orgs`);
  }

  // ─── access policy and domain routing ───────────────────────────────────────

  /** Read the self-registration allow-list. */
  accessPolicy(): Promise<AccessPolicy> {
    return this.http.get("/v1/auth/access-policy");
  }

  /** Replace the allow-list. This is a full replace, not a merge. */
  setAccessPolicy(policy: AccessPolicy): Promise<AccessPolicy> {
    return this.http.put("/v1/auth/access-policy", {
      domains: policy.domains ?? [],
      emails: policy.emails ?? [],
    });
  }

  domainOrgs(): Promise<DomainOrgList> {
    return this.http.get("/v1/auth/domain-orgs");
  }

  /** Route registrations from `domain` into an existing org. */
  setDomainOrg(domain: string, orgId: string, role = "member"): Promise<void> {
    return this.http.post("/v1/auth/domain-orgs", { domain, org_id: orgId, role });
  }

  deleteDomainOrg(domain: string): Promise<void> {
    return this.http.delete(`/v1/auth/domain-orgs/${encodeURIComponent(domain)}`);
  }

  // ─── stats and business audit ───────────────────────────────────────────────

  /** Usage statistics for the admin dashboard. */
  stats(): Promise<Record<string, unknown>> {
    return this.http.get("/v1/admin/stats");
  }

  /**
   * Business audit trail: logins, user and membership changes. Distinct from
   * `AdminClient.audit`, which is the HTTP access log.
   */
  auditEvents(limit = 100): Promise<AuditEventList> {
    return this.http.get("/v1/admin/audit-events", { limit });
  }
}
