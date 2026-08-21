import { HttpClient } from "./http.js";
import type {
  ApiKeyInfo,
  CreateKeyResult,
  KeyRole,
  Permission,
  PermissionCheck,
  PermissionList,
  Role,
  RoleList,
} from "./types.js";

export class AuthClient {
  constructor(private readonly http: HttpClient) {}

  /** List all API keys. Requires admin role. */
  listKeys(): Promise<ApiKeyInfo[]> {
    return this.http.get("/v1/auth/keys");
  }

  /** Create a new API key. Requires admin role. */
  createKey(name: string, role: KeyRole = "user"): Promise<CreateKeyResult> {
    return this.http.post("/v1/auth/keys", { name, role });
  }

  /** Revoke an API key by ID. Requires admin role. */
  revokeKey(id: string): Promise<void> {
    return this.http.delete(`/v1/auth/keys/${encodeURIComponent(id)}`);
  }

  /**
   * Change an API key's role. Platform admin only: it targets any key by global
   * id and can set an arbitrary role.
   */
  setKeyRole(id: string, role: string, permissions?: unknown): Promise<void> {
    return this.http.put(`/v1/auth/keys/${encodeURIComponent(id)}/role`, {
      role,
      permissions: permissions ?? null,
    });
  }

  // ── Roles (RBAC) ──────────────────────────────────────────────────────────

  listRoles(): Promise<RoleList> {
    return this.http.get("/v1/auth/roles");
  }

  /**
   * Create a role. `parentRoleId` makes it inherit that role's permissions —
   * the mechanism behind the built-in viewer < member < admin < owner ladder.
   */
  createRole(name: string, parentRoleId?: string, description?: string): Promise<Role> {
    return this.http.post("/v1/auth/roles", {
      name,
      parent_role_id: parentRoleId ?? null,
      description: description ?? null,
    });
  }

  deleteRole(roleId: string): Promise<void> {
    return this.http.delete(`/v1/auth/roles/${encodeURIComponent(roleId)}`);
  }

  permissions(roleId: string): Promise<PermissionList> {
    return this.http.get(`/v1/auth/roles/${encodeURIComponent(roleId)}/permissions`);
  }

  grant(roleId: string, permission: Permission): Promise<unknown> {
    return this.http.post(
      `/v1/auth/roles/${encodeURIComponent(roleId)}/permissions`,
      permission,
    );
  }

  /**
   * Revoke a permission. Sent as a DELETE carrying a body, which is what the
   * server route expects.
   */
  revoke(roleId: string, permission: Permission): Promise<unknown> {
    return this.http.deleteWithBody(
      `/v1/auth/roles/${encodeURIComponent(roleId)}/permissions`,
      permission,
    );
  }

  /**
   * Test whether a role may perform an action, resolving inheritance.
   * Read-only: it grants nothing.
   */
  can(role: string, resource: string, action: string): Promise<PermissionCheck> {
    return this.http.get("/v1/auth/roles/check", { role, resource, action });
  }
}
