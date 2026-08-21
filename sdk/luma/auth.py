from typing import Any, Optional

from ._http import Http


class AuthClient:
    def __init__(self, http: Http):
        self._http = http

    def list_keys(self) -> list:
        return self._http.get("/v1/auth/keys")

    async def alist_keys(self) -> list:
        return await self._http.aget("/v1/auth/keys")

    def create_key(self, name: str, role: str = "user") -> dict:
        return self._http.post("/v1/auth/keys", {"name": name, "role": role})

    async def acreate_key(self, name: str, role: str = "user") -> dict:
        return await self._http.apost("/v1/auth/keys", {"name": name, "role": role})

    def revoke_key(self, id: str) -> None:
        return self._http.delete(f"/v1/auth/keys/{id}")

    async def arevoke_key(self, id: str) -> None:
        return await self._http.adelete(f"/v1/auth/keys/{id}")

    def set_key_role(self, id: str, role: str,
                     permissions: Optional[Any] = None) -> None:
        """Change an API key's role. Platform admin only: it targets any key by
        global id and can set an arbitrary role."""
        return self._http.put(f"/v1/auth/keys/{id}/role",
                              {"role": role, "permissions": permissions})

    async def aset_key_role(self, id: str, role: str,
                            permissions: Optional[Any] = None) -> None:
        return await self._http.aput(f"/v1/auth/keys/{id}/role",
                                     {"role": role, "permissions": permissions})

    # ── roles (RBAC) ─────────────────────────────────────────────────────────

    def list_roles(self) -> dict:
        return self._http.get("/v1/auth/roles")

    async def alist_roles(self) -> dict:
        return await self._http.aget("/v1/auth/roles")

    def create_role(self, name: str, parent_role_id: Optional[str] = None,
                    description: Optional[str] = None) -> dict:
        """Create a role. ``parent_role_id`` makes it inherit that role's
        permissions — the mechanism behind the built-in
        viewer < member < admin < owner ladder."""
        return self._http.post("/v1/auth/roles", {
            "name": name,
            "parent_role_id": parent_role_id,
            "description": description,
        })

    async def acreate_role(self, name: str, parent_role_id: Optional[str] = None,
                           description: Optional[str] = None) -> dict:
        return await self._http.apost("/v1/auth/roles", {
            "name": name,
            "parent_role_id": parent_role_id,
            "description": description,
        })

    def delete_role(self, role_id: str) -> None:
        return self._http.delete(f"/v1/auth/roles/{role_id}")

    async def adelete_role(self, role_id: str) -> None:
        return await self._http.adelete(f"/v1/auth/roles/{role_id}")

    def permissions(self, role_id: str) -> dict:
        return self._http.get(f"/v1/auth/roles/{role_id}/permissions")

    async def apermissions(self, role_id: str) -> dict:
        return await self._http.aget(f"/v1/auth/roles/{role_id}/permissions")

    def grant(self, role_id: str, resource: str, action: str) -> dict:
        return self._http.post(f"/v1/auth/roles/{role_id}/permissions",
                               {"resource": resource, "action": action})

    async def agrant(self, role_id: str, resource: str, action: str) -> dict:
        return await self._http.apost(f"/v1/auth/roles/{role_id}/permissions",
                                      {"resource": resource, "action": action})

    def revoke(self, role_id: str, resource: str, action: str) -> dict:
        """Revoke a permission. Sent as a DELETE with a body, matching the
        server route."""
        return self._http._req("DELETE", f"/v1/auth/roles/{role_id}/permissions",
                               json={"resource": resource, "action": action})

    async def arevoke(self, role_id: str, resource: str, action: str) -> dict:
        return await self._http._areq("DELETE",
                                      f"/v1/auth/roles/{role_id}/permissions",
                                      json={"resource": resource, "action": action})

    def can(self, role: str, resource: str, action: str) -> dict:
        """Test whether a role may perform an action, resolving inheritance.
        Read-only: it grants nothing."""
        return self._http.get("/v1/auth/roles/check", params={
            "role": role, "resource": resource, "action": action})

    async def acan(self, role: str, resource: str, action: str) -> dict:
        return await self._http.aget("/v1/auth/roles/check", params={
            "role": role, "resource": resource, "action": action})
