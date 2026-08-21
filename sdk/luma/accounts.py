"""
accounts.py — Enterprise account layer: sessions, organizations, members.

Covers ``/v1/auth/*`` (register, login, sessions, multi-org) and the
``/v1/admin/*`` organization and user management routes.

Session tokens
--------------
:meth:`AccountsClient.login` returns an opaque ``lums_…`` token. It is a
*credential*, interchangeable with an API key in the ``Authorization`` header,
so the client does not store it for you — pass it to a new
:class:`~luma.Luma` instance, or set it on the transport, when you want
subsequent calls to use it.

Two behaviours worth knowing before you build on this:

- :meth:`refresh` and :meth:`switch_org` **revoke the token you presented** and
  return a new one. Replace it atomically; a retry with the old token fails.
- Removing a membership revokes that user's sessions bound to the org, so
  access stops immediately rather than at token expiry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._http import Http


class AccountsClient:
    """Accounts, sessions, organizations and members."""

    def __init__(self, http: Http):
        self._http = http

    # ── registration and sessions (public routes) ────────────────────────────

    def register(self, org_name: str, email: str, password: str) -> dict:
        """Create an organization and its first user.

        If the email's domain is mapped via :meth:`set_domain_org`, the user
        joins that organization instead of creating a new one. Passwords must
        be at least 8 characters.
        """
        return self._http.post(
            "/v1/auth/register",
            {"org_name": org_name, "email": email, "password": password},
        )

    async def aregister(self, org_name: str, email: str, password: str) -> dict:
        return await self._http.apost(
            "/v1/auth/register",
            {"org_name": org_name, "email": email, "password": password},
        )

    def login(self, email: str, password: str) -> dict:
        """Exchange credentials for a session token (``lums_…``, 7-day TTL)."""
        return self._http.post("/v1/auth/login", {"email": email, "password": password})

    async def alogin(self, email: str, password: str) -> dict:
        return await self._http.apost(
            "/v1/auth/login", {"email": email, "password": password}
        )

    def logout(self) -> None:
        """Revoke the token currently on the transport."""
        return self._http.post("/v1/auth/logout")

    async def alogout(self) -> None:
        return await self._http.apost("/v1/auth/logout")

    def refresh(self) -> dict:
        """Rotate the session token.

        The presented token is revoked, so replace it with the returned one
        before issuing further requests.
        """
        return self._http.post("/v1/auth/refresh")

    async def arefresh(self) -> dict:
        return await self._http.apost("/v1/auth/refresh")

    # ── session management ───────────────────────────────────────────────────

    def sessions(self) -> dict:
        """List the caller's active sessions. Tokens are never returned."""
        return self._http.get("/v1/auth/sessions")

    async def asessions(self) -> dict:
        return await self._http.aget("/v1/auth/sessions")

    def revoke_other_sessions(self) -> dict:
        """Sign out of every device except the current one."""
        return self._http.post("/v1/auth/sessions/revoke-all")

    async def arevoke_other_sessions(self) -> dict:
        return await self._http.apost("/v1/auth/sessions/revoke-all")

    # ── multi-org ────────────────────────────────────────────────────────────

    def my_orgs(self) -> dict:
        """Organizations the caller is a member of."""
        return self._http.get("/v1/auth/my-orgs")

    async def amy_orgs(self) -> dict:
        return await self._http.aget("/v1/auth/my-orgs")

    def switch_org(self, org_id: str) -> dict:
        """Rebind the session to another organization.

        Rotates the token: the old one is revoked and the new one carries the
        role held in the target org.
        """
        return self._http.post("/v1/auth/switch-org", {"org_id": org_id})

    async def aswitch_org(self, org_id: str) -> dict:
        return await self._http.apost("/v1/auth/switch-org", {"org_id": org_id})

    # ── organizations (admin) ────────────────────────────────────────────────

    def list_orgs(self) -> dict:
        """List organizations. A tenant-bound admin sees only their own."""
        return self._http.get("/v1/admin/orgs")

    async def alist_orgs(self) -> dict:
        return await self._http.aget("/v1/admin/orgs")

    def create_org(self, name: str) -> dict:
        """Create an organization. Platform admin only."""
        return self._http.post("/v1/admin/orgs", {"name": name})

    async def acreate_org(self, name: str) -> dict:
        return await self._http.apost("/v1/admin/orgs", {"name": name})

    def delete_org(self, org_id: str) -> None:
        """Delete an organization.

        Raises ``LumaNotFoundError`` — not a forbidden error — for an org the
        caller may not touch, because the server hides the existence of other
        organizations rather than confirming it with a 403.
        """
        return self._http.delete(f"/v1/admin/orgs/{org_id}")

    async def adelete_org(self, org_id: str) -> None:
        return await self._http.adelete(f"/v1/admin/orgs/{org_id}")

    # ── members ──────────────────────────────────────────────────────────────

    def members(self, org_id: str) -> dict:
        return self._http.get(f"/v1/admin/orgs/{org_id}/members")

    async def amembers(self, org_id: str) -> dict:
        return await self._http.aget(f"/v1/admin/orgs/{org_id}/members")

    def add_member(self, org_id: str, email: str, role: str) -> dict:
        """Add an **existing** account to the org.

        Use :meth:`invite` when the account may not exist yet. A caller cannot
        grant a role at or above their own.
        """
        return self._http.post(
            f"/v1/admin/orgs/{org_id}/members", {"email": email, "role": role}
        )

    async def aadd_member(self, org_id: str, email: str, role: str) -> dict:
        return await self._http.apost(
            f"/v1/admin/orgs/{org_id}/members", {"email": email, "role": role}
        )

    def invite(
        self, org_id: str, email: str, role: str, password: Optional[str] = None
    ) -> dict:
        """Add or create-and-add a user in one step.

        When the account is new and no ``password`` is given, the server
        generates a temporary one and returns it as ``temp_password``. That
        value is returned **once** and cannot be read back later.
        """
        body: Dict[str, Any] = {"email": email, "role": role}
        if password is not None:
            body["password"] = password
        return self._http.post(f"/v1/admin/orgs/{org_id}/invite", body)

    async def ainvite(
        self, org_id: str, email: str, role: str, password: Optional[str] = None
    ) -> dict:
        body: Dict[str, Any] = {"email": email, "role": role}
        if password is not None:
            body["password"] = password
        return await self._http.apost(f"/v1/admin/orgs/{org_id}/invite", body)

    def set_member_role(self, org_id: str, user_id: str, role: str) -> None:
        return self._http.put(
            f"/v1/admin/orgs/{org_id}/members/{user_id}", {"role": role}
        )

    async def aset_member_role(self, org_id: str, user_id: str, role: str) -> None:
        return await self._http.aput(
            f"/v1/admin/orgs/{org_id}/members/{user_id}", {"role": role}
        )

    def remove_member(self, org_id: str, user_id: str) -> None:
        """Remove a membership. Revokes that user's sessions bound to the org."""
        return self._http.delete(f"/v1/admin/orgs/{org_id}/members/{user_id}")

    async def aremove_member(self, org_id: str, user_id: str) -> None:
        return await self._http.adelete(f"/v1/admin/orgs/{org_id}/members/{user_id}")

    # ── users (admin) ────────────────────────────────────────────────────────

    def list_users(self) -> dict:
        """List users, scoped to the caller's org unless they are a platform admin."""
        return self._http.get("/v1/admin/users")

    async def alist_users(self) -> dict:
        return await self._http.aget("/v1/admin/users")

    def create_user(
        self, email: str, password: str, role: str, org_id: Optional[str] = None
    ) -> dict:
        """Create a user. ``org_id`` is honored only for platform admins."""
        body: Dict[str, Any] = {"email": email, "password": password, "role": role}
        if org_id is not None:
            body["org_id"] = org_id
        return self._http.post("/v1/admin/users", body)

    async def acreate_user(
        self, email: str, password: str, role: str, org_id: Optional[str] = None
    ) -> dict:
        body: Dict[str, Any] = {"email": email, "password": password, "role": role}
        if org_id is not None:
            body["org_id"] = org_id
        return await self._http.apost("/v1/admin/users", body)

    def set_user_role(self, user_id: str, role: str) -> None:
        return self._http.put(f"/v1/admin/users/{user_id}/role", {"role": role})

    async def aset_user_role(self, user_id: str, role: str) -> None:
        return await self._http.aput(f"/v1/admin/users/{user_id}/role", {"role": role})

    def delete_user(self, user_id: str) -> None:
        return self._http.delete(f"/v1/admin/users/{user_id}")

    async def adelete_user(self, user_id: str) -> None:
        return await self._http.adelete(f"/v1/admin/users/{user_id}")

    def user_orgs(self, user_id: str) -> dict:
        return self._http.get(f"/v1/admin/users/{user_id}/orgs")

    async def auser_orgs(self, user_id: str) -> dict:
        return await self._http.aget(f"/v1/admin/users/{user_id}/orgs")

    # ── access policy and domain routing ─────────────────────────────────────

    def access_policy(self) -> dict:
        """Read the self-registration allow-list (domains and emails)."""
        return self._http.get("/v1/auth/access-policy")

    async def aaccess_policy(self) -> dict:
        return await self._http.aget("/v1/auth/access-policy")

    def set_access_policy(
        self, domains: Optional[List[str]] = None, emails: Optional[List[str]] = None
    ) -> dict:
        """Replace the allow-list. This is a full replace, not a merge."""
        return self._http.put(
            "/v1/auth/access-policy",
            {"domains": domains or [], "emails": emails or []},
        )

    async def aset_access_policy(
        self, domains: Optional[List[str]] = None, emails: Optional[List[str]] = None
    ) -> dict:
        return await self._http.aput(
            "/v1/auth/access-policy",
            {"domains": domains or [], "emails": emails or []},
        )

    def domain_orgs(self) -> dict:
        return self._http.get("/v1/auth/domain-orgs")

    async def adomain_orgs(self) -> dict:
        return await self._http.aget("/v1/auth/domain-orgs")

    def set_domain_org(self, domain: str, org_id: str, role: str = "member") -> None:
        """Route registrations from ``domain`` into an existing org."""
        return self._http.post(
            "/v1/auth/domain-orgs",
            {"domain": domain, "org_id": org_id, "role": role},
        )

    async def aset_domain_org(
        self, domain: str, org_id: str, role: str = "member"
    ) -> None:
        return await self._http.apost(
            "/v1/auth/domain-orgs",
            {"domain": domain, "org_id": org_id, "role": role},
        )

    def delete_domain_org(self, domain: str) -> None:
        return self._http.delete(f"/v1/auth/domain-orgs/{domain}")

    async def adelete_domain_org(self, domain: str) -> None:
        return await self._http.adelete(f"/v1/auth/domain-orgs/{domain}")

    # ── stats and business audit ─────────────────────────────────────────────

    def stats(self) -> dict:
        """Usage statistics for the admin dashboard."""
        return self._http.get("/v1/admin/stats")

    async def astats(self) -> dict:
        return await self._http.aget("/v1/admin/stats")

    def audit_events(self, limit: int = 100) -> dict:
        """Business audit trail: logins, user and membership changes.

        Distinct from :meth:`~luma.admin.AdminClient.audit`, which is the HTTP
        access log.
        """
        return self._http.get("/v1/admin/audit-events", params={"limit": limit})

    async def aaudit_events(self, limit: int = 100) -> dict:
        return await self._http.aget("/v1/admin/audit-events", params={"limit": limit})
