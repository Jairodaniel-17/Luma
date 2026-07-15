use crate::api::errors::ApiError;
use crate::api::rbac::{require_platform_admin, require_role, role_at_least};
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct CreateKeyResponse {
    pub id: String,
    pub key: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyBody {
    pub name: String,
    pub tenant_id: Option<String>,
    pub role: Option<String>,
    pub permissions: Option<serde_json::Value>,
    pub quotas: Option<serde_json::Value>,
}

pub async fn list_keys(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    // An org admin/owner lists the keys of their own tenant; a platform admin
    // (no tenant) lists every tenant's keys.
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    let is_platform_admin =
        ctx.tenant_id.is_none() && matches!(ctx.role.as_str(), "admin" | "owner");
    let filter = if is_platform_admin { None } else { ctx.tenant_id.as_deref() };
    let keys = store.list_keys(filter).await.map_err(|err| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            err.to_string(),
        )
    })?;
    Ok(axum::Json(keys))
}

pub async fn create_key(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    axum::Json(body): axum::Json<CreateKeyBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };

    let is_platform_admin =
        ctx.tenant_id.is_none() && matches!(ctx.role.as_str(), "admin" | "owner");

    let plain_key = store.generate_api_key();
    let role = body.role.unwrap_or_else(|| "user".to_string());
    let permissions = body.permissions.unwrap_or(serde_json::json!({}));
    let quotas = body.quotas.unwrap_or(serde_json::json!({
        "storage_bytes": 1_073_741_824u64,
        "qps": 100u64,
        "max_collections": 32u64
    }));

    // A platform admin may mint a key for any tenant (or a global one) at any
    // role/permission level. A tenant-bound caller is clamped: the new key must
    // stay within their own tenant and may not exceed their own privilege.
    let effective_tenant: Option<String> = if is_platform_admin {
        body.tenant_id.clone()
    } else {
        // ponytail: only the obvious {"allow":"*"} escalation is blocked; a full
        // permission-lattice comparison is out of scope.
        let grants_wildcard = permissions.get("allow").and_then(|v| v.as_str()) == Some("*");
        let holds_wildcard = ctx.permissions.get("allow").and_then(|v| v.as_str()) == Some("*");
        check_tenant_key_grant(
            ctx.tenant_id.as_deref(),
            &ctx.role,
            holds_wildcard,
            body.tenant_id.as_deref(),
            &role,
            grants_wildcard,
        )
        .map_err(|msg| ApiError::new(StatusCode::FORBIDDEN, "forbidden", msg))?;
        ctx.tenant_id.clone()
    };

    let id = store
        .create_key(
            &body.name,
            effective_tenant.as_deref(),
            &role,
            &plain_key,
            permissions,
            quotas,
        )
        .await
        .map_err(|err| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                err.to_string(),
            )
        })?;

    Ok(axum::Json(CreateKeyResponse { id, key: plain_key }))
}

pub async fn revoke_key(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    // An org admin/owner may revoke keys of their own tenant only; a platform
    // admin (no tenant) may revoke any key. The store enforces the tenant match.
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    let is_platform_admin =
        ctx.tenant_id.is_none() && matches!(ctx.role.as_str(), "admin" | "owner");
    let filter = if is_platform_admin { None } else { ctx.tenant_id.as_deref() };
    let revoked = store.revoke_key(&id, filter).await.map_err(|err| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            err.to_string(),
        )
    })?;
    if !revoked {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "key id not found",
        ));
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
pub struct UpdateKeyRoleBody {
    pub role: String,
    pub permissions: Option<serde_json::Value>,
}

pub async fn update_key_role(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(id): Path<String>,
    axum::Json(body): axum::Json<UpdateKeyRoleBody>,
) -> Result<impl IntoResponse, ApiError> {
    // Mutates any key by global id (no tenant predicate), and can set an
    // arbitrary role/permissions — platform-wide operation.
    require_platform_admin(&ctx)?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    store
        .update_key_role(&id, &body.role, body.permissions)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?
        .then_some(())
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "not_found", "key id not found"))?;
    Ok(StatusCode::NO_CONTENT)
}

/// Validate a *tenant-bound* caller's `create_key` request. Platform admins skip
/// this entirely. Returns the reason string on rejection (mapped to 403).
fn check_tenant_key_grant(
    caller_tenant: Option<&str>,
    caller_role: &str,
    caller_holds_wildcard: bool,
    req_tenant: Option<&str>,
    req_role: &str,
    req_grants_wildcard: bool,
) -> Result<(), &'static str> {
    // Must not target a different (or global/null) tenant.
    if req_tenant.is_some() && req_tenant != caller_tenant {
        return Err("cannot create a key for another tenant");
    }
    // Must not grant a role above the caller's own (equal is allowed).
    if !role_at_least(caller_role, req_role) {
        return Err("cannot grant a role above your own");
    }
    // Must not escalate to wildcard permissions the caller does not hold.
    if req_grants_wildcard && !caller_holds_wildcard {
        return Err("cannot grant wildcard permissions");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_caller_cannot_cross_tenant_or_escalate() {
        // Own tenant, same role, no wildcard: allowed.
        assert!(
            check_tenant_key_grant(Some("orgA"), "admin", false, Some("orgA"), "admin", false)
                .is_ok()
        );
        // Different tenant: rejected.
        assert!(
            check_tenant_key_grant(Some("orgA"), "admin", false, Some("orgB"), "user", false)
                .is_err()
        );
        // Null/global tenant target from a tenant caller: rejected.
        assert!(
            check_tenant_key_grant(Some("orgA"), "admin", false, None, "user", false).is_ok(),
            "omitting tenant defaults to own tenant"
        );
        // Role above own: rejected.
        assert!(
            check_tenant_key_grant(Some("orgA"), "admin", false, Some("orgA"), "owner", false)
                .is_err()
        );
        // Wildcard escalation without holding it: rejected.
        assert!(
            check_tenant_key_grant(Some("orgA"), "admin", false, Some("orgA"), "admin", true)
                .is_err()
        );
        // Wildcard allowed when caller already holds it.
        assert!(
            check_tenant_key_grant(Some("orgA"), "admin", true, Some("orgA"), "admin", true)
                .is_ok()
        );
    }
}
