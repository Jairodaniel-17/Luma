import { useEffect, useState, type ReactNode } from "react";
import {
  api,
  getToken,
  setToken,
  clearToken,
  type UserRow,
  type OrgRow,
  type KeyRow,
  type AuditRow,
  type AccessPolicy,
  type HubResult,
} from "./api";

type Tab =
  | "dashboard"
  | "data"
  | "users"
  | "orgs"
  | "keys"
  | "access"
  | "audit"
  | "health";

const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Panel" },
  { id: "data", label: "Datos" },
  { id: "users", label: "Usuarios" },
  { id: "orgs", label: "Organizaciones" },
  { id: "keys", label: "API Keys" },
  { id: "access", label: "Acceso" },
  { id: "audit", label: "Auditoría" },
  { id: "health", label: "Salud" },
];

/* --- Iconos (SVG en línea, stroke=currentColor) --------------------------- */
const I = {
  dashboard: "M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z",
  data: "M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  users: "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8",
  orgs: "M3 21h18M5 21V7l7-4 7 4v14M9 9h.01M9 13h.01M9 17h.01M15 9h.01M15 13h.01M15 17h.01",
  keys: "M15.5 7.5a4.5 4.5 0 1 0-4.9 4.48L4 19v2h2l1-1h2v-2h2v-2l1.02-1.02A4.5 4.5 0 0 0 15.5 7.5zM16.5 6.5h.01",
  access: "M12 3l7 4v5c0 4.5-3 7.7-7 9-4-1.3-7-4.5-7-9V7l7-4zM9.5 12l2 2 3.5-3.5",
  audit: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M9 13h6M9 17h6M9 9h1",
  health: "M22 12h-4l-3 9L9 3l-3 9H2",
};
function Icon({ name }: { name: keyof typeof I }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={I[name]} />
    </svg>
  );
}

function Brand({ sub }: { sub?: string }) {
  return (
    <div className="brand">
      <span className="mark" />
      <span>Luma{sub && <span className="sub"> · {sub}</span>}</span>
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState<boolean>(!!getToken());
  if (!authed) return <Login onLogin={() => setAuthed(true)} />;
  return <Console onLogout={() => setAuthed(false)} />;
}

/* --- Login / Registro ----------------------------------------------------- */
function Login({ onLogin }: { onLogin: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [orgName, setOrgName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "register") await api.register(orgName, email, password);
      const { token, role } = await api.login(email, password);
      setToken(token);
      localStorage.setItem("luma_email", email);
      localStorage.setItem("luma_role", role ?? "");
      onLogin();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <div className="login">
        <Brand />
        <p className="tagline">
          {mode === "login"
            ? "Entra a tu organización."
            : "Crea tu organización en segundos."}
        </p>
        <div className="seg">
          <button className={mode === "login" ? "on" : ""} onClick={() => { setMode("login"); setError(null); }}>
            Iniciar sesión
          </button>
          <button className={mode === "register" ? "on" : ""} onClick={() => { setMode("register"); setError(null); }}>
            Registrarse
          </button>
        </div>
        <form onSubmit={submit}>
          {mode === "register" && (
            <label className="field">
              <span>Organización</span>
              <input placeholder="Acme Inc." value={orgName} onChange={(e) => setOrgName(e.target.value)} required />
            </label>
          )}
          <label className="field">
            <span>Correo</span>
            <input type="email" placeholder="tú@empresa.com" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label className="field">
            <span>Contraseña</span>
            <input type="password" placeholder={mode === "register" ? "mínimo 8 caracteres" : "••••••••"} value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
          </label>
          {error && <div className="error">{error}</div>}
          <button className="primary" disabled={busy} type="submit">
            {busy ? "…" : mode === "login" ? "Entrar" : "Crear y entrar"}
          </button>
        </form>
      </div>
    </div>
  );
}

/* --- Consola -------------------------------------------------------------- */
function Console({ onLogout }: { onLogout: () => void }) {
  const [tab, setTab] = useState<Tab>("dashboard");
  const email = localStorage.getItem("luma_email") || "usuario";
  const role = localStorage.getItem("luma_role") || "owner";

  async function doLogout() {
    try { await api.logout(); } catch { /* ignore */ }
    clearToken();
    localStorage.removeItem("luma_email");
    localStorage.removeItem("luma_role");
    onLogout();
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <Brand sub="Consola" />
        <nav>
          {TABS.map((t) => (
            <button key={t.id} className={`navitem ${t.id === tab ? "active" : ""}`} onClick={() => setTab(t.id)}>
              <Icon name={t.id} />
              <span>{t.label}</span>
            </button>
          ))}
        </nav>
        <div className="foot">
          <div className="userchip">
            <div className="avatar">{email.slice(0, 1).toUpperCase()}</div>
            <div className="who">
              <b>{email}</b>
              <span>{role}</span>
            </div>
          </div>
          <button className="logout" onClick={doLogout}>Cerrar sesión</button>
        </div>
      </aside>
      <main className="content">
        {tab === "dashboard" && <Dashboard />}
        {tab === "data" && <Data />}
        {tab === "users" && <Users />}
        {tab === "orgs" && <Orgs />}
        {tab === "keys" && <Keys />}
        {tab === "access" && <Access />}
        {tab === "audit" && <Audit />}
        {tab === "health" && <Health />}
      </main>
    </div>
  );
}

function PageHead({ title, desc, tag }: { title: string; desc?: string; tag?: string }) {
  return (
    <div className="page-head">
      <h2>{title}{tag && <span className="kbd">{tag}</span>}</h2>
      {desc && <p>{desc}</p>}
    </div>
  );
}

function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let live = true;
    fn().then((d) => live && setData(d)).catch((e) => live && setError((e as Error).message));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, ...deps]);
  return { data, error, refresh: () => setReload((n) => n + 1) };
}

/* --- Panel ---------------------------------------------------------------- */
function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${u[i]}`;
}
function Dashboard() {
  const { data, error } = useAsync(() => api.stats());
  if (error) return <ErrorBox msg={error} />;
  const cards: { label: string; value: string; icon: keyof typeof I }[] = [
    { label: "Organizaciones", value: String(data?.orgs ?? 0), icon: "orgs" },
    { label: "Usuarios", value: String(data?.users ?? 0), icon: "users" },
    { label: "Colecciones", value: String(data?.collections ?? 0), icon: "data" },
    { label: "Eventos de auditoría", value: String(data?.audit_events ?? 0), icon: "audit" },
    { label: "Almacenamiento", value: fmtBytes(data?.storage_bytes ?? 0), icon: "health" },
  ];
  return (
    <div>
      <PageHead title="Panel" desc="Resumen de tu instancia de Luma." />
      {!data ? <p className="muted">Cargando…</p> : (
        <div className="stat-grid">
          {cards.map((c) => (
            <div className="card stat" key={c.label}>
              <div className="icon"><Icon name={c.icon} /></div>
              <div className="stat-value">{c.value}</div>
              <div className="stat-label">{c.label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* --- Datos (playground) --------------------------------------------------- */
function Data() {
  const [namespace, setNamespace] = useState("docs");
  const [text, setText] = useState("");
  const [metadata, setMetadata] = useState("");
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);

  async function ingest(e: React.FormEvent) {
    e.preventDefault();
    setIngestMsg(null); setIngesting(true);
    try {
      let meta: unknown;
      if (metadata.trim()) {
        try { meta = JSON.parse(metadata); } catch { throw new Error("El metadata debe ser JSON válido"); }
      }
      const r = await api.hubIngest(namespace.trim(), text, meta);
      setIngestMsg(`Ingresado — doc_id: ${r.doc_id}`);
      setText("");
    } catch (err) { setIngestMsg((err as Error).message); }
    finally { setIngesting(false); }
  }

  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(10);
  const [sqlFilter, setSqlFilter] = useState("");
  const [results, setResults] = useState<HubResult[] | null>(null);
  const [searchErr, setSearchErr] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    setSearchErr(null); setSearching(true);
    try {
      const r = await api.hubSearch(namespace.trim(), query, limit, sqlFilter.trim() || undefined);
      setResults(r.results ?? []);
    } catch (err) { setSearchErr((err as Error).message); setResults(null); }
    finally { setSearching(false); }
  }

  return (
    <div>
      <PageHead title="Datos" tag="/v1/db" desc="Ingesta documentos y pruébalos con búsqueda semántica." />
      <label className="field" style={{ maxWidth: 280 }}>
        <span>Namespace</span>
        <input value={namespace} onChange={(e) => setNamespace(e.target.value)} />
      </label>

      <div className="card">
        <h3>Ingresar documento</h3>
        <form onSubmit={ingest}>
          <label className="field">
            <span>Texto</span>
            <textarea rows={4} value={text} placeholder="Texto del documento a indexar…" onChange={(e) => setText(e.target.value)} required />
          </label>
          <label className="field">
            <span>Metadata (JSON, opcional)</span>
            <textarea rows={2} value={metadata} placeholder={'{"categoria":"docs"}'} onChange={(e) => setMetadata(e.target.value)} />
          </label>
          <button className="primary" disabled={ingesting} type="submit">{ingesting ? "…" : "Ingresar"}</button>
        </form>
        {ingestMsg && <div className={ingestMsg.startsWith("Ingresado") ? "card notice" : "error"} style={{ marginTop: 14 }}>{ingestMsg}</div>}
      </div>

      <div className="card">
        <h3>Búsqueda semántica</h3>
        <form onSubmit={search}>
          <label className="field">
            <span>Consulta</span>
            <input value={query} placeholder="consulta en lenguaje natural…" onChange={(e) => setQuery(e.target.value)} required />
          </label>
          <div className="row">
            <label className="field" style={{ flex: 1, marginBottom: 0 }}>
              <span>Filtro SQL sobre metadata (opcional)</span>
              <input value={sqlFilter} placeholder="json_extract(metadata,'$.categoria') = 'docs'" onChange={(e) => setSqlFilter(e.target.value)} />
            </label>
            <label className="field" style={{ width: 100, marginBottom: 0 }}>
              <span>Límite</span>
              <input type="number" min={1} max={100} value={limit} onChange={(e) => setLimit(Number(e.target.value) || 10)} />
            </label>
          </div>
          <button className="primary" disabled={searching} type="submit">{searching ? "…" : "Buscar"}</button>
        </form>
        {searchErr && <ErrorBox msg={searchErr} />}
        {results && <SearchResults results={results} />}
      </div>
    </div>
  );
}

function SearchResults({ results }: { results: HubResult[] }) {
  if (!results.length) return <p className="empty">Sin coincidencias.</p>;
  return (
    <div style={{ marginTop: 16 }}>
      {results.map((r, i) => {
        const id = (r.id ?? r.doc_id ?? `#${i + 1}`) as string;
        const score = r.score as number | undefined;
        const snippet = firstString(r, ["snippet", "content", "text", "text_snippet"]);
        const snippets = Array.isArray(r.snippets) ? (r.snippets as unknown[]).map(String).join(" … ") : undefined;
        return (
          <div className="card result" key={i}>
            <div className="result-head">
              <code>{String(id)}</code>
              {typeof score === "number" && <span className="score">{score.toFixed(4)}</span>}
            </div>
            {(snippet || snippets) && <p className="snippet">{snippet || snippets}</p>}
            <details><summary>ver crudo</summary><pre>{JSON.stringify(r, null, 2)}</pre></details>
          </div>
        );
      })}
    </div>
  );
}
function firstString(obj: Record<string, unknown>, keys: string[]): string | undefined {
  for (const k of keys) { const v = obj[k]; if (typeof v === "string" && v.trim()) return v; }
  return undefined;
}

/* --- Usuarios ------------------------------------------------------------- */
function Users() {
  const { data, error, refresh } = useAsync(() => api.listUsers());
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("member");
  const [msg, setMsg] = useState<string | null>(null);

  async function add(e: React.FormEvent) {
    e.preventDefault(); setMsg(null);
    try { await api.createUser(email, password, role); setEmail(""); setPassword(""); refresh(); }
    catch (err) { setMsg((err as Error).message); }
  }
  return (
    <div>
      <PageHead title="Usuarios" desc="Invita miembros y gestiona sus roles." />
      <div className="card">
        <form className="row" onSubmit={add}>
          <label className="field" style={{ flex: 1, marginBottom: 0 }}><span>Correo</span>
            <input placeholder="persona@empresa.com" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>
          <label className="field" style={{ marginBottom: 0 }}><span>Contraseña</span>
            <input placeholder="mín. 8" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required /></label>
          <label className="field" style={{ marginBottom: 0 }}><span>Rol</span>
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="owner">owner</option><option value="admin">admin</option>
              <option value="member">member</option><option value="viewer">viewer</option>
            </select></label>
          <button type="submit">Añadir</button>
        </form>
        {msg && <ErrorBox msg={msg} />}
      </div>
      {error && <ErrorBox msg={error} />}
      <Table<UserRow> rows={data?.users ?? []} cols={["email", "role", "status"]}
        actions={(u) => <button className="danger" onClick={async () => { await api.deleteUser(u.id); refresh(); }}>eliminar</button>} />
    </div>
  );
}

/* --- Organizaciones ------------------------------------------------------- */
function Orgs() {
  const { data, error } = useAsync(() => api.listOrgs());
  if (error) return <ErrorBox msg={error} />;
  return (
    <div>
      <PageHead title="Organizaciones" desc="Tenants aislados en esta instancia." />
      <Table<OrgRow> rows={data?.orgs ?? []} cols={["id", "name"]} />
    </div>
  );
}

/* --- API Keys ------------------------------------------------------------- */
function Keys() {
  const { data, error, refresh } = useAsync(() => api.listKeys());
  const [name, setName] = useState("");
  const [role, setRole] = useState("member");
  const [newKey, setNewKey] = useState<string | null>(null);
  return (
    <div>
      <PageHead title="API Keys" desc="Tokens para acceder a la API de datos de forma programática." />
      <div className="card">
        <form className="row" onSubmit={async (e) => { e.preventDefault(); const r = await api.createKey(name, role); setNewKey(r.key); setName(""); refresh(); }}>
          <label className="field" style={{ flex: 1, marginBottom: 0 }}><span>Nombre de la key</span>
            <input placeholder="p. ej. servidor-produccion" value={name} onChange={(e) => setName(e.target.value)} required /></label>
          <label className="field" style={{ marginBottom: 0 }}><span>Rol</span>
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="member">member</option><option value="admin">admin</option><option value="viewer">viewer</option>
            </select></label>
          <button type="submit">Crear key</button>
        </form>
        {newKey && <div className="card notice" style={{ marginTop: 14, marginBottom: 0 }}>Nueva key (se muestra una sola vez): <code>{newKey}</code></div>}
      </div>
      {error && <ErrorBox msg={error} />}
      <Table<KeyRow> rows={data ?? []} cols={["name", "role"]}
        actions={(k) => <button className="danger" onClick={async () => { await api.revokeKey(k.id); refresh(); }}>revocar</button>} />
    </div>
  );
}

/* --- Acceso --------------------------------------------------------------- */
function Access() {
  const { data, error } = useAsync(() => api.getAccessPolicy());
  const [domains, setDomains] = useState("");
  const [emails, setEmails] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (data && !loaded) {
      setDomains((data.domains ?? []).join("\n"));
      setEmails((data.emails ?? []).join("\n"));
      setLoaded(true);
    }
  }, [data, loaded]);

  const parse = (t: string) => t.split(/[\n,]/).map((s) => s.trim()).filter(Boolean);

  async function save(e: React.FormEvent) {
    e.preventDefault(); setMsg(null); setBusy(true);
    try {
      const saved = await api.setAccessPolicy({ domains: parse(domains), emails: parse(emails) });
      setDomains(saved.domains.join("\n")); setEmails(saved.emails.join("\n")); setMsg("Guardado.");
    } catch (err) { setMsg((err as Error).message); }
    finally { setBusy(false); }
  }

  if (error) return <ErrorBox msg={error} />;
  const open = parse(domains).length === 0 && parse(emails).length === 0;

  return (
    <div>
      <PageHead title="Control de acceso" desc="Restringe quién puede auto-registrarse. Un dominio o correo por línea; déjalo vacío para registro abierto." />
      <div style={{ marginBottom: 18 }}>
        <span className={`pill-toggle ${open ? "pill-open" : "pill-restricted"}`}>
          {open ? "● Registro ABIERTO" : "● Registro RESTRINGIDO"}
        </span>
      </div>
      <form className="card" onSubmit={save}>
        <label className="field"><span>Dominios permitidos</span>
          <textarea rows={4} placeholder={"acme.com\npartner.io"} value={domains} onChange={(e) => setDomains(e.target.value)} /></label>
        <label className="field"><span>Correos permitidos (exactos)</span>
          <textarea rows={3} placeholder={"ceo@vendor.com"} value={emails} onChange={(e) => setEmails(e.target.value)} /></label>
        <button className="primary" disabled={busy} type="submit">{busy ? "…" : "Guardar política"}</button>
        {msg && <div className={msg === "Guardado." ? "card notice" : "error"} style={{ marginTop: 14, marginBottom: 0 }}>{msg}</div>}
      </form>
    </div>
  );
}

/* --- Auditoría ------------------------------------------------------------ */
function Audit() {
  const { data, error } = useAsync(() => api.auditEvents());
  if (error) return <ErrorBox msg={error} />;
  return (
    <div>
      <PageHead title="Auditoría" desc="Últimos 100 eventos registrados." />
      <Table<AuditRow> rows={data?.events ?? []} cols={["action", "resource", "user_id", "ip", "detail"]} />
    </div>
  );
}

/* --- Salud ---------------------------------------------------------------- */
function Health() {
  const { data, error } = useAsync(() => api.health());
  if (error) return <ErrorBox msg={error} />;
  return (
    <div>
      <PageHead title="Salud del servidor" desc="Estado en vivo del proceso." />
      <pre>{data ? JSON.stringify(data, null, 2) : "Cargando…"}</pre>
    </div>
  );
}

/* --- Tabla genérica ------------------------------------------------------- */
function Table<T extends Record<string, unknown>>({
  rows, cols, actions,
}: { rows: T[]; cols: string[]; actions?: (row: T) => ReactNode }) {
  if (!rows.length) return <p className="empty">Sin registros.</p>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{colLabel(c)}</th>)}{actions && <th />}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => <td key={c}>{cell(c, r[c])}</td>)}
              {actions && <td style={{ textAlign: "right" }}>{actions(r)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function colLabel(c: string): string {
  const m: Record<string, string> = { email: "Correo", role: "Rol", status: "Estado", id: "ID", name: "Nombre",
    action: "Acción", resource: "Recurso", user_id: "Usuario", ip: "IP", detail: "Detalle" };
  return m[c] ?? c;
}
function cell(col: string, v: unknown): ReactNode {
  if (col === "role" && typeof v === "string") return <span className={`badge role-${v}`}>{v}</span>;
  if (col === "status" && typeof v === "string") return <span className={`badge st-active`}>{v}</span>;
  if ((col === "id" || col === "user_id") && typeof v === "string") return <code>{v}</code>;
  if (v === null || v === undefined || v === "") return <span className="muted">—</span>;
  return String(v);
}

function ErrorBox({ msg }: { msg: string }) {
  return <div className="error">{msg}</div>;
}
