import { useEffect, useState, type ReactNode } from "react";
import {
  LayoutGrid,
  Database,
  Users as UsersIcon,
  Building2,
  KeyRound,
  ShieldCheck,
  ScrollText,
  Activity,
  HardDrive,
  Layers,
  BookOpen,
  Boxes,
  FileText,
  Package,
  Inbox,
  Brain,
  Radio,
  Image as ImageIcon,
  Search,
  type LucideIcon,
} from "lucide-react";
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
  | "engines"
  | "collections"
  | "data"
  | "users"
  | "orgs"
  | "keys"
  | "access"
  | "audit"
  | "health";

const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Panel" },
  { id: "engines", label: "Motores" },
  { id: "collections", label: "Colecciones" },
  { id: "data", label: "Datos" },
  { id: "users", label: "Usuarios" },
  { id: "orgs", label: "Organizaciones" },
  { id: "keys", label: "API Keys" },
  { id: "access", label: "Acceso" },
  { id: "audit", label: "Auditoría" },
  { id: "health", label: "Salud" },
];

/* --- Iconos (Lucide, 18px, line) ------------------------------------------ */
const ICONS: Record<string, LucideIcon> = {
  dashboard: LayoutGrid,
  engines: Boxes,
  collections: Layers,
  data: Database,
  users: UsersIcon,
  orgs: Building2,
  keys: KeyRound,
  access: ShieldCheck,
  audit: ScrollText,
  health: Activity,
  storage: HardDrive,
};
function Ico({ name }: { name: string }) {
  const C = ICONS[name] ?? LayoutGrid;
  return <C size={18} strokeWidth={1.75} aria-hidden />;
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
            <button key={t.id} className={`navitem ${t.id === tab ? "active" : ""}`} onClick={() => setTab(t.id)} aria-current={t.id === tab ? "page" : undefined}>
              <Ico name={t.id} />
              <span>{t.label}</span>
            </button>
          ))}
        </nav>
        <div className="foot">
          <a className="navitem" href="/docs" target="_blank" rel="noreferrer">
            <BookOpen size={18} strokeWidth={1.75} aria-hidden />
            <span>Documentación</span>
          </a>
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
        {tab === "engines" && <Engines />}
        {tab === "collections" && <Collections />}
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
  const cards: { label: string; value: string; icon: string }[] = [
    { label: "Organizaciones", value: String(data?.orgs ?? 0), icon: "orgs" },
    { label: "Usuarios", value: String(data?.users ?? 0), icon: "users" },
    { label: "Colecciones", value: String(data?.collections ?? 0), icon: "data" },
    { label: "Eventos de auditoría", value: String(data?.audit_events ?? 0), icon: "audit" },
    { label: "Almacenamiento", value: fmtBytes(data?.storage_bytes ?? 0), icon: "storage" },
  ];
  return (
    <div>
      <PageHead title="Panel" desc="Resumen de tu instancia de Luma." />
      {!data ? <p className="muted">Cargando…</p> : (
        <div className="stat-grid">
          {cards.map((c) => (
            <div className="card stat" key={c.label}>
              <div className="icon"><Ico name={c.icon} /></div>
              <div className="stat-value">{c.value}</div>
              <div className="stat-label">{c.label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* --- Motores (plataforma convergente) ------------------------------------- */
function Engines() {
  const cols = useAsync(() => api.listCollections());
  const kv = useAsync(() => api.listState());
  const nCols = cols.data?.collections?.length;
  const nKeys = Array.isArray(kv.data) ? kv.data.length : undefined;

  const core: { icon: LucideIcon; name: string; desc: string; ep: string; metric?: string }[] = [
    { icon: Layers, name: "Vectorial", desc: "Búsqueda por similitud (ANN). Índices HNSW / IVF-FLAT-Q8 / DiskANN.", ep: "/v1/vector", metric: nCols !== undefined ? `${nCols} colección${nCols === 1 ? "" : "es"}` : undefined },
    { icon: KeyRound, name: "Clave-Valor", desc: "Almacén KV con TTL y compare-and-swap, tipo Redis.", ep: "/v1/state", metric: nKeys !== undefined ? `${nKeys} clave${nKeys === 1 ? "" : "s"}` : undefined },
    { icon: FileText, name: "Documentos / SQL", desc: "Documentos JSON sobre SQLite embebido, con consultas.", ep: "/v1/doc" },
    { icon: Package, name: "Objetos", desc: "Almacenamiento binario tipo S3 / R2 por buckets.", ep: "/v1/blob" },
    { icon: Inbox, name: "Colas", desc: "Mensajería durable: encolar, recibir y confirmar.", ep: "/v1/queue" },
  ];
  const extra: { icon: LucideIcon; name: string; desc: string; ep: string }[] = [
    { icon: Brain, name: "Memoria de agentes", desc: "Memoria episódica, semántica y procedural (NS-Mem).", ep: "/v1/memory" },
    { icon: Search, name: "Hub RAG híbrido", desc: "Ingesta + búsqueda semántica y por palabra clave.", ep: "/v1/db" },
    { icon: Radio, name: "Eventos", desc: "Bus pub/sub con streaming SSE.", ep: "/v1/events" },
    { icon: ImageIcon, name: "Imágenes", desc: "Transformación de imágenes on-the-fly.", ep: "/v1/image" },
  ];

  return (
    <div>
      <PageHead title="Motores" desc="Luma es un motor de datos convergente: cinco primitivas núcleo más servicios de IA, en un solo binario." />
      <h3 className="section-label" style={{ marginTop: 0 }}>Núcleo</h3>
      <div className="engine-grid">
        {core.map((e) => <EngineCard key={e.name} {...e} />)}
      </div>
      <h3 className="section-label">Servicios de IA y convergencia</h3>
      <div className="engine-grid">
        {extra.map((e) => <EngineCard key={e.name} {...e} />)}
      </div>
    </div>
  );
}
function EngineCard({ icon: Icon, name, desc, ep, metric }: { icon: LucideIcon; name: string; desc: string; ep: string; metric?: string }) {
  return (
    <div className="engine-card">
      <div className="engine-head">
        <span className="engine-ico"><Icon size={18} strokeWidth={1.75} aria-hidden /></span>
        <span className="engine-name">{name}</span>
      </div>
      <p className="engine-desc">{desc}</p>
      <div className="engine-foot">
        <code>{ep}</code>
        {metric && <span className="engine-metric">{metric}</span>}
      </div>
    </div>
  );
}

/* --- Colecciones ---------------------------------------------------------- */
const ENGINE_LABEL: Record<string, string> = {
  HNSW: "HNSW · máxima precisión",
  IVF_FLAT_Q8: "IVF-FLAT-Q8 · equilibrio (por defecto)",
  DISKANN: "DiskANN · bajo consumo de RAM / escala",
};
function Collections() {
  const { data, error } = useAsync(() => api.listCollections());
  const cfg = useAsync(() => api.config());
  const engine = (cfg.data?.index_kind as string) || "";
  const rows = data?.collections ?? [];
  return (
    <div>
      <PageHead title="Colecciones" tag="/v1/vector" desc="Tus bases de datos vectoriales en esta organización." />
      {engine && (
        <p className="perm-note" style={{ marginTop: 0, marginBottom: 20 }}>
          Motor de índice activo: <strong>{ENGINE_LABEL[engine] ?? engine}</strong>. Luma soporta HNSW, IVF-FLAT-Q8 y DiskANN.
        </p>
      )}
      {error && <ErrorBox msg={error} />}
      {!error && rows.length === 0 ? (
        <p className="empty">Aún no hay colecciones. Crea una con la API (<code>POST /v1/vector/&lt;nombre&gt;</code>) o desde la pestaña Datos.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Colección</th><th>Dimensión</th><th>Métrica</th><th>Vectores</th></tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.collection}>
                  <td style={{ color: "var(--text)", fontWeight: 500 }}>{c.collection}</td>
                  <td className="num">{c.dim ?? "—"}</td>
                  <td>{c.metric ?? "—"}</td>
                  <td className="num">{c.count ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
      <RolePermissions />
    </div>
  );
}

/* --- Matriz de permisos por rol (referencia) ------------------------------ */
const ROLES = ["viewer", "member", "admin", "owner"] as const;
// Nivel mínimo requerido por capacidad (viewer<member<admin<owner). null = nadie
// (solo el operador de la instancia). Refleja los gates reales del backend.
const CAPS: { cap: string; min: (typeof ROLES)[number] | null }[] = [
  { cap: "Consultar e ingestar datos", min: "viewer" },
  { cap: "Ver salud del servidor", min: "viewer" },
  { cap: "Ver panel y estadísticas", min: "admin" },
  { cap: "Ver registro de auditoría", min: "admin" },
  { cap: "Gestionar API keys de la organización", min: "admin" },
  { cap: "Crear y eliminar usuarios (member, viewer)", min: "admin" },
  { cap: "Crear usuarios con rol admin", min: "owner" },
];
const LEVEL: Record<string, number> = { viewer: 10, member: 20, admin: 30, owner: 40 };
function RolePermissions() {
  return (
    <>
      <h3 className="section-label">Qué puede hacer cada rol</h3>
      <div className="table-wrap">
        <table className="table perm-table">
          <thead>
            <tr>
              <th>Capacidad</th>
              {ROLES.map((r) => <th key={r}>{r}</th>)}
            </tr>
          </thead>
          <tbody>
            {CAPS.map((row) => (
              <tr key={row.cap}>
                <td className="perm-cap">{row.cap}</td>
                {ROLES.map((r) => {
                  const ok = row.min != null && LEVEL[r] >= LEVEL[row.min];
                  return <td key={r}>{ok ? <span className="perm-yes">✓</span> : <span className="perm-no">–</span>}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="perm-note">
        El <strong>operador</strong> (el primer usuario que registró la instancia) añade a lo de <em>owner</em> el control de instancia: la política de registro (pestaña Acceso) y ver todas las organizaciones.
      </p>
    </>
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
      <div className={`access-state ${open ? "open" : "restricted"}`}>
        <span className="dot" />
        {open ? "Registro abierto — cualquier correo válido puede crear una organización" : "Registro restringido a los dominios / correos de abajo"}
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
  if (col === "status" && typeof v === "string") {
    const ok = v === "active";
    return <span className={`status ${ok ? "ok" : ""}`}><span className="dot" />{v}</span>;
  }
  if ((col === "id" || col === "user_id") && typeof v === "string") return <code>{v}</code>;
  if (v === null || v === undefined || v === "") return <span className="muted">—</span>;
  return String(v);
}

function ErrorBox({ msg }: { msg: string }) {
  return <div className="error">{msg}</div>;
}
