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
  Settings,
  type LucideIcon,
} from "lucide-react";
import {
  api,
  getToken,
  setToken,
  clearToken,
  type UserRow,
  type OrgRow,
  type MemberRow,
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
  | "config"
  | "audit"
  | "health"
  | "docs";

const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Panel" },
  { id: "engines", label: "Motores" },
  { id: "collections", label: "Colecciones" },
  { id: "data", label: "Datos" },
  { id: "users", label: "Usuarios" },
  { id: "orgs", label: "Organizaciones" },
  { id: "keys", label: "API Keys" },
  { id: "access", label: "Acceso" },
  { id: "config", label: "Configuración" },
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
  config: Settings,
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
          <button className={`navitem ${tab === "docs" ? "active" : ""}`} onClick={() => setTab("docs")}>
            <BookOpen size={18} strokeWidth={1.75} aria-hidden />
            <span>Documentación</span>
          </button>
          <div className="userchip">
            <div className="avatar">{email.slice(0, 1).toUpperCase()}</div>
            <div className="who">
              <b>{email}</b>
              <span>{role}</span>
            </div>
          </div>
          <button className="logout" onClick={doLogout}>Cerrar sesión</button>
          <button
            className="navitem"
            style={{ fontSize: 12, opacity: 0.75 }}
            onClick={async () => {
              if (!confirm("¿Cerrar tu sesión en todos los demás dispositivos? La actual se mantiene.")) return;
              try {
                const r = await api.revokeAllSessions();
                alert(`Listo: ${r.revoked} otra(s) sesión(es) cerrada(s).`);
              } catch (e) {
                alert(`No se pudo: ${(e as Error).message}`);
              }
            }}
          >
            Cerrar otras sesiones
          </button>
        </div>
      </aside>
      <main className={`content ${tab === "docs" ? "content-docs" : ""}`}>
        {/* Iframe de documentación: montado una sola vez (precarga en segundo
            plano al entrar) y solo se muestra/oculta — nunca se recarga al
            cambiar de pestaña, así abre al instante. */}
        <iframe
          className="docs-frame"
          src="/docs"
          title="Documentación de Luma"
          style={{ display: tab === "docs" ? "block" : "none" }}
        />
        <div style={{ display: tab === "docs" ? "none" : "block" }}>
          {tab === "dashboard" && <Dashboard />}
          {tab === "engines" && <Engines />}
          {tab === "collections" && <Collections />}
          {tab === "data" && <Data />}
          {tab === "users" && <Users />}
          {tab === "orgs" && <Orgs />}
          {tab === "keys" && <Keys />}
          {tab === "access" && <Access />}
          {tab === "config" && <Configuration />}
          {tab === "audit" && <Audit />}
          {tab === "health" && <Health />}
        </div>
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

/* --- Organizaciones (consola multi-org) ----------------------------------- */
const MEMBER_ROLES = ["owner", "admin", "member", "viewer"] as const;

function Orgs() {
  const { data, error, refresh } = useAsync(() => api.listOrgs());
  const [name, setName] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [selected, setSelected] = useState<OrgRow | null>(null);

  const orgs = data?.orgs ?? [];

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    try {
      await api.createOrg(name.trim());
      setName("");
      refresh();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }
  async function remove(o: OrgRow) {
    if (!confirm(`¿Eliminar la organización "${o.name}" y todos sus usuarios? Esta acción no se puede deshacer.`)) return;
    try {
      await api.deleteOrg(o.id);
      if (selected?.id === o.id) setSelected(null);
      refresh();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }

  return (
    <div>
      <PageHead title="Organizaciones" desc="Crea organizaciones y gestiona quién pertenece a cada una y con qué rol. Un mismo usuario puede pertenecer a varias." />
      <div className="card">
        <form className="row" onSubmit={create}>
          <label className="field" style={{ flex: 1, marginBottom: 0 }}><span>Nueva organización</span>
            <input placeholder="p. ej. Acme Corp" value={name} onChange={(e) => setName(e.target.value)} required /></label>
          <button type="submit">Crear organización</button>
        </form>
        {msg && <ErrorBox msg={msg} />}
      </div>
      {error && <ErrorBox msg={error} />}
      {!error && orgs.length === 0 ? (
        <p className="empty">Aún no hay organizaciones.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Nombre</th><th>ID</th><th style={{ textAlign: "right" }}>Acciones</th></tr>
            </thead>
            <tbody>
              {orgs.map((o) => (
                <tr key={o.id}>
                  <td>{o.name}</td>
                  <td><code className="muted">{o.id}</code></td>
                  <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    <button onClick={() => setSelected(selected?.id === o.id ? null : o)}>
                      {selected?.id === o.id ? "ocultar" : "miembros"}
                    </button>{" "}
                    <button className="danger" onClick={() => remove(o)}>eliminar</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selected && <OrgMembers org={selected} />}
    </div>
  );
}

function OrgMembers({ org }: { org: OrgRow }) {
  const { data, error, refresh } = useAsync(() => api.listOrgMembers(org.id), [org.id]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("member");
  const [msg, setMsg] = useState<string | null>(null);
  const members = data?.members ?? [];

  async function add(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    try {
      await api.addOrgMember(org.id, email.trim(), role);
      setEmail("");
      refresh();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }

  return (
    <div className="card" style={{ marginTop: 20 }}>
      <h3 className="section-label" style={{ marginTop: 0 }}>Miembros de {org.name}</h3>
      <form className="row" onSubmit={add}>
        <label className="field" style={{ flex: 1, marginBottom: 0 }}><span>Correo de un usuario existente</span>
          <input placeholder="persona@empresa.com" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>
        <label className="field" style={{ marginBottom: 0 }}><span>Rol en esta org</span>
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            {MEMBER_ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
          </select></label>
        <button type="submit">Añadir a la org</button>
      </form>
      <p className="perm-note" style={{ marginTop: 10 }}>El usuario debe existir ya (créalo en la pestaña Usuarios). Añadirlo aquí le da acceso a esta organización con el rol elegido, sin quitarle las demás.</p>
      {msg && <ErrorBox msg={msg} />}
      {error && <ErrorBox msg={error} />}
      {members.length === 0 ? (
        <p className="empty" style={{ marginBottom: 0 }}>Esta organización aún no tiene miembros.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Correo</th><th>Rol</th><th style={{ textAlign: "right" }}>Acciones</th></tr>
            </thead>
            <tbody>
              {members.map((m: MemberRow) => (
                <tr key={m.user_id}>
                  <td>{m.email}</td>
                  <td>
                    <select value={m.role} onChange={async (e) => {
                      try { await api.updateOrgMemberRole(org.id, m.user_id, e.target.value); refresh(); }
                      catch (err) { setMsg((err as Error).message); }
                    }}>
                      {MEMBER_ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                    </select>
                  </td>
                  <td style={{ textAlign: "right" }}>
                    <button className="danger" onClick={async () => {
                      try { await api.removeOrgMember(org.id, m.user_id); refresh(); }
                      catch (err) { setMsg((err as Error).message); }
                    }}>quitar</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
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

/* --- Configuración (luma.toml en caliente) -------------------------------- */
type EmbedPreset = { label: string; provider: string; url: string; model: string; needsKey: boolean };
const EMBED_PRESETS: Record<string, EmbedPreset> = {
  openai: { label: "OpenAI", provider: "openai", url: "https://api.openai.com/v1/embeddings", model: "text-embedding-3-small", needsKey: true },
  google: { label: "Google Gemini", provider: "google", url: "https://generativelanguage.googleapis.com/v1beta/openai/embeddings", model: "text-embedding-004", needsKey: true },
  ollama: { label: "Ollama (local)", provider: "ollama", url: "http://127.0.0.1:11434/api/embeddings", model: "nomic-embed-text", needsKey: false },
  selfhost: { label: "Servidor propio (compatible OpenAI)", provider: "openai", url: "http://127.0.0.1:7998/v1/embeddings", model: "", needsKey: false },
  cohere: { label: "Cohere", provider: "cohere", url: "https://api.cohere.ai", model: "embed-multilingual-v3.0", needsKey: true },
  huggingface: { label: "HuggingFace", provider: "huggingface", url: "https://api-inference.huggingface.co", model: "", needsKey: true },
  custom: { label: "Personalizado", provider: "custom", url: "", model: "", needsKey: false },
};
type LlmPreset = { label: string; provider: string; url: string; model: string };
const LLM_PRESETS: Record<string, LlmPreset> = {
  openai: { label: "OpenAI", provider: "openai", url: "https://api.openai.com/v1/chat/completions", model: "gpt-4o-mini" },
  google: { label: "Google Gemini", provider: "openai", url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", model: "gemini-1.5-flash" },
  ollama: { label: "Ollama (local)", provider: "ollama", url: "http://127.0.0.1:11434/v1/chat/completions", model: "llama3" },
  custom: { label: "Personalizado", provider: "custom", url: "", model: "" },
};
// Which preset matches the saved config (by URL, else by provider), else custom.
function matchPreset<T extends { url: string; provider: string }>(presets: Record<string, T>, url: string, provider: string): string {
  const byUrl = Object.entries(presets).find(([, p]) => p.url && p.url === url);
  if (byUrl) return byUrl[0];
  const byProv = Object.entries(presets).find(([k, p]) => k !== "custom" && p.provider === provider);
  return byProv ? byProv[0] : "custom";
}

function Configuration() {
  const { data, error } = useAsync(() => api.config());
  const colls = useAsync(() => api.listCollections());
  const [cfg, setCfg] = useState<Record<string, unknown> | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Embedding provider test state (the key here is used ONLY to test; the
  // running key still comes from the environment).
  const [embedKey, setEmbedKey] = useState("");
  const [probing, setProbing] = useState(false);
  const [probe, setProbe] = useState<{ ok: boolean; dim?: number; error?: string } | null>(null);

  useEffect(() => { if (data && !cfg) setCfg({ ...data }); }, [data, cfg]);

  const set = (k: string, v: unknown) => setCfg((c) => ({ ...(c || {}), [k]: v }));
  const str = (k: string) => String(cfg?.[k] ?? "");
  const num = (k: string) => (cfg?.[k] as number) ?? 0;

  const embedPreset = cfg ? matchPreset(EMBED_PRESETS, str("embedding_url"), str("embedding_provider")) : "custom";
  const llmPreset = cfg ? matchPreset(LLM_PRESETS, str("llm_url"), str("llm_provider")) : "custom";

  function applyEmbedPreset(key: string) {
    const p = EMBED_PRESETS[key];
    if (!p) return;
    setProbe(null);
    setCfg((c) => ({ ...(c || {}), embedding_provider: p.provider, embedding_url: p.url, embedding_model: p.model || (c?.embedding_model ?? "") }));
  }
  function applyLlmPreset(key: string) {
    const p = LLM_PRESETS[key];
    if (!p) return;
    setCfg((c) => ({ ...(c || {}), llm_provider: p.provider, llm_url: p.url, llm_model: p.model || (c?.llm_model ?? "") }));
  }

  async function detectDim() {
    setProbing(true); setProbe(null);
    try {
      const r = await api.probeEmbedding({ provider: str("embedding_provider"), url: str("embedding_url"), api_key: embedKey, model: str("embedding_model") });
      setProbe(r);
      if (r.ok && r.dim) set("embedding_dim", r.dim);
    } catch (err) {
      setProbe({ ok: false, error: (err as Error).message });
    } finally { setProbing(false); }
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!cfg) return;
    setBusy(true); setMsg(null);
    try {
      const r = await api.updateConfig(cfg);
      setMsg(r.message || "Guardado.");
    } catch (err) { setMsg((err as Error).message); }
    finally { setBusy(false); }
  }

  if (error) return <div><PageHead title="Configuración" /><ErrorBox msg={error} /></div>;
  if (!cfg) return <div><PageHead title="Configuración" /><p className="muted">Cargando…</p></div>;

  // Existing collections whose dim differs from the (newly detected) embedding
  // dim — they keep working with their own model, but new ones use the new dim.
  const targetDim = num("embedding_dim");
  const mismatched = (colls.data?.collections ?? []).filter((c) => c.dim != null && targetDim > 0 && c.dim !== targetDim);

  return (
    <div>
      <PageHead title="Configuración" tag="/v1/config" desc="Ajustes de la instancia (luma.toml). Los cambios se guardan y requieren reiniciar el servidor para aplicarse." />

      <div className="card">
        <h3>Motor vectorial</h3>
        <label className="field" style={{ maxWidth: 320 }}>
          <span>Índice por defecto para nuevas colecciones</span>
          <select value={str("index_kind")} onChange={(e) => set("index_kind", e.target.value)}>
            <option value="HNSW">HNSW — máxima precisión</option>
            <option value="IVF_FLAT_Q8">IVF_FLAT_Q8 — equilibrio</option>
            <option value="DISKANN">DISKANN — bajo consumo / escala</option>
          </select>
        </label>
      </div>

      <div className="card">
        <h3>Embeddings</h3>
        <div className="row" style={{ marginBottom: 0 }}>
          <label className="field" style={{ flex: 1, minWidth: 180 }}><span>Proveedor</span>
            <select value={embedPreset} onChange={(e) => applyEmbedPreset(e.target.value)}>
              {Object.entries(EMBED_PRESETS).map(([k, p]) => <option key={k} value={k}>{p.label}</option>)}
            </select></label>
          <label className="field" style={{ flex: 1, minWidth: 160 }}><span>Modelo</span>
            <input value={str("embedding_model")} placeholder="text-embedding-3-small" onChange={(e) => set("embedding_model", e.target.value)} /></label>
        </div>
        <label className="field" style={{ marginTop: 14 }}><span>URL del proveedor</span>
          <input value={str("embedding_url")} placeholder="https://…/v1/embeddings" onChange={(e) => { set("embedding_url", e.target.value); }} /></label>
        <div className="row" style={{ marginBottom: 0, alignItems: "flex-end" }}>
          <label className="field" style={{ flex: 1, minWidth: 200, marginBottom: 0 }}><span>Clave de API (solo para la prueba)</span>
            <input type="password" placeholder={EMBED_PRESETS[embedPreset]?.needsKey ? "requerida por este proveedor" : "opcional"} value={embedKey} onChange={(e) => setEmbedKey(e.target.value)} /></label>
          <button type="button" onClick={detectDim} disabled={probing || !str("embedding_url")}>
            {probing ? "Probando…" : "Probar y detectar dimensión"}
          </button>
        </div>
        {probe && (
          probe.ok
            ? <div className="card notice" style={{ marginTop: 14, marginBottom: 0 }}>✓ Conexión correcta · <strong>dimensión detectada: {probe.dim}</strong>. Se usará en las colecciones nuevas.</div>
            : <ErrorBox msg={`No se pudo conectar: ${probe.error}`} />
        )}
        <div className="row" style={{ marginTop: 14, marginBottom: 0, alignItems: "center" }}>
          <span className="perm-cap">Dimensión activa:</span>
          <strong style={{ fontVariantNumeric: "tabular-nums" }}>{targetDim || "—"}</strong>
          <span className="perm-note" style={{ margin: 0 }}>Se detecta con la prueba; no hace falta escribirla a mano.</span>
        </div>
        {mismatched.length > 0 && (
          <div className="access-state restricted" style={{ marginTop: 14 }}>
            <span className="dot" />
            <div>
              <strong>Atención:</strong> hay {mismatched.length} colección{mismatched.length === 1 ? "" : "es"} creada{mismatched.length === 1 ? "" : "s"} con otra dimensión ({mismatched.map((c) => `${c.collection}=${c.dim}`).join(", ")}).
              Al cambiar de modelo <strong>seguirán funcionando</strong> con su embedding original, pero no son compatibles con el nuevo (dim {targetDim}). Para unificarlas hay que re-indexar/migrar sus datos; mientras tanto no las mezcles con el modelo nuevo.
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Modelo de lenguaje (LLM)</h3>
        <div className="row" style={{ marginBottom: 0 }}>
          <label className="field" style={{ flex: 1, minWidth: 180 }}><span>Proveedor</span>
            <select value={llmPreset} onChange={(e) => applyLlmPreset(e.target.value)}>
              {Object.entries(LLM_PRESETS).map(([k, p]) => <option key={k} value={k}>{p.label}</option>)}
            </select></label>
          <label className="field" style={{ flex: 1, minWidth: 160 }}><span>Modelo</span>
            <input value={str("llm_model")} placeholder="gpt-4o-mini" onChange={(e) => set("llm_model", e.target.value)} /></label>
        </div>
        <label className="field" style={{ marginTop: 14, marginBottom: 0 }}><span>URL del proveedor</span>
          <input value={str("llm_url")} placeholder="https://…/v1/chat/completions" onChange={(e) => set("llm_url", e.target.value)} /></label>
      </div>

      <form onSubmit={save}>
        <button className="primary" disabled={busy} type="submit">{busy ? "…" : "Guardar configuración"}</button>
      </form>
      {msg && <div className={msg.toLowerCase().includes("restart") || msg.includes("reinic") || msg === "Guardado." ? "card notice" : "error"} style={{ marginTop: 14 }}>{msg}</div>}
      <p className="perm-note">
        La <strong>clave de API</strong> del proveedor en producción se carga desde variables de entorno (<code>EMBEDDING_API_KEY</code>, <code>LLM_API_KEY</code>) y no se guarda en <code>luma.toml</code>. La clave de arriba se usa solo para la prueba de conexión.
      </p>
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
