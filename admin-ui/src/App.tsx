import { useEffect, useState } from "react";
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

export default function App() {
  const [authed, setAuthed] = useState<boolean>(!!getToken());
  if (!authed) return <Login onLogin={() => setAuthed(true)} />;
  return <Console onLogout={() => setAuthed(false)} />;
}

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
      if (mode === "register") {
        await api.register(orgName, email, password);
      }
      const { token } = await api.login(email, password);
      setToken(token);
      onLogin();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <form className="card login" onSubmit={submit}>
        <h1>
          Luma <span className="muted">Admin</span>
        </h1>
        <p className="muted">
          {mode === "login" ? "Sign in to your organization" : "Create a new organization"}
        </p>
        {mode === "register" && (
          <input
            placeholder="Organization name"
            value={orgName}
            onChange={(e) => setOrgName(e.target.value)}
            required
          />
        )}
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error && <div className="error">{error}</div>}
        <button disabled={busy} type="submit">
          {busy ? "…" : mode === "login" ? "Sign in" : "Create & sign in"}
        </button>
        <button
          type="button"
          className="link"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError(null);
          }}
        >
          {mode === "login"
            ? "Need an organization? Register"
            : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}

function Console({ onLogout }: { onLogout: () => void }) {
  const [tab, setTab] = useState<Tab>("dashboard");
  async function doLogout() {
    try {
      await api.logout();
    } catch {
      /* ignore */
    }
    clearToken();
    onLogout();
  }
  const tabs: Tab[] = [
    "dashboard",
    "data",
    "users",
    "orgs",
    "keys",
    "access",
    "audit",
    "health",
  ];
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          Luma <span className="muted">Admin</span>
        </div>
        <nav>
          {tabs.map((t) => (
            <button
              key={t}
              className={t === tab ? "active" : ""}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>
        <button className="logout" onClick={doLogout}>
          Sign out
        </button>
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

function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let live = true;
    fn()
      .then((d) => live && setData(d))
      .catch((e) => live && setError((e as Error).message));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, ...deps]);
  return { data, error, refresh: () => setReload((n) => n + 1) };
}

function Dashboard() {
  const { data, error } = useAsync(() => api.stats());
  if (error) return <ErrorBox msg={error} />;
  if (!data) return <p className="muted">Loading…</p>;
  const cards: [string, string][] = [
    ["Organizations", String(data.orgs ?? 0)],
    ["Users", String(data.users ?? 0)],
    ["Collections", String(data.collections ?? 0)],
    ["Audit events", String(data.audit_events ?? 0)],
    ["Storage (bytes)", String(data.storage_bytes ?? 0)],
  ];
  return (
    <div>
      <h2>Usage</h2>
      <div className="stat-grid">
        {cards.map(([label, value]) => (
          <div className="card stat" key={label}>
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Data() {
  const [namespace, setNamespace] = useState("docs");

  const [text, setText] = useState("");
  const [metadata, setMetadata] = useState("");
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);

  async function ingest(e: React.FormEvent) {
    e.preventDefault();
    setIngestMsg(null);
    setIngesting(true);
    try {
      let meta: unknown;
      if (metadata.trim()) {
        try {
          meta = JSON.parse(metadata);
        } catch {
          throw new Error("Metadata must be valid JSON");
        }
      }
      const r = await api.hubIngest(namespace.trim(), text, meta);
      setIngestMsg(`Ingested — doc_id: ${r.doc_id}`);
      setText("");
    } catch (err) {
      setIngestMsg((err as Error).message);
    } finally {
      setIngesting(false);
    }
  }

  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(10);
  const [sqlFilter, setSqlFilter] = useState("");
  const [results, setResults] = useState<HubResult[] | null>(null);
  const [searchErr, setSearchErr] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    setSearchErr(null);
    setSearching(true);
    try {
      const r = await api.hubSearch(
        namespace.trim(),
        query,
        limit,
        sqlFilter.trim() || undefined,
      );
      setResults(r.results ?? []);
    } catch (err) {
      setSearchErr((err as Error).message);
      setResults(null);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div>
      <h2>
        Data playground <span className="muted">/v1/db</span>
      </h2>
      <label className="field">
        <span>Namespace</span>
        <input value={namespace} onChange={(e) => setNamespace(e.target.value)} />
      </label>

      <div className="card">
        <h3>Ingest a document</h3>
        <form onSubmit={ingest}>
          <label className="field">
            <span>Text</span>
            <textarea
              rows={4}
              value={text}
              placeholder="Document text to embed & index…"
              onChange={(e) => setText(e.target.value)}
              required
            />
          </label>
          <label className="field">
            <span>Metadata (JSON, optional)</span>
            <textarea
              rows={2}
              value={metadata}
              placeholder={'{"category":"docs"}'}
              onChange={(e) => setMetadata(e.target.value)}
            />
          </label>
          <button disabled={ingesting} type="submit">
            {ingesting ? "…" : "Ingest"}
          </button>
        </form>
        {ingestMsg && (
          <div className={ingestMsg.startsWith("Ingested") ? "card notice" : "error"}>
            {ingestMsg}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Semantic search</h3>
        <form onSubmit={search}>
          <label className="field">
            <span>Query</span>
            <input
              value={query}
              placeholder="natural-language query…"
              onChange={(e) => setQuery(e.target.value)}
              required
            />
          </label>
          <div className="row">
            <label className="field" style={{ flex: 1 }}>
              <span>SQL metadata filter (optional)</span>
              <input
                value={sqlFilter}
                placeholder="json_extract(metadata,'$.category') = 'docs'"
                onChange={(e) => setSqlFilter(e.target.value)}
              />
            </label>
            <label className="field">
              <span>Limit</span>
              <input
                type="number"
                min={1}
                max={100}
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value) || 10)}
              />
            </label>
          </div>
          <button disabled={searching} type="submit">
            {searching ? "…" : "Search"}
          </button>
        </form>
        {searchErr && <ErrorBox msg={searchErr} />}
        {results && <SearchResults results={results} />}
      </div>
    </div>
  );
}

function SearchResults({ results }: { results: HubResult[] }) {
  if (!results.length) return <p className="muted">No matches.</p>;
  return (
    <div>
      {results.map((r, i) => {
        const id = (r.id ?? r.doc_id ?? `#${i + 1}`) as string;
        const score = r.score as number | undefined;
        const snippet = firstString(r, ["snippet", "content", "text", "text_snippet"]);
        const snippets = Array.isArray(r.snippets)
          ? (r.snippets as unknown[]).map(String).join(" … ")
          : undefined;
        return (
          <div className="card result" key={i}>
            <div className="result-head">
              <code>{String(id)}</code>
              {typeof score === "number" && (
                <span className="score">{score.toFixed(4)}</span>
              )}
            </div>
            {(snippet || snippets) && <p className="snippet">{snippet || snippets}</p>}
            <details>
              <summary className="muted">raw</summary>
              <pre>{JSON.stringify(r, null, 2)}</pre>
            </details>
          </div>
        );
      })}
    </div>
  );
}

function firstString(
  obj: Record<string, unknown>,
  keys: string[],
): string | undefined {
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === "string" && v.trim()) return v;
  }
  return undefined;
}

function Users() {
  const { data, error, refresh } = useAsync(() => api.listUsers());
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("member");
  const [msg, setMsg] = useState<string | null>(null);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    try {
      await api.createUser(email, password, role);
      setEmail("");
      setPassword("");
      refresh();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }
  return (
    <div>
      <h2>Users</h2>
      <form className="row" onSubmit={add}>
        <input placeholder="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input placeholder="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <select value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="owner">owner</option>
          <option value="admin">admin</option>
          <option value="member">member</option>
          <option value="viewer">viewer</option>
        </select>
        <button type="submit">Add user</button>
      </form>
      {msg && <ErrorBox msg={msg} />}
      {error && <ErrorBox msg={error} />}
      <Table<UserRow>
        rows={data?.users ?? []}
        cols={["email", "role", "status"]}
        actions={(u) => (
          <button
            className="danger"
            onClick={async () => {
              await api.deleteUser(u.id);
              refresh();
            }}
          >
            delete
          </button>
        )}
      />
    </div>
  );
}

function Orgs() {
  const { data, error } = useAsync(() => api.listOrgs());
  if (error) return <ErrorBox msg={error} />;
  return (
    <div>
      <h2>Organizations</h2>
      <Table<OrgRow> rows={data?.orgs ?? []} cols={["id", "name"]} />
    </div>
  );
}

function Keys() {
  const { data, error, refresh } = useAsync(() => api.listKeys());
  const [name, setName] = useState("");
  const [role, setRole] = useState("member");
  const [newKey, setNewKey] = useState<string | null>(null);
  return (
    <div>
      <h2>API keys</h2>
      <form
        className="row"
        onSubmit={async (e) => {
          e.preventDefault();
          const r = await api.createKey(name, role);
          setNewKey(r.key);
          setName("");
          refresh();
        }}
      >
        <input placeholder="key name" value={name} onChange={(e) => setName(e.target.value)} required />
        <select value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="member">member</option>
          <option value="admin">admin</option>
          <option value="viewer">viewer</option>
        </select>
        <button type="submit">Create key</button>
      </form>
      {newKey && (
        <div className="card notice">
          New key (shown once): <code>{newKey}</code>
        </div>
      )}
      {error && <ErrorBox msg={error} />}
      <Table<KeyRow>
        rows={data ?? []}
        cols={["name", "role"]}
        actions={(k) => (
          <button
            className="danger"
            onClick={async () => {
              await api.revokeKey(k.id);
              refresh();
            }}
          >
            revoke
          </button>
        )}
      />
    </div>
  );
}

function Access() {
  const { data, error } = useAsync(() => api.getAccessPolicy());
  const [domains, setDomains] = useState("");
  const [emails, setEmails] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Seed the editable text areas once the current policy arrives.
  useEffect(() => {
    if (data && !loaded) {
      setDomains((data.domains ?? []).join("\n"));
      setEmails((data.emails ?? []).join("\n"));
      setLoaded(true);
    }
  }, [data, loaded]);

  function parse(text: string): string[] {
    return text
      .split(/[\n,]/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    setBusy(true);
    try {
      const policy: AccessPolicy = {
        domains: parse(domains),
        emails: parse(emails),
      };
      const saved = await api.setAccessPolicy(policy);
      setDomains(saved.domains.join("\n"));
      setEmails(saved.emails.join("\n"));
      setMsg("Saved.");
    } catch (err) {
      setMsg((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (error) return <ErrorBox msg={error} />;
  const open =
    parse(domains).length === 0 && parse(emails).length === 0;

  return (
    <div>
      <h2>Access control</h2>
      <p className="muted">
        Restrict who can self-register. List allowed email domains and/or exact
        addresses (one per line). Leave both empty to allow open registration.
      </p>
      <div className={open ? "card notice" : "card"}>
        {open
          ? "Registration is OPEN — anyone with a valid email can create an organization."
          : "Registration is RESTRICTED to the domains / emails below."}
      </div>
      <form onSubmit={save}>
        <label className="field">
          <span>Allowed domains</span>
          <textarea
            rows={5}
            placeholder={"acme.com\npartner.io"}
            value={domains}
            onChange={(e) => setDomains(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Allowed emails (exact)</span>
          <textarea
            rows={4}
            placeholder={"ceo@vendor.com"}
            value={emails}
            onChange={(e) => setEmails(e.target.value)}
          />
        </label>
        <button disabled={busy} type="submit">
          {busy ? "…" : "Save policy"}
        </button>
      </form>
      {msg && <div className={msg === "Saved." ? "card notice" : "error"}>{msg}</div>}
    </div>
  );
}

function Audit() {
  const { data, error } = useAsync(() => api.auditEvents());
  if (error) return <ErrorBox msg={error} />;
  return (
    <div>
      <h2>Audit log</h2>
      <Table<AuditRow>
        rows={data?.events ?? []}
        cols={["action", "resource", "user_id", "ip", "detail"]}
      />
    </div>
  );
}

function Health() {
  const { data, error } = useAsync(() => api.health());
  if (error) return <ErrorBox msg={error} />;
  return (
    <div>
      <h2>Server health</h2>
      <pre className="card">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

function Table<T extends Record<string, unknown>>({
  rows,
  cols,
  actions,
}: {
  rows: T[];
  cols: string[];
  actions?: (row: T) => React.ReactNode;
}) {
  if (!rows.length) return <p className="muted">No records.</p>;
  return (
    <table className="table">
      <thead>
        <tr>
          {cols.map((c) => (
            <th key={c}>{c}</th>
          ))}
          {actions && <th />}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {cols.map((c) => (
              <td key={c}>{fmt(r[c])}</td>
            ))}
            {actions && <td>{actions(r)}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return String(v);
}

function ErrorBox({ msg }: { msg: string }) {
  return <div className="error">{msg}</div>;
}
