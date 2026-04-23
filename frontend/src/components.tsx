import { Link, Outlet } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export function AppShell() {
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary, refetchInterval: 15000 });
  const snapshotLabel = summary.data?.latest_snapshot_ts ? new Date(summary.data.latest_snapshot_ts).toLocaleString() : "Awaiting feed";
  const marketState = summary.data?.stale_data ? "Feed stale" : "Market live";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <span>ML</span>
          </div>
          <div>
            <span className="eyebrow">MLPM EXCHANGE</span>
            <h1>Market Desk</h1>
            <p className="brand-copy">MLB pricing, signal surveillance, and operator execution in one flow.</p>
          </div>
        </div>
        <nav className="nav">
          <Link to="/" className="nav-link" activeProps={{ className: "nav-link active" }}>
            Cockpit
          </Link>
          <Link to="/research" className="nav-link" activeProps={{ className: "nav-link active" }}>
            Research
          </Link>
          <Link to="/ops" className="nav-link" activeProps={{ className: "nav-link active" }}>
            Ops
          </Link>
        </nav>
        <div className="rail-section">
          <div className="rail-section-label">Desk pulse</div>
          <div className="sidebar-card compact">
            <div className="sidebar-label">Champion</div>
            <div className="sidebar-value">{summary.data?.champion_model ?? "Unknown"}</div>
          </div>
          <div className="sidebar-card compact">
            <div className="sidebar-label">Market state</div>
            <div className={`pill ${summary.data?.stale_data ? "warn" : "ok"}`}>{summary.data?.stale_data ? "Stale" : "Live"}</div>
          </div>
          <div className="sidebar-card compact">
            <div className="sidebar-label">Flagged quotes</div>
            <div className="sidebar-value">{summary.data?.flagged_discrepancies ?? 0}</div>
          </div>
        </div>
        <div className="rail-footer">
          <span>Snapshot</span>
          <strong>{snapshotLabel}</strong>
        </div>
      </aside>

      <main className="content">
        <header className="header">
          <div className="header-copy">
            <div className="header-strip">
              <span className="status-dot" />
              <span>{marketState}</span>
              <span className="header-strip-divider" />
              <span>Single-user trading workstation</span>
            </div>
            <div>
              <p className="eyebrow">MLB prediction market desk</p>
              <h2>Trading dashboard for signal selection and execution timing</h2>
            </div>
          </div>
          <div className="header-meta">
            <Stat label="Actionable bets" value={summary.data?.actionable_bets ?? "-"} />
            <Stat label="Max edge" value={summary.data ? `${summary.data.max_edge_bps} bps` : "-"} />
            <Stat label="Latest snapshot" value={snapshotLabel} />
          </div>
        </header>
        <Outlet />
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong className="mono">{value}</strong>
    </div>
  );
}
