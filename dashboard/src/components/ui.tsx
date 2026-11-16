import { useEffect, useState, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Menu, Moon, Sun, X } from 'lucide-react';
import type { Decision } from '../types';

const navigation = [
  { to: '/verify', label: 'Demo' },
  { to: '/pipeline', label: 'How it works' },
  { to: '/evaluation', label: 'Results' },
  { to: '/audit', label: 'Audit trail' },
  { to: '/about', label: 'About' },
];

export function SiteHeader({ githubUrl, dark, onToggleTheme }: { githubUrl: string; dark: boolean; onToggleTheme: () => void }) {
  const [menuOpen, setMenuOpen] = useState(false);
  return <header className="site-header">
    <div className="site-header-inner">
      <Link to="/" className="brand-lockup" aria-label="Capstone Project home" onClick={() => setMenuOpen(false)}>
        <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
        <span className="brand-name">Capstone Project</span>
      </Link>
      <button className="mobile-nav-toggle" type="button" onClick={() => setMenuOpen((open) => !open)} aria-expanded={menuOpen} aria-label={menuOpen ? 'Close navigation' : 'Open navigation'}>{menuOpen ? <X /> : <Menu />}</button>
      <nav className={`site-nav ${menuOpen ? 'site-nav-open' : ''}`} aria-label="Main navigation">
        {navigation.map((item) => <NavLink key={item.to} to={item.to} end={item.to === '/'} onClick={() => setMenuOpen(false)} className={({ isActive }) => `site-nav-link ${isActive ? 'site-nav-link-active' : ''}`}>{item.label}</NavLink>)}
      </nav>
      <div className="header-actions">
        <button className="theme-toggle" type="button" onClick={onToggleTheme} aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`} title={`Switch to ${dark ? 'light' : 'dark'} theme`}>{dark ? <Sun /> : <Moon />}</button>
        <a className="github-link" href={githubUrl} target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a>
      </div>
    </div>
  </header>;
}

export function SiteFooter({ githubUrl }: { githubUrl: string }) {
  return <footer className="site-footer">
    <div className="site-footer-inner">
      <div><strong>Capstone Project</strong><span>NIT Jamshedpur · Computer Science &amp; Engineering</span><span>Supervisor: Dr. Dilip Kumar</span></div>
      <nav aria-label="Project links"><a href={githubUrl} target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a><Link to="/about#team">Project team</Link></nav>
    </div>
  </footer>;
}

export function PageFrame({ children }: { children: ReactNode }) { return <div className="page-frame">{children}</div>; }
export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return <div className="page-header"><div><h1>{title}</h1>{description && <p>{description}</p>}</div>{action && <div className="page-header-action">{action}</div>}</div>;
}
export function Panel({ children, className = '', title, subtitle, action }: { children: ReactNode; className?: string; title?: string; subtitle?: string; action?: ReactNode }) {
  return <section className={`panel ${className}`}>
    {(title || subtitle || action) && <div className="panel-header"><div>{title && <h2>{title}</h2>}{subtitle && <p>{subtitle}</p>}</div>{action}</div>}
    {children}
  </section>;
}
export function SourceTag({ children }: { children: ReactNode }) { return <span className="source-tag">{children}</span>; }
export function MetricCard({ label, value, note, className = '' }: { label: string; value: string; note: string; className?: string }) {
  return <div className={`metric-card ${className}`}><span className="metric-top">{label}</span><strong className="metric-value"><CountUp value={value} /></strong><span className="metric-note">{note}</span></div>;
}
export function CountUp({ value }: { value: string }) {
  const [display, setDisplay] = useState(value);
  useEffect(() => {
    const match = value.match(/^(-?\d+(?:\.\d+)?)(.*)$/);
    if (!match || window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setDisplay(value); return; }
    const target = Number(match[1]);
    const suffix = match[2];
    const decimals = match[1].split('.')[1]?.length ?? 0;
    const duration = 480;
    const started = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(`${(target * eased).toFixed(decimals)}${suffix}`);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  return <span aria-label={value}>{display}</span>;
}
export function VerdictBadge({ decision, compact: _compact = false }: { decision: Decision; compact?: boolean }) {
  return <span className={`verdict verdict-${decision.toLowerCase()}`} aria-label={`Verdict: ${decision}`}><span className="verdict-dot" aria-hidden="true" /><span>{decision}</span></span>;
}
export function Skeleton() { return <div className="skeleton-block" aria-label="Loading content"><span /><span /><span /></div>; }
export function Button({ children, className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement>) { return <button className={`button ${className}`} {...props}>{children}</button>; }
