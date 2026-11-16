import { lazy, Suspense, useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import { X } from 'lucide-react';
import { loadDatasets } from './lib/api';
import type { DataBundle } from './types';
import { PageFrame, SiteFooter, SiteHeader, Skeleton } from './components/ui';

const HomePage = lazy(() => import('./pages/HomePage').then((module) => ({ default: module.HomePage })));
const VerifyPage = lazy(() => import('./pages/VerifyPage').then((module) => ({ default: module.VerifyPage })));
const PipelinePage = lazy(() => import('./pages/PipelinePage').then((module) => ({ default: module.PipelinePage })));
const EvaluationPage = lazy(() => import('./pages/EvaluationPage').then((module) => ({ default: module.EvaluationPage })));
const AuditPage = lazy(() => import('./pages/AuditPage').then((module) => ({ default: module.AuditPage })));
const AboutPage = lazy(() => import('./pages/AboutPage').then((module) => ({ default: module.AboutPage })));

const githubUrl = 'https://github.com/HanuP69/Capstone-Project';
const benchmarkNotice = 'Results are from a synthetic benchmark built from NIT Jamshedpur ordinances. Live circular ingestion is future work.';

export default function App() {
  const [data, setData] = useState<DataBundle | null>(null);
  const [loadError, setLoadError] = useState('');
  const [noticeVisible, setNoticeVisible] = useState(true);
  const [dark, setDark] = useState(() => document.documentElement.dataset.theme === 'dark');

  const refreshData = () => {
    setLoadError('');
    loadDatasets().then(setData).catch((error: Error) => setLoadError(error.message));
  };
  useEffect(() => { refreshData(); }, []);
  useEffect(() => {
    const theme = dark ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem('capstone-theme', theme); } catch { /* Theme still applies for this page view. */ }
  }, [dark]);

  const pageContent = !data ? (
    loadError ? <PageFrame><div className="error-card" role="alert"><div><h1>Project data could not be loaded</h1><p>{loadError}</p></div><button className="button button-primary" onClick={refreshData}>Try again</button></div></PageFrame> :
    <PageFrame><Skeleton /><Skeleton /><Skeleton /></PageFrame>
  ) : (
    <Suspense fallback={<PageFrame><Skeleton /><Skeleton /></PageFrame>}>
      <Routes>
        <Route path="/" element={<HomePage data={data} />} />
        <Route path="/verify" element={<VerifyPage data={data} />} />
        <Route path="/pipeline" element={<PipelinePage />} />
        <Route path="/evaluation" element={<EvaluationPage data={data} />} />
        <Route path="/audit" element={<AuditPage data={data} />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<HomePage data={data} />} />
      </Routes>
    </Suspense>
  );

  return <div className="site-shell">
    <SiteHeader githubUrl={githubUrl} dark={dark} onToggleTheme={() => setDark((value) => !value)} />
    {noticeVisible && <div className="notice-wrap"><aside className="benchmark-notice" role="note"><span>{benchmarkNotice}</span><button type="button" className="notice-close" aria-label="Dismiss benchmark notice" onClick={() => setNoticeVisible(false)}><X /></button></aside></div>}
    <main className="content-area">{pageContent}</main>
    <SiteFooter githubUrl={githubUrl} />
  </div>;
}
