'use client';

import Link from 'next/link';
import {
  ChangeEvent,
  DragEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

type Box = [number, number, number, number];
type ViewMode = 'page' | 'overview' | 'module';

type InspectionRecord = {
  id: string;
  candidate_key: string;
  sequence: number;
  project: string | null;
  revision: string | null;
  author: string | null;
  drawing_date: string | null;
  module: string | null;
  module_index: number;
  page_index: number;
  fai: string | null;
  spc: string | null;
  description: string | null;
  nominal: string | null;
  usl: string | null;
  lsl: string | null;
  hundred_percent: string | null;
  dc: string | null;
  points: string | null;
  page_width: number;
  page_height: number;
  model_bbox: Box;
  effective_bbox: Box;
  user_override: boolean;
  crop_url: string;
  page_url: string;
  updated_at: string;
};

type Stage1Module = {
  bbox_pixels: Box;
  description?: string;
};

type Stage2Cluster = {
  recovery_key: string;
  fai_number: string | null;
  bbox_pixels: Box;
};

type Stage2Module = {
  module_index: number;
  module_bbox_full_image: Box;
  module_width: number;
  module_height: number;
  valid_cluster_count: number;
  valid_clusters: Stage2Cluster[];
  visualization_url: string;
};

type ProgressPage = {
  page_index: number;
  page_width: number;
  page_height: number;
  has_overview: boolean;
  overview_url: string;
  stage1_modules: Stage1Module[];
  stage2_modules: Record<string, Stage2Module>;
};

type ScanProgress = {
  stage1_module_count: number;
  candidate_count: number;
  refined_count: number;
  percent: number;
  pages: ProgressPage[];
};

type ScanRun = {
  id: string;
  pdf_name: string;
  status: 'queued' | 'rendering' | 'processing' | 'completed' | 'failed';
  total_pages: number;
  current_page: number;
  error: string | null;
  records: InspectionRecord[];
  progress: ScanProgress;
};

type Overlay = {
  key: string;
  box: Box;
  label: string;
  onClick: () => void;
};

const API_BASE = process.env.NEXT_PUBLIC_CLAW_VIEW_API ?? 'http://127.0.0.1:8002';
const apiUrl = (path: string) => `${API_BASE}${path}`;
const emptyCell = (value: string | number | null | undefined) => value ?? '';

function clampBox(box: Box, width: number, height: number): Box {
  const x1 = Math.max(0, Math.min(width - 8, Math.round(box[0])));
  const y1 = Math.max(0, Math.min(height - 8, Math.round(box[1])));
  const x2 = Math.max(x1 + 8, Math.min(width, Math.round(box[2])));
  const y2 = Math.max(y1 + 8, Math.min(height, Math.round(box[3])));
  return [x1, y1, x2, y2];
}

function InteractiveDrawing({
  imageUrl,
  alt,
  coordinateWidth,
  coordinateHeight,
  selection,
  selectionLabel,
  overlays,
  onSave,
}: {
  imageUrl: string;
  alt: string;
  coordinateWidth: number;
  coordinateHeight: number;
  selection: Box | null;
  selectionLabel?: string;
  overlays?: Overlay[];
  onSave?: (box: Box) => Promise<void>;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const draftRef = useRef<Box | null>(selection);
  const dragRef = useRef<{
    mode: 'move' | 'nw' | 'ne' | 'sw' | 'se';
    x: number;
    y: number;
    box: Box;
  } | null>(null);
  const [draft, setDraft] = useState<Box | null>(selection);
  const [saving, setSaving] = useState(false);

  const setDraftValue = (box: Box) => {
    draftRef.current = box;
    setDraft(box);
  };

  const beginDrag = (
    event: ReactPointerEvent<HTMLDivElement | HTMLSpanElement>,
    mode: 'move' | 'nw' | 'ne' | 'sw' | 'se',
  ) => {
    if (!draftRef.current) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { mode, x: event.clientX, y: event.clientY, box: draftRef.current };
  };

  const moveDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = dragRef.current;
    const layer = layerRef.current;
    if (!start || !layer) return;
    const rect = layer.getBoundingClientRect();
    const dx = (event.clientX - start.x) * coordinateWidth / rect.width;
    const dy = (event.clientY - start.y) * coordinateHeight / rect.height;
    let [x1, y1, x2, y2] = start.box;
    if (start.mode === 'move') {
      const width = x2 - x1;
      const height = y2 - y1;
      x1 = Math.max(0, Math.min(coordinateWidth - width, x1 + dx));
      y1 = Math.max(0, Math.min(coordinateHeight - height, y1 + dy));
      x2 = x1 + width;
      y2 = y1 + height;
    } else {
      if (start.mode.includes('w')) x1 = Math.max(0, Math.min(x2 - 8, x1 + dx));
      if (start.mode.includes('e')) x2 = Math.min(coordinateWidth, Math.max(x1 + 8, x2 + dx));
      if (start.mode.includes('n')) y1 = Math.max(0, Math.min(y2 - 8, y1 + dy));
      if (start.mode.includes('s')) y2 = Math.min(coordinateHeight, Math.max(y1 + 8, y2 + dy));
    }
    setDraftValue([x1, y1, x2, y2]);
  };

  const endDrag = async () => {
    if (!dragRef.current || !draftRef.current || !onSave) return;
    dragRef.current = null;
    const next = clampBox(draftRef.current, coordinateWidth, coordinateHeight);
    setDraftValue(next);
    setSaving(true);
    try {
      await onSave(next);
    } finally {
      setSaving(false);
    }
  };

  const percentStyle = (box: Box) => ({
    left: `${box[0] / coordinateWidth * 100}%`,
    top: `${box[1] / coordinateHeight * 100}%`,
    width: `${(box[2] - box[0]) / coordinateWidth * 100}%`,
    height: `${(box[3] - box[1]) / coordinateHeight * 100}%`,
  });

  return (
    <div ref={layerRef} className="drawing-image-layer">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imageUrl} alt={alt} draggable={false} />
      {overlays?.map((overlay) => (
        <button
          key={overlay.key}
          type="button"
          className="progress-click-box"
          style={percentStyle(overlay.box)}
          onClick={overlay.onClick}
        >
          <span>{overlay.label}</span>
        </button>
      ))}
      {draft && (
        <div
          className={`inline-crop-box ${saving ? 'is-saving' : ''}`}
          style={percentStyle(draft)}
          onPointerDown={(event) => beginDrag(event, 'move')}
          onPointerMove={moveDrag}
          onPointerUp={() => void endDrag()}
        >
          <span className="inline-crop-label">{saving ? '正在保存…' : `${selectionLabel ?? '截图'} · 拖动调整`}</span>
          {(['nw', 'ne', 'sw', 'se'] as const).map((corner) => (
            <span key={corner} className={`crop-handle crop-handle-${corner}`} onPointerDown={(event) => beginDrag(event, corner)} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function ScanPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [run, setRun] = useState<ScanRun | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('page');
  const [viewPage, setViewPage] = useState(0);
  const [viewModule, setViewModule] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [draggingFile, setDraggingFile] = useState(false);
  const [error, setError] = useState('');

  const refreshRun = useCallback(async (runId: string) => {
    const response = await fetch(apiUrl(`/api/runs/${runId}`), { cache: 'no-store' });
    if (!response.ok) throw new Error(await response.text());
    const next = await response.json() as ScanRun;
    setRun(next);
    setViewPage((current) => current || next.current_page || 1);
  }, []);

  useEffect(() => {
    if (!run || run.status === 'completed' || run.status === 'failed') return;
    const timer = window.setInterval(() => void refreshRun(run.id), 900);
    return () => window.clearInterval(timer);
  }, [refreshRun, run]);

  const uploadPdf = async (file?: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('请选择 PDF 文件');
      return;
    }
    setUploading(true);
    setError('');
    setSelectedId(null);
    setViewMode('page');
    try {
      const body = new FormData();
      body.append('pdf', file);
      const response = await fetch(apiUrl('/api/runs'), { method: 'POST', body });
      if (!response.ok) throw new Error(await response.text());
      const created = await response.json() as { run_id: string };
      await refreshRun(created.run_id);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'PDF 上传失败');
    } finally {
      setUploading(false);
    }
  };

  const activeRecord = useMemo(() => {
    if (!run?.records.length) return null;
    if (selectedId) return run.records.find((record) => record.id === selectedId) ?? null;
    return [...run.records].sort((a, b) => b.sequence - a.sequence)[0];
  }, [run, selectedId]);

  const progressPage = run?.progress.pages.find((page) => page.page_index === viewPage) ?? null;
  const activeModule = viewModule && progressPage ? progressPage.stage2_modules[String(viewModule)] : null;

  const saveRecord = useCallback(async (record: InspectionRecord, bbox: Box) => {
    if (!run) return;
    const response = await fetch(apiUrl(`/api/runs/${run.id}/records/${record.id}/crop`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox }),
    });
    if (!response.ok) throw new Error(await response.text());
    const updated = await response.json() as InspectionRecord;
    setRun((current) => current ? {
      ...current,
      records: current.records.map((item) => item.id === updated.id ? updated : item),
    } : current);
  }, [run]);

  const selectRecord = (record: InspectionRecord, mode: ViewMode = 'page') => {
    setSelectedId(record.id);
    setViewPage(record.page_index);
    setViewModule(record.module_index || null);
    setViewMode(mode);
  };

  const moduleOverlays: Overlay[] = progressPage?.stage1_modules.map((module, index) => ({
    key: `module-${index + 1}`,
    box: module.bbox_pixels,
    label: `MODULE ${index + 1}`,
    onClick: () => {
      setViewModule(index + 1);
      setViewMode('module');
    },
  })) ?? [];

  const clusterOverlays: Overlay[] = activeModule?.valid_clusters.map((cluster) => ({
    key: cluster.recovery_key,
    box: cluster.bbox_pixels,
    label: `FAI ${cluster.fai_number ?? '—'}`,
    onClick: () => {
      const record = run?.records.find((item) => item.page_index === viewPage && item.candidate_key === cluster.recovery_key);
      if (record) selectRecord(record, 'module');
    },
  })) ?? [];

  const moduleSelection = activeRecord && activeModule && activeRecord.page_index === viewPage && activeRecord.module_index === activeModule.module_index
    ? clampBox([
        activeRecord.effective_bbox[0] - activeModule.module_bbox_full_image[0],
        activeRecord.effective_bbox[1] - activeModule.module_bbox_full_image[1],
        activeRecord.effective_bbox[2] - activeModule.module_bbox_full_image[0],
        activeRecord.effective_bbox[3] - activeModule.module_bbox_full_image[1],
      ], activeModule.module_width, activeModule.module_height)
    : null;

  const currentPage = activeRecord && viewMode === 'page' ? activeRecord.page_index : viewPage || run?.current_page || 1;
  const progress = run?.progress;

  return (
    <main className="scan-page">
      <header className="scan-header">
        <Link className="brand" href="/"><span className="brand-mark" aria-hidden="true">CV</span><span>Claw View</span></Link>
        <div className="scan-progress" aria-live="polite">
          <span className={`status-dot status-${run?.status ?? 'idle'}`} />
          {run ? <><strong>{run.status === 'completed' ? '处理完成' : run.status === 'failed' ? '处理失败' : '正在处理'}</strong><span>第 {run.current_page || 0}/{run.total_pages || '—'} 页</span><span className="progress-file">{run.pdf_name}</span></> : <span>等待 PDF</span>}
        </div>
        <button className="upload-compact" type="button" onClick={() => inputRef.current?.click()}>{uploading ? '上传中…' : '选择 PDF'}</button>
        <input ref={inputRef} className="visually-hidden" type="file" accept="application/pdf,.pdf" onChange={(event: ChangeEvent<HTMLInputElement>) => void uploadPdf(event.target.files?.[0])} />
      </header>

      <section className="scan-workspace">
        <aside className="inspection-panel">
          <div className="panel-heading"><div><span className="panel-kicker">INSPECTION INDEX</span><h1>检验信息</h1></div><span className="record-count">{run?.records.length ?? 0} 条</span></div>
          <div className="inspection-table-wrap"><table className="inspection-table">
            <thead><tr><th>No.</th><th>Project</th><th>Revision</th><th>Author</th><th>Date</th><th>Module</th><th>Page</th><th>FAI</th><th>SPC</th><th>Description</th><th>Nominal</th><th>USL</th><th>LSL</th><th>100%</th><th>DC</th><th>Points</th><th>SPC截图</th></tr></thead>
            <tbody>
              {run?.records.map((record, index) => (
                <tr key={record.id} className={activeRecord?.id === record.id ? 'is-selected' : ''} onClick={() => selectRecord(record)}>
                  <td>{index + 1}</td><td>{emptyCell(record.project)}</td><td>{emptyCell(record.revision)}</td><td>{emptyCell(record.author)}</td><td>{emptyCell(record.drawing_date)}</td><td>{emptyCell(record.module)}</td><td>page {record.page_index}</td><td className="fai-value">{emptyCell(record.fai)}</td><td>{emptyCell(record.spc)}</td><td className="description-cell">{emptyCell(record.description)}</td><td>{emptyCell(record.nominal)}</td><td>{emptyCell(record.usl)}</td><td>{emptyCell(record.lsl)}</td><td>{emptyCell(record.hundred_percent)}</td><td>{emptyCell(record.dc)}</td><td>{emptyCell(record.points)}</td>
                  <td className="thumbnail-cell">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`${apiUrl(record.crop_url)}?v=${encodeURIComponent(record.updated_at)}`} alt={`FAI ${record.fai ?? ''} 截图`} />
                    {record.user_override && <span>已修改</span>}
                  </td>
                </tr>
              ))}
              {!run?.records.length && <tr className="empty-table-row"><td colSpan={17}>crop2_refined 完成后，结果会按 FAI 从小到大出现在这里</td></tr>}
            </tbody>
          </table></div>
        </aside>

        <section className={`drawing-panel ${draggingFile ? 'is-file-dragging' : ''}`} onDragOver={(event: DragEvent<HTMLElement>) => { event.preventDefault(); setDraggingFile(true); }} onDragLeave={() => setDraggingFile(false)} onDrop={(event: DragEvent<HTMLElement>) => { event.preventDefault(); setDraggingFile(false); void uploadPdf(event.dataTransfer.files?.[0]); }}>
          <div className="drawing-toolbar">
            <div><span>{viewMode === 'overview' ? 'STAGE 1 OVERVIEW' : viewMode === 'module' ? `MODULE ${viewModule}` : 'PDF PAGE'}</span><strong>{run ? `${currentPage} / ${run.total_pages || '—'}` : '— / —'}</strong></div>
            {run && <div className="progress-summary">
              <span>模块 <b>{progress?.stage1_module_count ?? 0}</b></span>
              <span>FAI <b>{progress?.candidate_count ?? 0}</b></span>
              <span>REFINED <b>{progress?.refined_count ?? 0}</b></span>
              <span className="progress-percent"><i style={{ width: `${progress?.percent ?? 0}%` }} /><b>{progress?.percent ?? 0}%</b></span>
              <button type="button" onClick={() => { setViewPage(run.current_page || 1); setViewMode('overview'); }}>查看进度</button>
              {viewMode !== 'page' && <button type="button" onClick={() => setViewMode('page')}>返回原图</button>}
            </div>}
          </div>

          {!run ? (
            <button className="pdf-dropzone" type="button" onClick={() => inputRef.current?.click()}><span className="dropzone-symbol">＋</span><strong>拖拽 PDF 到这里</strong><small>或从本地目录选择工程图纸</small></button>
          ) : run.status === 'rendering' && !run.total_pages ? (
            <div className="viewer-message"><span className="loader-ring" /><strong>正在渲染 PDF 页面</strong></div>
          ) : (
            <div className="drawing-scroll">
              {viewMode === 'overview' && progressPage?.has_overview ? (
                <InteractiveDrawing key={`overview-${viewPage}`} imageUrl={apiUrl(progressPage.overview_url)} alt={`第 ${viewPage} 页 Stage 1 模块总览`} coordinateWidth={progressPage.page_width} coordinateHeight={progressPage.page_height} selection={null} overlays={moduleOverlays} />
              ) : viewMode === 'module' && activeModule ? (
                <InteractiveDrawing
                  key={`module-${viewPage}-${activeModule.module_index}-${activeRecord?.id ?? 'none'}-${activeRecord?.updated_at ?? ''}`}
                  imageUrl={apiUrl(activeModule.visualization_url)}
                  alt={`模块 ${activeModule.module_index} FAI 总览`}
                  coordinateWidth={activeModule.module_width}
                  coordinateHeight={activeModule.module_height}
                  selection={moduleSelection}
                  selectionLabel={activeRecord ? `FAI ${activeRecord.fai ?? '—'}` : undefined}
                  overlays={clusterOverlays}
                  onSave={activeRecord && moduleSelection ? async (local) => saveRecord(activeRecord, [local[0] + activeModule.module_bbox_full_image[0], local[1] + activeModule.module_bbox_full_image[1], local[2] + activeModule.module_bbox_full_image[0], local[3] + activeModule.module_bbox_full_image[1]]) : undefined}
                />
              ) : (
                <InteractiveDrawing
                  key={`page-${currentPage}-${activeRecord?.id ?? 'none'}-${activeRecord?.updated_at ?? ''}`}
                  imageUrl={apiUrl(`/api/runs/${run.id}/pages/${currentPage}`)}
                  alt={`PDF 第 ${currentPage} 页`}
                  coordinateWidth={activeRecord?.page_width ?? progressPage?.page_width ?? 1}
                  coordinateHeight={activeRecord?.page_height ?? progressPage?.page_height ?? 1}
                  selection={activeRecord?.page_index === currentPage ? activeRecord.effective_bbox : null}
                  selectionLabel={activeRecord ? `FAI ${activeRecord.fai ?? '—'}` : undefined}
                  onSave={activeRecord?.page_index === currentPage ? async (box) => saveRecord(activeRecord, box) : undefined}
                />
              )}
            </div>
          )}
          {draggingFile && <div className="drop-overlay">松开以开始扫描 PDF</div>}
          {error && <div className="scan-error">{error}</div>}
          {run?.error && <div className="scan-error">{run.error}</div>}
        </section>
      </section>
    </main>
  );
}
