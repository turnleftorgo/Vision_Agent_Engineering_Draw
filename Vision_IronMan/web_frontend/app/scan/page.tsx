'use client';

import Link from 'next/link';
import { createPortal } from 'react-dom';
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
type EditableField = 'project' | 'revision' | 'author' | 'drawing_date' | 'module' | 'fai' | 'spc' | 'description' | 'nominal' | 'usl' | 'lsl' | 'hundred_percent' | 'dc' | 'points';

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
  marker_bbox: Box | null;
  effective_bbox: Box;
  user_override: boolean;
  crop_url: string;
  page_url: string;
  updated_at: string;
  extraction_status: 'not_started' | 'queued' | 'processing' | 'completed' | 'needs_review' | 'failed';
  extraction_error: string | null;
  user_edited_fields?: string[];
};

type Stage1Module = {
  bbox_pixels: Box;
  description?: string;
  crop_url?: string;
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
  active_refinements: { candidate_key: string; module_index: number; fai_number: string | null; round_number: number; bbox: Box }[];
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
  status: 'queued' | 'rendering' | 'processing' | 'completed' | 'failed' | 'interrupted';
  total_pages: number;
  current_page: number;
  current_module: number;
  error: string | null;
  records: InspectionRecord[];
  progress: ScanProgress;
};

type Overlay = {
  key: string;
  box: Box;
  label: string;
  point?: [number, number];
  kind?: 'box' | 'circle';
  selected?: boolean;
  onClick: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onZoom?: () => void;
  hitboxOnly?: boolean;
};

type FocusRequest = { scopeKey: string; box: Box };

const API_BASE = process.env.NEXT_PUBLIC_CLAW_VIEW_API ?? 'http://127.0.0.1:8003';
const ACTIVE_RUN_STORAGE_KEY = 'claw-view.active-run.v1';
const apiUrl = (path: string) => `${API_BASE}${path}`;
const emptyCell = (value: string | number | null | undefined) => value ?? '';

type ActiveRunPointer = { run_id: string; backend_instance_id: string };
type ApiError = Error & { status?: number };

function EditableCell({ value, onSave, className = '' }: {
  value: string | number | null | undefined;
  onSave: (value: string) => Promise<void>;
  className?: string;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(value ?? ''));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const skipBlurSave = useRef(false);
  const save = async () => {
    setSaving(true);
    setError('');
    try {
      await onSave(draft);
      setEditing(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : '保存失败');
    } finally {
      setSaving(false);
    }
  };
  return editing ? (
    <input
      className={`editable-cell-input ${className}`}
      value={draft}
      autoFocus
      disabled={saving}
      aria-label="编辑单元格"
      onClick={(event) => event.stopPropagation()}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => {
        if (skipBlurSave.current) { skipBlurSave.current = false; return; }
        if (!saving) void save();
      }}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') { skipBlurSave.current = true; setDraft(String(value ?? '')); setEditing(false); }
      }}
      title={error || undefined}
    />
  ) : (
    <button
      type="button"
      className={`editable-cell-value ${className}`}
      onClick={(event) => { event.stopPropagation(); setDraft(String(value ?? '')); setError(''); setEditing(true); }}
      title={error || '点击编辑'}
    >{saving ? '保存中…' : emptyCell(value) || <span className="editable-placeholder">—</span>}</button>
  );
}

function CropPreview({ record, anchor, onClose }: {
  record: InspectionRecord;
  anchor: HTMLElement;
  onClose: () => void;
}) {
  const previewRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  useEffect(() => {
    const updatePosition = () => {
      const preview = previewRef.current;
      if (!preview) return;
      if (!anchor.isConnected) { onClose(); return; }
      const anchorRect = anchor.getBoundingClientRect();
      const rowRect = (anchor.closest('tr') ?? anchor).getBoundingClientRect();
      const tableViewport = anchor.closest('.inspection-table-scroll')?.getBoundingClientRect();
      if (tableViewport && (rowRect.bottom <= tableViewport.top || rowRect.top >= tableViewport.bottom)) {
        onClose();
        return;
      }
      const gap = 8;
      const edge = 12;
      const headerHeight = preview.querySelector('.crop-preview-header')?.getBoundingClientRect().height ?? 0;
      const imageMaxHeight = Math.max(0, rowRect.top - gap - edge - headerHeight - 18);
      preview.style.setProperty('--crop-preview-image-max-height', `${imageMaxHeight}px`);
      const previewRect = preview.getBoundingClientRect();
      const left = Math.max(edge, Math.min(
        window.innerWidth - previewRect.width - edge,
        anchorRect.left + anchorRect.width / 2 - previewRect.width / 2,
      ));
      const top = Math.max(edge, rowRect.top - previewRect.height - gap);
      setPosition({ left, top });
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && (previewRef.current?.contains(target) || anchor.contains(target))) return;
      if (target instanceof Element && target.closest('.drawing-panel')) return;
      onClose();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    const observer = new ResizeObserver(updatePosition);
    if (previewRef.current) observer.observe(previewRef.current);
    observer.observe(anchor);
    const table = anchor.closest('table');
    if (table) observer.observe(table);
    updatePosition();
    window.addEventListener('resize', updatePosition);
    document.addEventListener('scroll', updatePosition, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updatePosition);
      document.removeEventListener('scroll', updatePosition, true);
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [anchor, onClose, record.updated_at]);

  return createPortal(
    <div
      ref={previewRef}
      className="crop-preview"
      role="dialog"
      aria-label={`FAI ${record.fai ?? '—'} 当前完整截图`}
      style={{ left: position?.left ?? 0, top: position?.top ?? 0, visibility: position ? 'visible' : 'hidden' }}
    >
      <div className="crop-preview-header">
        <strong>FAI {record.fai ?? '—'} · 当前完整截图</strong>
        <button type="button" aria-label="关闭截图预览" onClick={onClose}>×</button>
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`${apiUrl(record.crop_url)}?v=${encodeURIComponent(record.updated_at)}`} alt={`FAI ${record.fai ?? '—'} 当前完整截图`} />
    </div>,
    document.body,
  );
}

function clampBox(box: Box, width: number, height: number): Box {
  const x1 = Math.max(0, Math.min(width - 8, Math.round(box[0])));
  const y1 = Math.max(0, Math.min(height - 8, Math.round(box[1])));
  const x2 = Math.max(x1 + 8, Math.min(width, Math.round(box[2])));
  const y2 = Math.max(y1 + 8, Math.min(height, Math.round(box[3])));
  return [x1, y1, x2, y2];
}

function focusWindow(box: Box, width: number, height: number): Box {
  const focusWidth = Math.min(width, Math.max(8, (box[2] - box[0]) * 3));
  const focusHeight = Math.min(height, Math.max(8, (box[3] - box[1]) * 3));
  const centerX = (box[0] + box[2]) / 2;
  const centerY = (box[1] + box[3]) / 2;
  const x1 = Math.round(Math.max(0, Math.min(width - focusWidth, centerX - focusWidth / 2)));
  const y1 = Math.round(Math.max(0, Math.min(height - focusHeight, centerY - focusHeight / 2)));
  return [x1, y1, x1 + focusWidth, y1 + focusHeight];
}

function InteractiveDrawing({
  imageUrl,
  alt,
  coordinateWidth,
  coordinateHeight,
  selection,
  selectionLabel,
  overlays,
  refiningFrames,
  onSave,
  scopeKey,
  focusRequest,
  onRequestFocus,
  onClearFocus,
  onDraftChange,
  onSaveError,
  hideFocusControls = false,
  syncSelection = false,
  drawMode = false,
  onDrawComplete,
}: {
  imageUrl: string;
  alt: string;
  coordinateWidth: number;
  coordinateHeight: number;
  selection: Box | null;
  selectionLabel?: string;
  overlays?: Overlay[];
  refiningFrames?: { key: string; box: Box }[];
  onSave?: (box: Box) => Promise<void>;
  scopeKey: string;
  focusRequest: FocusRequest | null;
  onRequestFocus: (box: Box, scopeKey: string) => void;
  onClearFocus: () => void;
  onDraftChange?: (box: Box) => void;
  onSaveError?: (error: unknown) => void;
  hideFocusControls?: boolean;
  syncSelection?: boolean;
  drawMode?: boolean;
  onDrawComplete?: (box: Box) => void;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draftRef = useRef<Box | null>(selection);
  const dragRef = useRef<{
    mode: 'move' | 'nw' | 'ne' | 'sw' | 'se';
    x: number;
    y: number;
    box: Box;
  } | null>(null);
  const [draft, setDraft] = useState<Box | null>(selection);
  const [saving, setSaving] = useState(false);
  const [focusBounds, setFocusBounds] = useState<Box | null>(() => focusRequest?.scopeKey === scopeKey ? focusWindow(focusRequest.box, coordinateWidth, coordinateHeight) : null);
  const [imageReady, setImageReady] = useState(false);
  const [drawBox, setDrawBox] = useState<Box | null>(null);
  const drawStartRef = useRef<[number, number] | null>(null);

  useEffect(() => {
    if (!syncSelection) return;
    if (dragRef.current) return;
    draftRef.current = selection;
    setDraft(selection);
  }, [selection, syncSelection]);

  const displayBounds = useMemo<Box>(
    () => focusBounds ?? [0, 0, coordinateWidth, coordinateHeight],
    [coordinateHeight, coordinateWidth, focusBounds],
  );
  const displayWidth = Math.max(1, displayBounds[2] - displayBounds[0]);
  const displayHeight = Math.max(1, displayBounds[3] - displayBounds[1]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!focusBounds || !imageReady || !canvas || !image?.naturalWidth || !image.naturalHeight) return;
    const pixelRatio = Math.min(1200 / displayWidth, 900 / displayHeight);
    canvas.width = Math.max(1, Math.round(displayWidth * pixelRatio));
    canvas.height = Math.max(1, Math.round(displayHeight * pixelRatio));
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(
      image,
      displayBounds[0], displayBounds[1], displayWidth, displayHeight,
      0, 0, canvas.width, canvas.height,
    );
  }, [displayBounds, displayHeight, displayWidth, focusBounds, imageReady]);

  const setDraftValue = (box: Box) => {
    draftRef.current = box;
    setDraft(box);
    onDraftChange?.(box);
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
    const dx = (event.clientX - start.x) * displayWidth / rect.width;
    const dy = (event.clientY - start.y) * displayHeight / rect.height;
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
    } catch (error) {
      if (onSaveError) onSaveError(error);
      else throw error;
    } finally {
      setSaving(false);
    }
  };

  const percentStyle = (box: Box) => ({
    left: `${(box[0] - displayBounds[0]) / displayWidth * 100}%`,
    top: `${(box[1] - displayBounds[1]) / displayHeight * 100}%`,
    width: `${(box[2] - box[0]) / displayWidth * 100}%`,
    height: `${(box[3] - box[1]) / displayHeight * 100}%`,
  });

  const pointStyle = (point: [number, number]) => ({
    left: `${(point[0] - displayBounds[0]) / displayWidth * 100}%`,
    top: `${(point[1] - displayBounds[1]) / displayHeight * 100}%`,
  });

  const pointerToImage = (event: ReactPointerEvent<HTMLDivElement>): [number, number] => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    return [
      displayBounds[0] + (event.clientX - rect.left) / rect.width * displayWidth,
      displayBounds[1] + (event.clientY - rect.top) / rect.height * displayHeight,
    ];
  };
  const beginDraw = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drawMode || !onDrawComplete || (event.target as HTMLElement).closest('button, .inline-crop-box')) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointerToImage(event);
    drawStartRef.current = point;
    setDrawBox([point[0], point[1], point[0], point[1]]);
  };
  const moveDraw = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drawStartRef.current) return;
    const [sx, sy] = drawStartRef.current;
    const [x, y] = pointerToImage(event);
    setDrawBox([Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]);
  };
  const finishDraw = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drawStartRef.current) return;
    const [sx, sy] = drawStartRef.current;
    const [x, y] = pointerToImage(event);
    drawStartRef.current = null;
    const next = clampBox([Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)], coordinateWidth, coordinateHeight);
    setDrawBox(null);
    if (next[2] - next[0] >= 8 && next[3] - next[1] >= 8) onDrawComplete?.(next);
  };

  const openFocus = (box: Box) => {
    setFocusBounds(focusWindow(box, coordinateWidth, coordinateHeight));
    onRequestFocus(box, scopeKey);
  };

  return (
    <div ref={layerRef} className={`drawing-image-layer ${focusBounds ? 'is-focused' : ''} ${drawMode ? 'is-drawing' : ''}`} onPointerDown={beginDraw} onPointerMove={moveDraw} onPointerUp={finishDraw} onPointerCancel={() => { drawStartRef.current = null; setDrawBox(null); }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={imageRef} className={focusBounds ? 'drawing-source-image is-hidden' : 'drawing-source-image'} src={imageUrl} alt={alt} draggable={false} onLoad={() => setImageReady(true)} />
      {focusBounds && <canvas ref={canvasRef} className="drawing-focus-canvas" aria-label={`${alt} 局部放大`} />}
      {focusBounds && !hideFocusControls && <button type="button" className="focus-return" onClick={() => { setFocusBounds(null); onClearFocus(); }}>返回全图</button>}
      {refiningFrames?.map((frame) => (
        <div key={frame.key} className="refinement-frame" style={percentStyle(frame.box)} aria-hidden="true" />
      ))}
      {overlays?.map((overlay) => (
        overlay.kind === 'circle' && overlay.point ? (
          <div key={overlay.key} className="audit-marker-position" style={pointStyle(overlay.point)}>
            <button
              type="button"
              className={`audit-marker ${overlay.selected ? 'is-selected' : ''}`}
              aria-label={`审核 FAI ${overlay.label}`}
              aria-pressed={overlay.selected}
              onMouseEnter={overlay.onMouseEnter}
              onMouseLeave={overlay.onMouseLeave}
              onFocus={overlay.onMouseEnter}
              onBlur={overlay.onMouseLeave}
              onClick={overlay.onClick}
            >{overlay.label}</button>
          </div>
        ) : (
          <div
            key={overlay.key}
            className={`progress-overlay ${overlay.hitboxOnly ? 'overview-module-hitbox' : ''}`}
            style={percentStyle(overlay.box)}
          >
            <button type="button" className={`progress-click-box ${overlay.hitboxOnly ? 'overview-hitbox' : ''}`} aria-label={overlay.hitboxOnly ? `打开 ${overlay.label}` : `选择 ${overlay.label}`} title={overlay.hitboxOnly ? `打开 ${overlay.label}` : undefined} onClick={overlay.onClick}>
              <span>{overlay.label}</span>
            </button>
            {!focusBounds && !overlay.hitboxOnly && <button type="button" className="frame-magnifier" aria-label={`放大 ${overlay.label}`} title="放大框选区域" onClick={(event) => { event.stopPropagation(); overlay.onZoom?.(); }}>⌕</button>}
          </div>
        )
      ))}
      {drawBox && <div className="audit-new-box" style={percentStyle(drawBox)} aria-label="手动画框预览" />}
      {draft && !drawMode && (
        <div
          className={`inline-crop-box ${saving ? 'is-saving' : ''}`}
          style={percentStyle(draft)}
          onPointerDown={(event) => beginDrag(event, 'move')}
          onPointerMove={moveDrag}
          onPointerUp={() => void endDrag()}
        >
          <span className="inline-crop-label">{saving ? '正在保存…' : `${selectionLabel ?? '截图'} · 拖动调整`}</span>
          {!focusBounds && !hideFocusControls && <button type="button" className="frame-magnifier selected-frame-magnifier" aria-label="放大当前截图框" title="放大框选区域" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); openFocus(draft); }}>⌕</button>}
          {(['nw', 'ne', 'sw', 'se'] as const).map((corner) => (
            <span key={corner} className={`crop-handle crop-handle-${corner}`} onPointerDown={(event) => beginDrag(event, corner)} />
          ))}
        </div>
      )}
    </div>
  );
}

function AuditCropCanvas({ imageUrl, bbox, markerBox, pageWidth, pageHeight, label }: {
  imageUrl: string;
  bbox: Box;
  markerBox: Box | null;
  pageWidth: number;
  pageHeight: number;
  label: string;
}) {
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageReady, setImageReady] = useState(false);

  useEffect(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!imageReady || !image?.naturalWidth || !canvas) return;
    const [x1, y1, x2, y2] = clampBox(bbox, pageWidth, pageHeight);
    const cropWidth = x2 - x1;
    const cropHeight = y2 - y1;
    const scale = Math.min(1, 1600 / cropWidth, 1100 / cropHeight);
    canvas.width = Math.max(1, Math.round(cropWidth * scale));
    canvas.height = Math.max(1, Math.round(cropHeight * scale));
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, x1, y1, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
    if (markerBox) {
      const [mx1, my1, mx2, my2] = clampBox(markerBox, pageWidth, pageHeight);
      const markerPad = Math.max(3, Math.round(Math.max(mx2 - mx1, my2 - my1) * 0.08));
      const left = Math.max(0, mx1 - x1 - markerPad);
      const top = Math.max(0, my1 - y1 - markerPad);
      const right = Math.min(cropWidth, mx2 - x1 + markerPad);
      const bottom = Math.min(cropHeight, my2 - y1 + markerPad);
      if (left < right && top < bottom) {
        context.strokeStyle = 'rgb(220, 0, 0)';
        context.lineWidth = Math.max(2, Math.round(Math.min(cropWidth, cropHeight) * 0.004)) * scale;
        context.strokeRect(left * scale, top * scale, (right - left) * scale, (bottom - top) * scale);
      }
    }
  }, [bbox, imageReady, markerBox, pageHeight, pageWidth]);

  return (
    <div className="audit-crop-canvas-wrap">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={imageRef} className="audit-crop-source" src={imageUrl} alt="" aria-hidden="true" draggable={false} onLoad={() => setImageReady(true)} />
      <canvas ref={canvasRef} className="audit-crop-canvas" aria-label={label} />
    </div>
  );
}

function AuditPanel({
  record,
  pageImageUrl,
  bbox,
  onDraftChange,
  onSave,
  onSaveError,
  onUpdateField,
  error,
}: {
  record: InspectionRecord;
  pageImageUrl: string;
  bbox: Box;
  onDraftChange: (bbox: Box) => void;
  onSave: (bbox: Box) => Promise<void>;
  onSaveError: (error: unknown) => void;
  onUpdateField: (field: EditableField, value: string) => Promise<void>;
  error: string;
}) {
  const auditScopeKey = `audit-${record.id}`;
  const fields: { label: string; field: EditableField | null; value: string | null }[] = [
    { label: 'MODULE', field: 'module', value: record.module },
    { label: 'PAGE', field: null, value: `page ${record.page_index}` },
    { label: 'FAI', field: 'fai', value: record.fai },
    { label: 'SPC', field: 'spc', value: record.spc },
    { label: 'DESCRIPTION', field: 'description', value: record.description },
    { label: 'NOMINAL', field: 'nominal', value: record.nominal },
    { label: 'USL', field: 'usl', value: record.usl },
    { label: 'LSL', field: 'lsl', value: record.lsl },
    { label: '100%', field: 'hundred_percent', value: record.hundred_percent },
    { label: 'DC', field: 'dc', value: record.dc },
    { label: 'POINTS', field: 'points', value: record.points },
  ];

  return (
    <div className="audit-panel">
      <div className="audit-record-header">
        <div><span className="panel-kicker">AUDIT MODE · PAGE {record.page_index}</span><h1>FAI {record.fai ?? '—'}</h1></div>
        <span className={`audit-save-state ${error ? 'has-error' : ''}`}>{error || (record.user_override ? '人工截图' : '模型截图')}</span>
      </div>
      <div className="audit-fields" aria-label={`FAI ${record.fai ?? '—'} 检验字段`}>
        {fields.map(({ label, field, value }) => {
          const modelFilling = Boolean(field && record.extraction_status === 'processing' && !value && !record.user_edited_fields?.includes(field));
          return <div key={label} className={`audit-field audit-field-${field ?? 'page'} ${modelFilling ? 'is-model-filling' : ''}`}>
            <span>{label}</span>
            {field
              ? <EditableCell value={value} onSave={(next) => onUpdateField(field, next)} />
              : <strong>{value}</strong>}
          </div>;
        })}
      </div>
      <div className="audit-images">
        <section className="audit-image-card">
          <header><strong>FAI 截图</strong><span>完整选区 · 拖动右图框实时更新</span></header>
          <AuditCropCanvas imageUrl={pageImageUrl} bbox={bbox} markerBox={record.marker_bbox} pageWidth={record.page_width} pageHeight={record.page_height} label={`FAI ${record.fai ?? '—'} 当前截图`} />
        </section>
        <section className="audit-image-card audit-zoom-card">
          <header><strong>3× 局部放大</strong><span>拖动边框或四角调整截图</span></header>
          <div className="audit-zoom-editor">
            <InteractiveDrawing
              key={`audit-editor-${record.id}`}
              imageUrl={pageImageUrl}
              alt={`FAI ${record.fai ?? '—'} 局部放大截图编辑`}
              coordinateWidth={record.page_width}
              coordinateHeight={record.page_height}
              selection={bbox}
              selectionLabel={`FAI ${record.fai ?? '—'}`}
              overlays={[]}
              onSave={onSave}
              onDraftChange={onDraftChange}
              onSaveError={onSaveError}
              syncSelection
              scopeKey={auditScopeKey}
              focusRequest={{ scopeKey: auditScopeKey, box: record.effective_bbox }}
              onRequestFocus={() => undefined}
              onClearFocus={() => undefined}
              hideFocusControls
            />
          </div>
        </section>
      </div>
    </div>
  );
}

function AuditFaiSelector({ records, selectedId, onSelect }: {
  records: InspectionRecord[];
  selectedId: string;
  onSelect: (record: InspectionRecord) => void;
}) {
  return <div className="audit-fai-selector" aria-label="切换审核 FAI">
    {records.map((record) => <button
      key={record.id}
      type="button"
      className={record.id === selectedId ? 'is-selected' : ''}
      aria-pressed={record.id === selectedId}
      onClick={() => onSelect(record)}
    >{record.fai ?? '—'}</button>)}
  </div>;
}

export default function ScanPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const [run, setRun] = useState<ScanRun | null>(null);
  const [isRestoring, setIsRestoring] = useState(true);
  const [auditMode, setAuditMode] = useState(false);
  const [hoveredAuditId, setHoveredAuditId] = useState<string | null>(null);
  const [lockedAuditId, setLockedAuditId] = useState<string | null>(null);
  const lockedAuditIdRef = useRef<string | null>(null);
  const [auditDraft, setAuditDraft] = useState<{ recordId: string; bbox: Box } | null>(null);
  const [manualPendingRecords, setManualPendingRecords] = useState<InspectionRecord[]>([]);
  const [manualDrawMode, setManualDrawMode] = useState<'page' | 'module' | null>(null);
  const manualEditsRef = useRef<Record<string, Partial<InspectionRecord>>>({});
  const manualCounterRef = useRef(0);
  const [auditError, setAuditError] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [previewRecordId, setPreviewRecordId] = useState<string | null>(null);
  const [previewAnchor, setPreviewAnchor] = useState<HTMLElement | null>(null);
  const [pageFrameRequested, setPageFrameRequested] = useState(false);
  const [focusRequest, setFocusRequest] = useState<FocusRequest | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('page');
  const [viewPage, setViewPage] = useState(0);
  const [viewModule, setViewModule] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [draggingFile, setDraggingFile] = useState(false);
  const [error, setError] = useState('');
  const [exporting, setExporting] = useState(false);
  const [editError, setEditError] = useState('');
  const [leftWidth, setLeftWidth] = useState(60);
  const resizeRef = useRef<{ startX: number; startWidth: number } | null>(null);
  useEffect(() => { lockedAuditIdRef.current = lockedAuditId; }, [lockedAuditId]);
  const closePreview = useCallback(() => {
    setPreviewRecordId(null);
    setPreviewAnchor(null);
  }, []);

  const refreshRun = useCallback(async (runId: string) => {
    const response = await fetch(apiUrl(`/api/runs/${runId}`), { cache: 'no-store' });
    if (!response.ok) {
      const error = new Error(await response.text()) as ApiError;
      error.status = response.status;
      throw error;
    }
    const next = await response.json() as ScanRun;
    if (activeRunIdRef.current === runId) {
      setRun(next);
      setViewPage((current) => current || next.current_page || 1);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const parsePointer = (): ActiveRunPointer | null => {
      try {
        const value = JSON.parse(window.localStorage.getItem(ACTIVE_RUN_STORAGE_KEY) ?? 'null') as ActiveRunPointer | null;
        return value?.run_id && value.backend_instance_id ? value : null;
      } catch {
        window.localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
        return null;
      }
    };
    const pointer = parsePointer();
    activeRunIdRef.current = pointer?.run_id ?? null;

    const restore = async () => {
      let delay = 500;
      while (!cancelled) {
        try {
          const healthResponse = await fetch(apiUrl('/api/health'), { cache: 'no-store' });
          if (!healthResponse.ok) throw new Error(`health: ${healthResponse.status}`);
          const health = await healthResponse.json() as { backend_instance_id: string };
          if (cancelled) return;
          if (!pointer) {
            setIsRestoring(false);
            return;
          }
          if (pointer.backend_instance_id !== health.backend_instance_id) {
            window.localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
            activeRunIdRef.current = null;
            setRun(null);
            setSelectedId(null);
            setIsRestoring(false);
            return;
          }
          try {
            await refreshRun(pointer.run_id);
            if (!cancelled) setIsRestoring(false);
            return;
          } catch (restoreError) {
            const status = (restoreError as ApiError).status;
            if (status === 404 || status === 410) {
              window.localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
              activeRunIdRef.current = null;
              setRun(null);
              setSelectedId(null);
              setIsRestoring(false);
              return;
            }
            throw restoreError;
          }
        } catch {
          if (!pointer) {
            if (!cancelled) setIsRestoring(false);
            return;
          }
          await new Promise((resolve) => window.setTimeout(resolve, delay));
          delay = Math.min(delay * 2, 8000);
        }
      }
    };
    void restore();
    return () => { cancelled = true; };
  }, [refreshRun]);

  useEffect(() => {
    if (isRestoring || !run || run.status === 'interrupted') return;
    const extractionActive = run.records.some((record) => record.extraction_status === 'queued' || record.extraction_status === 'processing');
    const refinementActive = run.progress.pages.some((page) => page.active_refinements?.length);
    if ((run.status === 'completed' || run.status === 'failed') && !extractionActive && !refinementActive) return;
    let cancelled = false;
    let timer = 0;
    let delay = 900;
    const poll = async () => {
      try {
        await refreshRun(run.id);
        delay = 900;
      } catch (pollError) {
        if (activeRunIdRef.current !== run.id) return;
        const status = (pollError as ApiError).status;
        if (status === 404 || status === 410) {
          window.localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
          activeRunIdRef.current = null;
          setRun(null);
          setSelectedId(null);
          return;
        }
        delay = Math.min(delay * 2, 8000);
      }
      if (!cancelled) timer = window.setTimeout(() => void poll(), delay);
    };
    timer = window.setTimeout(() => void poll(), delay);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [isRestoring, refreshRun, run]);

  const uploadPdf = async (file?: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('请选择 PDF 文件');
      return;
    }
    setUploading(true);
    setError('');
    closePreview();
    setAuditMode(false);
    setHoveredAuditId(null);
    lockedAuditIdRef.current = null;
    setLockedAuditId(null);
    setAuditDraft(null);
    setAuditError('');
    setSelectedId(null);
    setPageFrameRequested(false);
    setViewMode('page');
    try {
      const body = new FormData();
      body.append('pdf', file);
      const response = await fetch(apiUrl('/api/runs'), { method: 'POST', body });
      if (!response.ok) throw new Error(await response.text());
      const created = await response.json() as { run_id: string; backend_instance_id: string };
      const pointer: ActiveRunPointer = { run_id: created.run_id, backend_instance_id: created.backend_instance_id };
      activeRunIdRef.current = created.run_id;
      window.localStorage.setItem(ACTIVE_RUN_STORAGE_KEY, JSON.stringify(pointer));
      setRun(null);
      setIsRestoring(false);
      await refreshRun(created.run_id);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'PDF 上传失败');
    } finally {
      setUploading(false);
    }
  };

  const updateRecordField = async (recordId: string, field: EditableField, value: string) => {
    if (!run) return;
    if (manualPendingRecords.some((record) => record.id === recordId)) {
      manualEditsRef.current[recordId] = { ...manualEditsRef.current[recordId], [field]: value || null };
      setManualPendingRecords((current) => current.map((record) => record.id === recordId ? {
        ...record,
        [field]: value || null,
        user_edited_fields: [...new Set([...(record.user_edited_fields ?? []), field])],
      } : record));
      return;
    }
    const response = await fetch(apiUrl(`/api/runs/${run.id}/records/${recordId}`), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ field, value: value === '' ? null : value }),
    });
    if (!response.ok) throw new Error(await response.text());
    const updated = await response.json() as InspectionRecord;
    setRun((current) => current ? {
      ...current,
      records: current.records.map((record) => record.id === recordId ? updated : record),
    } : current);
  };

  const updateMetadata = async (field: 'project' | 'revision' | 'author' | 'drawing_date', value: string) => {
    if (!run) return;
    const response = await fetch(apiUrl(`/api/runs/${run.id}/metadata`), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ field, value: value === '' ? null : value }),
    });
    if (!response.ok) throw new Error(await response.text());
    setRun((current) => current ? {
      ...current,
      records: current.records.map((record) => ({ ...record, [field]: value || null })),
    } : current);
  };

  const exportExcel = async () => {
    if (!run || exporting) return;
    setExporting(true);
    setEditError('');
    try {
      const response = await fetch(apiUrl(`/api/runs/${run.id}/export.xlsx`));
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${run.pdf_name.replace(/\.pdf$/i, '').replace(/[^\w.-]+/g, '_')}_inspection.xlsx`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (exportError) {
      setEditError(exportError instanceof Error ? exportError.message : 'Excel 导出失败');
    } finally {
      setExporting(false);
    }
  };

  const activeRecord = useMemo(() => {
    if (!run?.records.length) return null;
    if (selectedId) return run.records.find((record) => record.id === selectedId) ?? null;
    return [...run.records].sort((a, b) => b.sequence - a.sequence)[0];
  }, [run, selectedId]);
  const selectedRecord = useMemo(
    () => selectedId ? run?.records.find((record) => record.id === selectedId) ?? null : null,
    [run, selectedId],
  );
  const previewRecord = useMemo(
    () => previewRecordId ? run?.records.find((record) => record.id === previewRecordId) ?? null : null,
    [run, previewRecordId],
  );

  const documentRecord = useMemo(
    () => run?.records.find((record) => record.project || record.revision || record.author || record.drawing_date) ?? null,
    [run],
  );

  const progressPage = run?.progress.pages.find((page) => page.page_index === viewPage) ?? null;
  const currentPage = viewPage || run?.current_page || 1;
  const activeStage1Module = viewModule && progressPage
    ? progressPage.stage1_modules[viewModule - 1] ?? null
    : null;

  const saveRecord = useCallback(async (record: InspectionRecord, bbox: Box): Promise<InspectionRecord | null> => {
    if (!run) return null;
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
    return updated;
  }, [run]);

  const selectRecord = (record: InspectionRecord, mode: ViewMode = 'page', fromTable = false) => {
    closePreview();
    setSelectedId(record.id);
    setPageFrameRequested(fromTable);
    setFocusRequest(null);
    setViewPage(record.page_index);
    setViewModule(record.module_index || null);
    setViewMode(mode);
  };

  const selectAndFocusRecord = (record: InspectionRecord, mode: ViewMode, scopeKey: string, box: Box) => {
    closePreview();
    setSelectedId(record.id);
    setPageFrameRequested(false);
    setViewPage(record.page_index);
    setViewModule(record.module_index || null);
    setViewMode(mode);
    setFocusRequest({ scopeKey, box });
  };

  const changePage = (requested: number) => {
    if (!run || !Number.isFinite(requested)) return;
    const next = Math.max(1, Math.min(run.total_pages || 1, Math.round(requested)));
    closePreview();
    setHoveredAuditId(null);
    lockedAuditIdRef.current = null;
    setLockedAuditId(null);
    setAuditDraft(null);
    setAuditError('');
    setSelectedId(null);
    setPageFrameRequested(false);
    setFocusRequest(null);
    setViewModule(null);
    setViewPage(next);
    setViewMode('page');
  };

  const pageScopeKey = `page-${currentPage}`;
  const moduleOrigin = activeStage1Module?.bbox_pixels ?? null;
  const moduleWidth = moduleOrigin ? Math.max(1, moduleOrigin[2] - moduleOrigin[0]) : 1;
  const moduleHeight = moduleOrigin ? Math.max(1, moduleOrigin[3] - moduleOrigin[1]) : 1;
  const moduleScopeKey = `module-${currentPage}-${viewModule}`;
  const pageRefiningFrames = (progressPage?.active_refinements ?? []).map((item) => ({
    key: item.candidate_key,
    box: item.bbox,
  }));
  const moduleRefiningFrames = moduleOrigin
    ? (progressPage?.active_refinements ?? [])
        .filter((item) => item.module_index === viewModule)
        .map((item) => ({
          key: item.candidate_key,
          box: [
            item.bbox[0] - moduleOrigin[0], item.bbox[1] - moduleOrigin[1],
            item.bbox[2] - moduleOrigin[0], item.bbox[3] - moduleOrigin[1],
          ] as Box,
        }))
    : [];
  const overviewModuleOverlays: Overlay[] = progressPage?.stage1_modules.map((module, index) => ({
    key: `overview-module-${index + 1}`,
    box: module.bbox_pixels,
    label: `MODULE ${index + 1}`,
    hitboxOnly: true,
    onClick: () => {
      closePreview();
      setSelectedId(null);
      setPageFrameRequested(false);
      setFocusRequest(null);
      setViewModule(index + 1);
      setViewMode('module');
    },
  })) ?? [];
  const moduleRecords = run?.records.filter((record) =>
    record.page_index === currentPage && record.module_index === viewModule,
  ) ?? [];
  const moduleOverlays: Overlay[] = moduleOrigin
    ? moduleRecords.filter((record) => record.id !== selectedId).map((record) => {
        const localBox: Box = [
          record.effective_bbox[0] - moduleOrigin[0], record.effective_bbox[1] - moduleOrigin[1],
          record.effective_bbox[2] - moduleOrigin[0], record.effective_bbox[3] - moduleOrigin[1],
        ];
        return {
          key: record.id,
          box: localBox,
          label: `FAI ${record.fai ?? '—'}`,
          onClick: () => selectRecord(record, 'module'),
          onZoom: () => selectAndFocusRecord(record, 'module', moduleScopeKey, localBox),
        };
      })
    : [];
  const auditPageRecords = run?.records.filter((record) => record.page_index === currentPage) ?? [];
  const pageManualPendingRecords = manualPendingRecords.filter((record) => record.page_index === currentPage);
  const auditDisplayRecords = [...auditPageRecords, ...pageManualPendingRecords];
  const hoveredAuditRecord = auditDisplayRecords.find((record) => record.id === hoveredAuditId) ?? null;
  const lockedAuditRecord = auditDisplayRecords.find((record) => record.id === lockedAuditId) ?? null;
  const auditSelectorModuleIndex = lockedAuditRecord?.module_index || viewModule;
  const auditSelectorRecords = auditSelectorModuleIndex
    ? auditDisplayRecords.filter((record) => record.module_index === auditSelectorModuleIndex)
    : [];
  const lockedManualPendingRecord = manualPendingRecords.find((record) => record.id === lockedAuditRecord?.id) ?? null;
  const activeAuditRecord = auditMode ? hoveredAuditRecord ?? lockedAuditRecord : null;
  const activeAuditBox = activeAuditRecord
    ? auditDraft?.recordId === activeAuditRecord.id ? auditDraft.bbox : activeAuditRecord.effective_bbox
    : null;
  const auditOverlays: Overlay[] = auditMode ? auditDisplayRecords.map<Overlay>((record) => ({
    key: `audit-${record.id}`,
    box: record.effective_bbox,
    point: record.marker_bbox
      ? [(record.marker_bbox[0] + record.marker_bbox[2]) / 2, (record.marker_bbox[1] + record.marker_bbox[3]) / 2]
      : [(record.effective_bbox[0] + record.effective_bbox[2]) / 2, (record.effective_bbox[1] + record.effective_bbox[3]) / 2],
    kind: 'circle',
    label: record.fai ?? '—',
    // Keep the provisional hand-drawn marker black until Stage 2 resolves it.
    selected: !pageManualPendingRecords.some((pending) => pending.id === record.id) && activeAuditRecord?.id === record.id,
    onMouseEnter: () => {
      setHoveredAuditId(record.id);
      setAuditDraft({ recordId: record.id, bbox: record.effective_bbox });
      setAuditError('');
    },
    onMouseLeave: () => setHoveredAuditId((current) => current === record.id ? null : current),
    onClick: () => {
      setLockedAuditId(record.id);
      setHoveredAuditId(null);
      setSelectedId(record.id);
      setPageFrameRequested(false);
      setAuditDraft({ recordId: record.id, bbox: record.effective_bbox });
      if (record.module_index > 0) {
        setViewModule(record.module_index);
        setViewMode('module');
      } else {
        setViewModule(null);
        setViewMode('page');
      }
      setAuditError('');
    },
  })) : [];
  const pendingComponentOverlay: Overlay[] = auditMode && moduleOrigin
    ? pageManualPendingRecords.filter((record) => record.module_index > 0 && record.module_index === viewModule).map((record) => ({
        key: `audit-${record.id}`,
        box: [
          record.effective_bbox[0] - moduleOrigin[0], record.effective_bbox[1] - moduleOrigin[1],
          record.effective_bbox[2] - moduleOrigin[0], record.effective_bbox[3] - moduleOrigin[1],
        ] as Box,
        point: [
          (record.effective_bbox[0] + record.effective_bbox[2]) / 2 - moduleOrigin[0],
          (record.effective_bbox[1] + record.effective_bbox[3]) / 2 - moduleOrigin[1],
        ] as [number, number],
        kind: 'circle' as const, label: '—', selected: false,
        onClick: () => { setLockedAuditId(record.id); setHoveredAuditId(null); },
      }))
    : [];
  const toggleAuditMode = () => {
    const entering = !auditMode;
    closePreview();
    setFocusRequest(null);
    setViewMode('page');
    setViewModule(null);
    setPageFrameRequested(false);
    setHoveredAuditId(null);
    setAuditError('');
    if (entering) {
      const initialRecord = selectedRecord?.page_index === currentPage ? selectedRecord : null;
      lockedAuditIdRef.current = initialRecord?.id ?? null;
      setLockedAuditId(initialRecord?.id ?? null);
      setAuditDraft(initialRecord ? { recordId: initialRecord.id, bbox: initialRecord.effective_bbox } : null);
    } else {
      lockedAuditIdRef.current = null;
      setLockedAuditId(null);
      setAuditDraft(null);
    }
    setAuditMode(entering);
  };
  const saveAuditCrop = async (bbox: Box) => {
    if (!activeAuditRecord) return;
    if (manualPendingRecords.some((record) => record.id === activeAuditRecord.id)) {
      setAuditDraft({ recordId: activeAuditRecord.id, bbox });
      setManualPendingRecords((current) => current.map((record) => record.id === activeAuditRecord.id
        ? { ...record, effective_bbox: bbox, model_bbox: bbox }
        : record));
      return;
    }
    setAuditError('');
    const updated = await saveRecord(activeAuditRecord, bbox);
    setAuditDraft({ recordId: activeAuditRecord.id, bbox: updated?.effective_bbox ?? bbox });
  };
  const handleAuditSaveError = (saveError: unknown) => {
    setAuditError(saveError instanceof Error ? saveError.message : '截图保存失败，请重试');
  };

  const submitManualAuditBox = async (sourceBox: Box, source: 'page' | 'module', retryId?: string) => {
    if (!run) return;
    const retryRecord = retryId ? manualPendingRecords.find((record) => record.id === retryId) : null;
    const pageBox: Box = retryRecord
      ? sourceBox
      : source === 'module' && moduleOrigin
      ? [sourceBox[0] + moduleOrigin[0], sourceBox[1] + moduleOrigin[1], sourceBox[2] + moduleOrigin[0], sourceBox[3] + moduleOrigin[1]]
      : sourceBox;
    const pageWidth = progressPage?.page_width ?? 1;
    const pageHeight = progressPage?.page_height ?? 1;
    manualCounterRef.current += 1;
    const tempId = retryRecord?.id ?? `manual-pending-${manualCounterRef.current}`;
    const carriedEdits = manualEditsRef.current[tempId] ?? {};
    const pending: InspectionRecord = retryRecord ? {
      ...retryRecord,
      model_bbox: pageBox,
      effective_bbox: pageBox,
      extraction_status: 'processing',
      extraction_error: null,
    } : {
      id: tempId, candidate_key: tempId,
      sequence: Math.max(0, ...run.records.map((record) => record.sequence), ...manualPendingRecords.map((record) => record.sequence)) + 1,
      project: documentRecord?.project ?? null, revision: documentRecord?.revision ?? null,
      author: documentRecord?.author ?? null, drawing_date: documentRecord?.drawing_date ?? null,
      module: source === 'module' && viewModule ? `MODULE ${viewModule}` : null,
      module_index: source === 'module' ? viewModule ?? 0 : 0, page_index: currentPage,
      fai: null, spc: null, description: null, nominal: null, usl: null, lsl: null,
      hundred_percent: null, dc: null, points: null, page_width: pageWidth, page_height: pageHeight,
      model_bbox: pageBox, marker_bbox: null, effective_bbox: pageBox, user_override: false,
      crop_url: '', page_url: apiUrl(`/api/runs/${run.id}/pages/${currentPage}`), updated_at: new Date().toISOString(),
      extraction_status: 'processing', extraction_error: null,
      ...carriedEdits,
      user_edited_fields: Object.keys(carriedEdits),
    };
    manualEditsRef.current[tempId] = { ...carriedEdits };
    setManualPendingRecords((current) => retryRecord
      ? current.map((record) => record.id === tempId ? pending : record)
      : [...current, pending]);
    lockedAuditIdRef.current = tempId;
    setLockedAuditId(tempId);
    setHoveredAuditId(null);
    setAuditDraft({ recordId: tempId, bbox: pageBox });
    setAuditError('');
    setManualDrawMode(null);
    try {
      const response = await fetch(apiUrl(`/api/runs/${run.id}/pages/${currentPage}/audit/detect`), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: pageBox.map(Math.round), module_index: pending.module_index || null, candidate_key: pending.candidate_key }),
      });
      if (!response.ok) throw new Error(await response.text());
      let record = await response.json() as InspectionRecord;
      const edits = manualEditsRef.current[tempId] ?? {};
      for (const [field, value] of Object.entries(edits)) {
        const update = await fetch(apiUrl(`/api/runs/${run.id}/records/${record.id}`), {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ field, value }),
        });
        if (update.ok) record = await update.json() as InspectionRecord;
      }
      delete manualEditsRef.current[tempId];
      setRun((current) => current ? {
        ...current,
        records: current.records.some((item) => item.id === record.id)
          ? current.records.map((item) => item.id === record.id ? record : item)
          : [...current.records, record],
      } : current);
      setManualPendingRecords((current) => current.filter((item) => item.id !== tempId));
      setAuditDraft((current) => current?.recordId === tempId
        ? { recordId: record.id, bbox: record.effective_bbox }
        : current);
      if (lockedAuditIdRef.current === tempId) {
        lockedAuditIdRef.current = record.id;
        setLockedAuditId(record.id);
        if (record.module_index > 0) setViewModule(record.module_index);
        else { setViewModule(null); setViewMode('page'); }
        setAuditError('');
      }
    } catch (detectError) {
      const message = detectError instanceof Error ? detectError.message : 'FAI 识别失败；手动画框已保留';
      setManualPendingRecords((current) => current.map((record) => record.id === tempId
        ? { ...record, extraction_status: 'failed', extraction_error: message }
        : record));
      if (lockedAuditIdRef.current === tempId) setAuditError(message);
    }
  };
  const selectedModuleRecord = selectedRecord && selectedRecord.page_index === currentPage && selectedRecord.module_index === viewModule
    ? selectedRecord
    : null;
  const moduleSelection = selectedModuleRecord && moduleOrigin
    ? clampBox([
        selectedModuleRecord.effective_bbox[0] - moduleOrigin[0],
        selectedModuleRecord.effective_bbox[1] - moduleOrigin[1],
        selectedModuleRecord.effective_bbox[2] - moduleOrigin[0],
        selectedModuleRecord.effective_bbox[3] - moduleOrigin[1],
      ], moduleWidth, moduleHeight)
    : null;

  const currentPageModuleCount = progressPage?.stage1_modules.length ?? 0;
  const currentPageCandidateCount = progressPage
    ? Object.values(progressPage.stage2_modules).reduce(
        (sum, module) => sum + module.valid_cluster_count,
        0,
      )
    : 0;
  const currentPageCompletedModuleCount = progressPage
    ? Object.keys(progressPage.stage2_modules).length
    : 0;
  const currentPageRefinedCount = run?.records.filter(
    (record) => record.page_index === currentPage,
  ).length ?? 0;
  const currentPagePercent = currentPageModuleCount
    ? Math.min(100, Math.round((currentPageCompletedModuleCount / currentPageModuleCount) * 100))
    : 0;

  return (
    <main className="scan-page">
      <header className="scan-header">
        <div className="scan-brand-group">
          <Link className="brand" href="/"><span className="brand-mark" aria-hidden="true">CV</span><span>Claw View</span></Link>
          <Link className="back-home-link" href="/">← 主界面</Link>
        </div>
        <div className="scan-progress" aria-live="polite">
          <span className={`status-dot status-${run?.status ?? 'idle'}`} />
          {isRestoring ? <strong>正在恢复任务…</strong> : run ? <><strong>{run.status === 'completed' ? '处理完成' : run.status === 'failed' ? '处理失败' : run.status === 'interrupted' ? '任务已中断' : '正在处理'}</strong><span>第 {run.current_page || 0}/{run.total_pages || '—'} 页</span><span className="progress-file">{run.pdf_name}</span></> : <span>等待 PDF</span>}
        </div>
        <div className="scan-header-actions">
          <button className={`audit-mode-toggle ${auditMode ? 'is-active' : ''}`} type="button" aria-pressed={auditMode} disabled={!run || uploading || isRestoring} onClick={toggleAuditMode}>{auditMode ? '退出审核模式' : '审核模式'}</button>
          <button className="upload-compact" type="button" disabled={uploading || isRestoring} onClick={() => inputRef.current?.click()}>{uploading ? '上传中…' : isRestoring ? '恢复中…' : '选择 PDF'}</button>
        </div>
        <input ref={inputRef} className="visually-hidden" type="file" accept="application/pdf,.pdf" onChange={(event: ChangeEvent<HTMLInputElement>) => void uploadPdf(event.target.files?.[0])} />
      </header>
      <section
        className="scan-workspace"
        style={{ gridTemplateColumns: `${leftWidth}% 8px minmax(0, 1fr)` }}
      >
        <aside className={`inspection-panel ${auditMode ? 'is-audit-mode' : ''}`}>
          <div className="inspection-metadata" aria-label="图纸资料">
            <div><span>PROJECT</span><EditableCell value={documentRecord?.project} onSave={(value) => updateMetadata('project', value)} /></div>
            <div><span>REVISION</span><EditableCell value={documentRecord?.revision} onSave={(value) => updateMetadata('revision', value)} /></div>
            <div><span>AUTHOR</span><EditableCell value={documentRecord?.author} onSave={(value) => updateMetadata('author', value)} /></div>
            <div><span>DATE</span><EditableCell value={documentRecord?.drawing_date} onSave={(value) => updateMetadata('drawing_date', value)} /></div>
          </div>
          {auditMode ? (
            activeAuditRecord && activeAuditBox && run ? (
              <AuditPanel
                record={activeAuditRecord}
                pageImageUrl={apiUrl(`/api/runs/${run.id}/pages/${activeAuditRecord.page_index}`)}
                bbox={activeAuditBox}
                onDraftChange={(bbox) => setAuditDraft({ recordId: activeAuditRecord.id, bbox })}
                onSave={saveAuditCrop}
                onSaveError={handleAuditSaveError}
                onUpdateField={(field, value) => updateRecordField(activeAuditRecord.id, field, value)}
                error={auditError}
              />
            ) : (
              <div className="audit-empty-state"><span className="audit-empty-mark">◎</span><strong>选择一个 FAI 开始审核</strong><small>将鼠标悬停在右侧圆圈上预览，点击即可锁定</small></div>
            )
          ) : <>
          <div className="panel-heading"><div><span className="panel-kicker">INSPECTION INDEX</span><h1>检验信息</h1></div><span className="record-count">{run?.records.length ?? 0} 条</span></div>
          <div className="inspection-table-wrap"><div className="inspection-table-scroll"><table className="inspection-table">
            <thead><tr><th>No.</th><th>Module</th><th>Page</th><th>FAI</th><th>SPC</th><th>Description</th><th>Nominal</th><th>USL</th><th>LSL</th><th>SPC截图</th><th>100%</th><th>DC</th><th>Points</th></tr></thead>
            <tbody>
              {run?.records.map((record, index) => (
                <tr key={record.id} className={`${selectedId === record.id ? 'is-selected ' : ''}${record.extraction_status === 'queued' || record.extraction_status === 'processing' ? 'is-ai-filling' : ''}`} onClick={() => selectRecord(record, 'page', true)}>
                  <td>{index + 1}</td>
                  <td><EditableCell value={record.module} onSave={(value) => updateRecordField(record.id, 'module', value)} /></td>
                  <td>page {record.page_index}</td>
                  <td className="fai-value"><EditableCell value={record.fai} onSave={(value) => updateRecordField(record.id, 'fai', value)} /></td>
                  <td><EditableCell value={record.spc} onSave={(value) => updateRecordField(record.id, 'spc', value)} /></td>
                  <td className="description-cell"><EditableCell value={record.description} onSave={(value) => updateRecordField(record.id, 'description', value)} /></td>
                  <td><EditableCell value={record.nominal} onSave={(value) => updateRecordField(record.id, 'nominal', value)} /></td>
                  <td><EditableCell value={record.usl} onSave={(value) => updateRecordField(record.id, 'usl', value)} /></td>
                  <td><EditableCell value={record.lsl} onSave={(value) => updateRecordField(record.id, 'lsl', value)} /></td>
                  <td className="thumbnail-cell">
                    <button
                      type="button"
                      className="thumbnail-trigger"
                      aria-label={`查看 FAI ${record.fai ?? '—'} 当前完整截图`}
                      onClick={(event) => {
                        event.stopPropagation();
                        selectRecord(record, 'page', true);
                        setPreviewRecordId(record.id);
                        setPreviewAnchor(event.currentTarget);
                      }}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={`${apiUrl(record.crop_url)}?v=${encodeURIComponent(record.updated_at)}`} alt={`FAI ${record.fai ?? ''} 截图`} />
                    </button>
                  </td>
                  <td><EditableCell value={record.hundred_percent} onSave={(value) => updateRecordField(record.id, 'hundred_percent', value)} /></td>
                  <td><EditableCell value={record.dc} onSave={(value) => updateRecordField(record.id, 'dc', value)} /></td>
                  <td><EditableCell value={record.points} onSave={(value) => updateRecordField(record.id, 'points', value)} /></td>
                </tr>
              ))}
              {!run?.records.length && <tr className="empty-table-row"><td colSpan={13}>crop2_refined 完成后，结果会按 FAI 从小到大出现在这里</td></tr>}
            </tbody>
          </table></div><div className="inspection-table-actions"><span>{editError}</span><button type="button" disabled={!run?.records.length || exporting} onClick={() => void exportExcel()}>{exporting ? '正在导出…' : '导出 Excel'}</button></div></div>
          </>}
        </aside>

        <div
          className="resize-divider"
          role="separator"
          aria-label={auditMode ? '调整审核面板与图纸宽度' : '调整表格与图纸宽度'}
          aria-orientation="vertical"
          aria-valuemin={30}
          aria-valuemax={75}
          aria-valuenow={Math.round(leftWidth)}
          onPointerDown={(event: ReactPointerEvent<HTMLDivElement>) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            resizeRef.current = { startX: event.clientX, startWidth: leftWidth };
          }}
          onPointerMove={(event: ReactPointerEvent<HTMLDivElement>) => {
            const start = resizeRef.current;
            const container = event.currentTarget.parentElement;
            if (!start || !container) return;
            const width = container.getBoundingClientRect().width;
            const next = start.startWidth + ((event.clientX - start.startX) / width) * 100;
            setLeftWidth(Math.max(30, Math.min(75, next)));
          }}
          onPointerUp={(event: ReactPointerEvent<HTMLDivElement>) => {
            resizeRef.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerCancel={() => { resizeRef.current = null; }}
        />

        <section className={`drawing-panel ${draggingFile ? 'is-file-dragging' : ''}`} onDragOver={(event: DragEvent<HTMLElement>) => { event.preventDefault(); setDraggingFile(true); }} onDragLeave={() => setDraggingFile(false)} onDrop={(event: DragEvent<HTMLElement>) => { event.preventDefault(); setDraggingFile(false); void uploadPdf(event.dataTransfer.files?.[0]); }}>
          <div className="drawing-toolbar">
            <div className="page-navigation">
              <span>{viewMode === 'overview' ? 'MODULE OVERVIEW' : viewMode === 'module' ? `MODULE ${viewModule}` : 'PDF PAGE'}</span>
              <button type="button" aria-label="上一页" disabled={!run || currentPage <= 1} onClick={() => changePage(currentPage - 1)}>▲</button>
              <input
                aria-label="跳转到页码"
                inputMode="numeric"
                value={run ? currentPage : ''}
                onChange={(event) => {
                  const typed = Number(event.target.value);
                  if (event.target.value && Number.isInteger(typed)) changePage(typed);
                }}
              />
              <strong>/ {run?.total_pages || '—'}</strong>
              <button type="button" aria-label="下一页" disabled={!run || currentPage >= (run.total_pages || 1)} onClick={() => changePage(currentPage + 1)}>▼</button>
            </div>
            {run && <div className="progress-summary">
              <span>模块 <b>{currentPageModuleCount}</b></span>
              <span>FAI <b>{currentPageCandidateCount}</b></span>
              <span>REFINED <b>{currentPageRefinedCount}</b></span>
              <span>处理中模块 <b>{run.current_module ? `${run.current_module}/${currentPageModuleCount || '—'}` : `—/${currentPageModuleCount || '—'}`}</b></span>
              <span className="progress-percent"><i style={{ width: `${currentPagePercent}%` }} /><b>{currentPagePercent}%</b></span>
              {!auditMode && <button type="button" onClick={() => { setFocusRequest(null); setSelectedId(null); setPageFrameRequested(false); setViewModule(null); setViewMode('overview'); }}>查看部件全览</button>}
              {viewMode !== 'page' && <button type="button" onClick={() => { setFocusRequest(null); setViewMode('page'); }}>返回原图</button>}
            </div>}
          </div>

          {isRestoring ? (
            <div className="viewer-message"><span className="loader-ring" /><strong>正在检查后端并恢复扫描任务</strong></div>
          ) : !run ? (
            <button className="pdf-dropzone" type="button" onClick={() => inputRef.current?.click()}><span className="dropzone-symbol">＋</span><strong>拖拽 PDF 到这里</strong><small>或从本地目录选择工程图纸</small></button>
          ) : run.status === 'rendering' && !run.total_pages ? (
            <div className="viewer-message"><span className="loader-ring" /><strong>正在处理 PDF 文件并传入模型</strong></div>
          ) : (
            <div className="drawing-scroll">
              {auditMode && viewMode === 'module' && lockedAuditRecord && activeAuditBox && activeStage1Module?.crop_url && moduleOrigin ? (
                <>
                  <InteractiveDrawing
                    key={`audit-component-${currentPage}-${viewModule}`}
                    imageUrl={apiUrl(activeStage1Module.crop_url)}
                    alt={`FAI ${lockedAuditRecord.fai ?? '—'} 所属部件`}
                    coordinateWidth={moduleWidth}
                    coordinateHeight={moduleHeight}
                    selection={[
                      activeAuditBox[0] - moduleOrigin[0], activeAuditBox[1] - moduleOrigin[1],
                      activeAuditBox[2] - moduleOrigin[0], activeAuditBox[3] - moduleOrigin[1],
                    ]}
                    selectionLabel={`FAI ${lockedAuditRecord.fai ?? '—'}`}
                    syncSelection
                    overlays={pendingComponentOverlay}
                    onSave={async (local) => { await saveAuditCrop([
                      local[0] + moduleOrigin[0], local[1] + moduleOrigin[1],
                      local[2] + moduleOrigin[0], local[3] + moduleOrigin[1],
                    ]); }}
                    drawMode={manualDrawMode === 'module'}
                    onDrawComplete={(box) => void submitManualAuditBox(box, 'module')}
                    scopeKey={`audit-component-${currentPage}-${viewModule}`}
                    focusRequest={null}
                    onRequestFocus={() => undefined}
                    onClearFocus={() => undefined}
                    hideFocusControls
                  />
                  <div className="audit-right-controls">
                    <AuditFaiSelector records={auditSelectorRecords} selectedId={lockedAuditRecord.id} onSelect={(record) => {
                      setLockedAuditId(record.id); setHoveredAuditId(null); setSelectedId(record.id);
                      setAuditDraft({ recordId: record.id, bbox: record.effective_bbox });
                      if (record.module_index > 0) setViewModule(record.module_index);
                      else { setViewModule(null); setViewMode('page'); }
                    }} />
                    {lockedManualPendingRecord?.extraction_status === 'failed' && activeAuditBox && <button type="button" className="audit-draw-toggle" onClick={() => void submitManualAuditBox(activeAuditBox, 'page', lockedManualPendingRecord.id)}>重新识别</button>}
                    <button type="button" className={`audit-draw-toggle ${manualDrawMode === 'module' ? 'is-active' : ''}`} onClick={() => setManualDrawMode((mode) => mode === 'module' ? null : 'module')}>⌖ 在部件图框选 FAI</button>
                  </div>
                </>
              ) : viewMode === 'overview' && progressPage?.has_overview ? (
                <InteractiveDrawing
                  key={`overview-${currentPage}`}
                  imageUrl={apiUrl(progressPage.overview_url)}
                  alt={`第 ${currentPage} 页部件全览`}
                  coordinateWidth={progressPage.page_width}
                  coordinateHeight={progressPage.page_height}
                  selection={null}
                  overlays={overviewModuleOverlays}
                  scopeKey={`overview-${currentPage}`}
                  focusRequest={null}
                  onRequestFocus={() => undefined}
                  onClearFocus={() => undefined}
                />
              ) : viewMode === 'module' && activeStage1Module?.crop_url && moduleOrigin ? (
                <InteractiveDrawing
                  key={`module-${viewPage}-${viewModule}-${selectedId ?? 'none'}`}
                  imageUrl={apiUrl(activeStage1Module.crop_url)}
                  alt={`模块 ${viewModule} 原始图`}
                  coordinateWidth={moduleWidth}
                  coordinateHeight={moduleHeight}
                  selection={moduleSelection}
                  selectionLabel={selectedModuleRecord ? `FAI ${selectedModuleRecord.fai ?? '—'}` : undefined}
                  overlays={moduleOverlays}
                  refiningFrames={moduleRefiningFrames}
                  onSave={selectedModuleRecord && moduleSelection ? async (local) => { await saveRecord(selectedModuleRecord, [local[0] + moduleOrigin[0], local[1] + moduleOrigin[1], local[2] + moduleOrigin[0], local[3] + moduleOrigin[1]]); } : undefined}
                  scopeKey={moduleScopeKey}
                  focusRequest={focusRequest}
                  onRequestFocus={(box, scope) => setFocusRequest({ scopeKey: scope, box })}
                  onClearFocus={() => setFocusRequest(null)}
                />
              ) : (
                <InteractiveDrawing
                  key={auditMode ? `audit-page-${currentPage}` : `page-${currentPage}-${viewMode}-${selectedId ?? 'none'}-${pageFrameRequested ? 'selected' : 'clean'}`}
                  imageUrl={apiUrl(`/api/runs/${run.id}/pages/${currentPage}`)}
                  alt={`PDF 第 ${currentPage} 页`}
                  coordinateWidth={progressPage?.page_width ?? activeRecord?.page_width ?? 1}
                  coordinateHeight={progressPage?.page_height ?? activeRecord?.page_height ?? 1}
                  selection={auditMode && lockedAuditRecord ? activeAuditBox : pageFrameRequested && selectedRecord?.page_index === currentPage ? selectedRecord.effective_bbox : null}
                  selectionLabel={auditMode && lockedAuditRecord ? `FAI ${lockedAuditRecord.fai ?? '—'}` : pageFrameRequested && selectedRecord?.page_index === currentPage ? `FAI ${selectedRecord.fai ?? '—'}` : undefined}
                  overlays={auditMode ? auditOverlays : []}
                  refiningFrames={auditMode ? [] : pageRefiningFrames}
                  onSave={auditMode && lockedAuditRecord ? saveAuditCrop : pageFrameRequested && selectedRecord?.page_index === currentPage ? async (box) => { await saveRecord(selectedRecord, box); } : undefined}
                  drawMode={auditMode && manualDrawMode === 'page'}
                  onDrawComplete={auditMode ? (box) => void submitManualAuditBox(box, 'page') : undefined}
                  scopeKey={pageScopeKey}
                  focusRequest={focusRequest}
                  onRequestFocus={(box, scope) => setFocusRequest({ scopeKey: scope, box })}
                  onClearFocus={() => setFocusRequest(null)}
                />
              )}
              {auditMode && !(viewMode === 'module' && lockedAuditRecord && activeStage1Module?.crop_url && moduleOrigin) && <div className="audit-right-controls">
                {lockedManualPendingRecord?.extraction_status === 'failed' && activeAuditBox && <button type="button" className="audit-draw-toggle" onClick={() => void submitManualAuditBox(activeAuditBox, 'page', lockedManualPendingRecord.id)}>重新识别</button>}
                <button type="button" className={`audit-draw-toggle ${manualDrawMode === 'page' ? 'is-active' : ''}`} onClick={() => setManualDrawMode((mode) => mode === 'page' ? null : 'page')}>⌖ 在整页图框选 FAI</button>
              </div>}
            </div>
          )}
          {draggingFile && <div className="drop-overlay">松开以开始扫描 PDF</div>}
          {error && <div className="scan-error">{error}</div>}
          {run?.error && <div className="scan-error">{run.error}</div>}
        </section>
      </section>
      {previewRecord && previewAnchor && <CropPreview record={previewRecord} anchor={previewAnchor} onClose={closePreview} />}
    </main>
  );
}
