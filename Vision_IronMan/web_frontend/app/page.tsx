'use client';

import Link from 'next/link';
import { PointerEvent, useEffect, useRef, useState } from 'react';

const pattern = Array.from({ length: 54 }, (_, index) =>
  ['C', 'L', 'A', 'W'][index % 4],
);

type Position = { x: number; y: number };

export default function Home() {
  const heroRef = useRef<HTMLElement>(null);
  const dragOffset = useRef<Position>({ x: 0, y: 0 });
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [heroSize, setHeroSize] = useState({ width: 1200, height: 560 });
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const hero = heroRef.current;
    if (!hero) return;

    const updateSize = () => {
      const rect = hero.getBoundingClientRect();
      setHeroSize({ width: rect.width, height: rect.height });
      setPosition((current) => {
        if (current.x !== 0 || current.y !== 0) {
          return {
            x: Math.min(current.x, Math.max(16, rect.width - 250)),
            y: Math.min(current.y, Math.max(16, rect.height - 190)),
          };
        }
        return {
          x: Math.max(28, rect.width * 0.69),
          y: Math.max(24, rect.height * 0.14),
        };
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(hero);
    return () => observer.disconnect();
  }, []);

  const moveBlob = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging || !heroRef.current) return;
    const rect = heroRef.current.getBoundingClientRect();
    const blobWidth = Math.min(335, Math.max(215, rect.width * 0.23));
    const blobHeight = blobWidth / 1.38;
    const nextX = event.clientX - rect.left - dragOffset.current.x;
    const nextY = event.clientY - rect.top - dragOffset.current.y;
    setPosition({
      x: Math.max(-blobWidth * 0.1, Math.min(nextX, rect.width - blobWidth * 0.9)),
      y: Math.max(-blobHeight * 0.15, Math.min(nextY, rect.height - blobHeight * 0.85)),
    });
  };

  return (
    <main className="site-shell">
      <header className="site-header">
        <Link className="brand" href="/" aria-label="Claw View home">
          <span className="brand-mark" aria-hidden="true">CV</span>
          <span>Claw View</span>
        </Link>
        <div className="header-meta" aria-label="Product identity">
          <span>VISION SYSTEM</span>
          <span className="header-dot" aria-hidden="true" />
          <span>CN</span>
        </div>
      </header>

      <section ref={heroRef} className="hero" aria-labelledby="hero-title">
        <div className="letter-field" aria-hidden="true">
          {pattern.map((letter, index) => <span key={index}>{letter}</span>)}
        </div>

        <div className="hero-copy">
          <p className="eyebrow">ENGINEERING DRAWING INTELLIGENCE</p>
          <h1 id="hero-title">HELLO, I&apos;M CLAW VIEW</h1>
          <p className="drag-hint">拖动黑色图形，探索 Claw View</p>
        </div>

        <div
          className={`blob ${dragging ? 'is-dragging' : ''}`}
          style={{ left: position.x, top: position.y }}
          onPointerDown={(event) => {
            const blobRect = event.currentTarget.getBoundingClientRect();
            dragOffset.current = {
              x: event.clientX - blobRect.left,
              y: event.clientY - blobRect.top,
            };
            event.currentTarget.setPointerCapture(event.pointerId);
            setDragging(true);
          }}
          onPointerMove={moveBlob}
          onPointerUp={(event) => {
            event.currentTarget.releasePointerCapture(event.pointerId);
            setDragging(false);
          }}
          onPointerCancel={() => setDragging(false)}
          role="img"
          aria-label="可拖拽的黑色动态图形"
        >
          <div
            className="blob-world"
            style={{
              width: heroSize.width,
              height: heroSize.height,
              transform: `translate(${-position.x}px, ${-position.y}px)`,
            }}
            aria-hidden="true"
          >
            <div className="letter-field letter-field-inverse">
              {pattern.map((letter, index) => <span key={index}>{letter}</span>)}
            </div>
            <div className="hero-copy hero-copy-inverse">
              <p className="eyebrow">ENGINEERING DRAWING INTELLIGENCE</p>
              <h2>你好，我是 CLAW VIEW</h2>
              <p className="drag-hint">READ · TRACE · COMPARE</p>
            </div>
          </div>
          <span className="blob-grip" aria-hidden="true">DRAG</span>
        </div>
      </section>

      <nav className="entry-grid" aria-label="Claw View tools">
        <Link className="entry-card entry-card-scan" href="/scan">
          <span className="entry-index">01</span>
          <span className="entry-text">
            <strong>开始智能扫描图纸</strong>
            <small>识别模块、FAI 标识与完整部件</small>
          </span>
          <span className="entry-arrow" aria-hidden="true">↗</span>
        </Link>
        <Link className="entry-card entry-card-compare" href="/compare">
          <span className="entry-index">02</span>
          <span className="entry-text">
            <strong>对比不同版本图纸</strong>
            <small>发现版本之间的结构与标注变化</small>
          </span>
          <span className="entry-arrow" aria-hidden="true">↗</span>
        </Link>
      </nav>
    </main>
  );
}
