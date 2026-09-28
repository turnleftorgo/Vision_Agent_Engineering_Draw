import Link from 'next/link';

export default function ComparePage() {
  return (
    <main className="placeholder-page">
      <header className="site-header">
        <Link className="brand" href="/">
          <span className="brand-mark" aria-hidden="true">CV</span><span>Claw View</span>
        </Link>
        <span className="header-meta">REVISION COMPARE</span>
      </header>
      <section className="placeholder-main">
        <div className="placeholder-content">
          <span className="placeholder-number">MODULE 02 · COMING NEXT</span>
          <h1>对比不同版本图纸</h1>
          <p>这个入口已经就位。下一阶段将在这里加入双图上传、版本差异定位和变更结果浏览。</p>
          <Link className="back-link" href="/">← 返回 Claw View</Link>
        </div>
      </section>
    </main>
  );
}
