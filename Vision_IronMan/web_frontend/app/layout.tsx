import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Claw View — Engineering Drawing Intelligence',
  description: 'Scan engineering drawings and compare drawing revisions with Claw View.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
