import fs from 'fs';
import path from 'path';
import DocPortalClient from '@/components/DocPortalClient';
import Head from 'next/head';

interface DocItem {
  name: string;
  fileName: string;
  content: string;
}

export const dynamic = 'force-dynamic';

export default async function DocsPage() {
  const files = [
    { name: 'REST & WebSocket API', fileName: 'api_documentation.md' },
    { name: 'System Architecture', fileName: 'architecture.md' },
    { name: 'Physics Models', fileName: 'physics_models.md' },
    { name: 'Maseng-Bakken Model', fileName: 'rain_model.md' },
    { name: 'Inverse Narrowcasting', fileName: 'inverse_rain_rate.md' },
    { name: 'Validation Methodology', fileName: 'validation.md' },
    { name: 'Performance Benchmarks', fileName: 'benchmarks.md' },
    { name: 'References', fileName: 'references.md' }
  ];

  const docs: DocItem[] = files.map(file => {
    const filePath = path.join(process.cwd(), '../docs', file.fileName);
    const content = fs.readFileSync(filePath, 'utf8');
    return {
      ...file,
      content
    };
  });

  return (
    <>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" />
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css" />
      <DocPortalClient docs={docs} />
    </>
  );
}
