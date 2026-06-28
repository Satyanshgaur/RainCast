'use client';

import React, { useState, useEffect, useRef } from 'react';
import Script from 'next/script';
import Navbar from '@/components/Navbar';

interface DocItem {
  name: string;
  fileName: string;
  content: string;
}

interface DocPortalClientProps {
  docs: DocItem[];
}

declare global {
  interface Window {
    marked: any;
    hljs: any;
    renderMathInElement: any;
    mermaid: any;
  }
}

export default function DocPortalClient({ docs }: DocPortalClientProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [htmlContent, setHtmlContent] = useState('');
  const [scriptsLoaded, setScriptsLoaded] = useState({
    marked: false,
    hljs: false,
    katex: false,
    katexAuto: false,
    mermaid: false,
  });
  const contentRef = useRef<HTMLDivElement>(null);

  const activeDoc = docs[activeTab];

  // Handler to parse alerts in markdown
  function parseAlerts(markdown: string) {
    const alertTypes = ['NOTE', 'WARNING', 'IMPORTANT', 'TIP', 'CAUTION'];
    let parsed = markdown;
    
    alertTypes.forEach(type => {
      const regex = new RegExp(`> \\[\\!${type}\\]\\n> ([\\s\\S]*?)(?=\\n\\n|\\n[^>]|$)`, 'g');
      parsed = parsed.replace(regex, (match, content) => {
        const cleanContent = content.replace(/^>\s?/gm, '').trim();
        const alertClass = type.toLowerCase();
        return `\n<div class="doc-alert ${alertClass}"><strong>${type}</strong><br>${cleanContent}</div>\n`;
      });
    });
    
    return parsed;
  }

  // Adjust relative image paths to load directly relative to served root
  function adjustAssetPaths(markdown: string) {
    let adjusted = markdown;
    adjusted = adjusted.replace(/\((plots\/.*?)\)/g, '(http://localhost:8000/docs/$1)');
    adjusted = adjusted.replace(/src="(plots\/.*?)"/g, 'src="http://localhost:8000/docs/$1"');
    adjusted = adjusted.replace(/\(\.\.\/(val_and_bench\/.*?)\)/g, '(http://localhost:8000/$1)');
    return adjusted;
  }

  // Render markdown content dynamically
  useEffect(() => {
    if (!activeDoc || !window.marked) return;

    let md = activeDoc.content;
    md = parseAlerts(md);
    md = adjustAssetPaths(md);

    try {
      const html = window.marked.parse(md);
      setHtmlContent(html);
    } catch (err) {
      console.error(err);
      setHtmlContent(`<div class="doc-alert warning"><strong>Error</strong><br>Failed to parse markdown.</div>`);
    }
  }, [activeTab, activeDoc, scriptsLoaded.marked]);

  // Apply math formulas, code highlight and mermaid charts
  useEffect(() => {
    if (!htmlContent || !contentRef.current) return;

    // 1. Process Mermaid blocks
    if (window.mermaid) {
      const mermaidBlocks = contentRef.current.querySelectorAll('pre code.language-mermaid');
      mermaidBlocks.forEach((block, index) => {
        const pre = block.parentNode;
        if (!pre) return;
        const code = block.textContent?.trim() || '';
        const id = `mermaid-chart-next-${index}`;
        
        const div = document.createElement('div');
        div.className = 'mermaid';
        div.id = id;
        div.textContent = code;
        
        pre.parentNode?.replaceChild(div, pre);
      });

      if (contentRef.current.querySelectorAll('.mermaid').length > 0) {
        try {
          window.mermaid.init(undefined, contentRef.current.querySelectorAll('.mermaid'));
        } catch (mErr) {
          console.error('Mermaid render error:', mErr);
        }
      }
    }

    // 2. Process LaTeX math formulas
    if (window.renderMathInElement) {
      window.renderMathInElement(contentRef.current, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false
      });
    }

    // 3. Highlight code syntax
    if (window.hljs) {
      contentRef.current.querySelectorAll('pre code').forEach((block) => {
        window.hljs.highlightElement(block);
      });
    }
  }, [htmlContent]);

  return (
    <>
      <Navbar />
      <div className="gradient-wash"></div>

      {/* Script injections using next/script */}
      <Script
        src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, marked: true }))}
      />
      <Script
        src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, hljs: true }))}
      />
      <Script
        src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, katex: true }))}
      />
      <Script
        src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, katexAuto: true }))}
      />
      <Script
        src="https://cdn.jsdelivr.net/npm/mermaid@10.2.4/dist/mermaid.min.js"
        onLoad={() => {
          if (window.mermaid) {
            window.mermaid.initialize({
              startOnLoad: false,
              theme: 'dark',
              themeVariables: {
                background: '#012624',
                primaryColor: '#011d1c',
                primaryTextColor: '#ffffff',
                lineColor: '#00827c',
              }
            });
          }
          setScriptsLoaded(prev => ({ ...prev, mermaid: true }));
        }}
      />

      <main className="page-content" style={{ paddingTop: '120px' }}>
        <div className="section-container">
          <div className="docs-layout">
            {/* Sidebar Left Navigation */}
            <aside className="docs-sidebar">
              <ul className="sidebar-menu">
                {docs.map((doc, idx) => (
                  <li key={doc.fileName}>
                    <button
                      type="button"
                      className={`sidebar-link w-full text-left ${activeTab === idx ? 'active' : ''}`}
                      onClick={() => {
                        setActiveTab(idx);
                        window.scrollTo({
                          top: 0,
                          behavior: 'smooth'
                        });
                      }}
                    >
                      {doc.name}
                    </button>
                  </li>
                ))}
              </ul>
            </aside>

            {/* Document Content Display Right Pane */}
            <div className="docs-body glass-card p-[48px]">
              <article className="markdown-body" ref={contentRef} dangerouslySetInnerHTML={{ __html: htmlContent }} />
            </div>
          </div>
        </div>
      </main>

      <footer className="footer-container">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="brand-name">Raincast</span>
            <span className="footer-meta font-mono">v2.1.0-API</span>
          </div>
          <div className="footer-copy">
            &copy; 2026 Raincast. Technical Reference Suite.
          </div>
        </div>
      </footer>
    </>
  );
}
