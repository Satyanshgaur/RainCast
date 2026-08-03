/**
 * Bpowell Style UI — Interactive Logic (wbsite/app.js)
 */

document.addEventListener('DOMContentLoaded', () => {
  initReadingProgress();
  initTocDrawer();
  initDiagramSlot();
  initTableUploader();
  initImpairmentExplorer();
  initMathJaxFormatting();
});

/* Reading Progress Bar */
function initReadingProgress() {
  const progressBar = document.getElementById('readingProgressBar');
  if (!progressBar) return;

  window.addEventListener('scroll', () => {
    const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (totalHeight <= 0) return;
    const progress = (window.scrollY / totalHeight) * 100;
    progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  });
}

/* Floating TOC Drawer */
function initTocDrawer() {
  const tocBtn = document.getElementById('tocToggleBtn');
  const tocMenu = document.getElementById('tocMenu');
  if (!tocBtn || !tocMenu) return;

  tocBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    tocMenu.classList.toggle('active');
  });

  document.addEventListener('click', (e) => {
    if (!tocMenu.contains(e.target) && e.target !== tocBtn) {
      tocMenu.classList.remove('active');
    }
  });

  const links = tocMenu.querySelectorAll('.toc-link');
  links.forEach(link => {
    link.addEventListener('click', () => {
      tocMenu.classList.remove('active');
    });
  });
}

/* Diagram Container Slot / Preview & Upload Capabilities */
function initDiagramSlot() {
  const uploadInput = document.getElementById('diagramFileInput');
  const diagramViewport = document.getElementById('diagramViewport');
  const diagramTypeSelect = document.getElementById('diagramTypeSelect');

  if (diagramTypeSelect) {
    diagramTypeSelect.addEventListener('change', (e) => {
      renderBuiltinDiagram(e.target.value);
    });
  }

  if (uploadInput && diagramViewport) {
    uploadInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      if (file.type.includes('image') || file.type.includes('svg')) {
        reader.onload = (event) => {
          diagramViewport.innerHTML = `<img src="${event.target.result}" alt="Uploaded Diagram" style="max-width:100%; height:auto; display:block;" />`;
          const caption = document.getElementById('figureCaption');
          if (caption) caption.textContent = `Uploaded Custom Diagram: ${file.name}`;
        };
        reader.readAsDataURL(file);
      } else {
        reader.onload = (event) => {
          diagramViewport.innerHTML = `<pre style="font-family:var(--font-mono); font-size:12px; text-align:left; overflow-x:auto;">${escapeHtml(event.target.result)}</pre>`;
          const caption = document.getElementById('figureCaption');
          if (caption) caption.textContent = `Uploaded File: ${file.name}`;
        };
        reader.readAsText(file);
      }
    });
  }
}

function renderBuiltinDiagram(type) {
  const diagramViewport = document.getElementById('diagramViewport');
  const caption = document.getElementById('figureCaption');
  if (!diagramViewport) return;

  if (type === 'pipeline') {
    diagramViewport.innerHTML = `
      <svg viewBox="0 0 800 240" style="width:100%; height:auto; max-height:260px;" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="20" width="140" height="60" fill="none" stroke="#111111" stroke-width="1.5"/>
        <text x="80" y="55" font-family="Space Grotesk" font-size="12" font-weight="600" text-anchor="middle">SGP4 Orbit Prop</text>
        
        <path d="M 150 50 L 210 50" stroke="#111111" stroke-width="1.5" marker-end="url(#arrow)"/>
        
        <rect x="210" y="20" width="160" height="60" fill="none" stroke="#111111" stroke-width="1.5"/>
        <text x="290" y="55" font-family="Space Grotesk" font-size="12" font-weight="600" text-anchor="middle">Slant Path Geometry</text>
        
        <path d="M 370 50 L 430 50" stroke="#111111" stroke-width="1.5"/>
        
        <rect x="430" y="20" width="160" height="60" fill="none" stroke="#111111" stroke-width="1.5"/>
        <text x="510" y="55" font-family="Space Grotesk" font-size="12" font-weight="600" text-anchor="middle">ITU Atmospheric Loss</text>
        
        <path d="M 590 50 L 650 50" stroke="#111111" stroke-width="1.5"/>
        
        <rect x="650" y="20" width="140" height="60" fill="none" stroke="#111111" stroke-width="1.5"/>
        <text x="720" y="55" font-family="Space Grotesk" font-size="12" font-weight="600" text-anchor="middle">SNR Telemetry</text>

        <rect x="210" y="140" width="160" height="60" fill="none" stroke="#111111" stroke-width="1.5"/>
        <text x="290" y="175" font-family="Space Grotesk" font-size="12" font-weight="600" text-anchor="middle">Maseng-Bakken Rain</text>

        <path d="M 290 140 L 290 80" stroke="#111111" stroke-width="1.5"/>
      </svg>
    `;
    if (caption) caption.textContent = "Figure 1: Forward Observation Model Pipeline Architecture";
  } else if (type === 'impairments') {
    diagramViewport.innerHTML = `
      <div style="display:flex; justify-content:space-around; width:100%; gap:10px; font-family:var(--font-mono); font-size:12px;">
        <div style="border:1px solid #111; padding:12px;">Scintillation (Phase Noise)</div>
        <div style="border:1px solid #111; padding:12px;">Tracking Mispointing Step</div>
        <div style="border:1px solid #111; padding:12px;">Gaseous Absorption Offset</div>
        <div style="border:1px solid #111; padding:12px;">ADC Quantization Noise</div>
      </div>
    `;
    if (caption) caption.textContent = "Figure 2: Receiver & Channel Physical Impairment Taxonomy";
  }
}

/* Interactive Table Custom Data Uploader / Filter */
function initTableUploader() {
  const csvInput = document.getElementById('tableFileInput');
  const customTableContainer = document.getElementById('customTableContainer');

  if (csvInput && customTableContainer) {
    csvInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const rows = text.split('\n').map(r => r.split(','));
        if (rows.length < 1) return;

        let html = '<table class="minimal-table"><thead><tr>';
        rows[0].forEach(cell => {
          html += `<th>${escapeHtml(cell.trim())}</th>`;
        });
        html += '</tr></thead><tbody>';

        for (let i = 1; i < rows.length; i++) {
          if (!rows[i] || rows[i].length < rows[0].length) continue;
          html += '<tr>';
          rows[i].forEach(cell => {
            html += `<td>${escapeHtml(cell.trim())}</td>`;
          });
          html += '</tr>';
        }
        html += '</tbody></table>';

        customTableContainer.innerHTML = html;
      };
      reader.readAsText(file);
    });
  }
}

/* Interactive Impairment Explorer Simulator Widget */
function initImpairmentExplorer() {
  const checkboxes = document.querySelectorAll('.explorer-cb');
  const resR2 = document.getElementById('calcR2');
  const resRMSE = document.getElementById('calcRMSE');
  const resF1 = document.getElementById('calcF1');
  if (checkboxes.length === 0) return;

  function updateMetrics() {
    let activeCount = 0;
    let trackingActive = false;

    checkboxes.forEach(cb => {
      if (cb.checked) {
        activeCount++;
        if (cb.dataset.impairment === 'tracking') trackingActive = true;
      }
    });

    let baseR2 = 0.9588;
    let baseRMSE = 1.5534;
    let baseF1 = 0.9989;

    if (activeCount === 0) {
      // Clean
    } else {
      // Penalty calculation based on empirical regression curve
      let r2Penalty = activeCount * 0.07;
      if (trackingActive) r2Penalty += 0.28; // Tracking dominant drop
      baseR2 = Math.max(-0.15, baseR2 - r2Penalty);

      let rmseAdd = activeCount * 0.75;
      if (trackingActive) rmseAdd += 2.8;
      baseRMSE += rmseAdd;

      let f1Drop = activeCount * 0.04;
      if (trackingActive) f1Drop += 0.08;
      baseF1 = Math.max(0.45, baseF1 - f1Drop);
    }

    if (resR2) resR2.textContent = baseR2.toFixed(4);
    if (resRMSE) resRMSE.textContent = `${baseRMSE.toFixed(4)} mm/h`;
    if (resF1) resF1.textContent = baseF1.toFixed(4);
  }

  checkboxes.forEach(cb => cb.addEventListener('change', updateMetrics));
}

function initMathJaxFormatting() {
  if (window.MathJax) {
    window.MathJax.typesetPromise && window.MathJax.typesetPromise();
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
