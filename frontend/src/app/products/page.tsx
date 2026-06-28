'use client';

import Navbar from '@/components/Navbar';
import { useLinkStore, selectLinkBudget } from '@/store/useLinkStore';
import { motion } from 'framer-motion';

export default function ProductsPage() {
  const store = useLinkStore();
  const derived = selectLinkBudget(store);

  const presets = {
    frequency: [
      { name: '12 GHz (Ku)', value: 12 },
      { name: '14 GHz (Ku)', value: 14 },
      { name: '20 GHz (Ka)', value: 20 },
      { name: '30 GHz (Ka)', value: 30 },
    ],
    distance: [
      { name: 'LEO (600km)', value: 600 },
      { name: 'MEO (8,000km)', value: 8000 },
      { name: 'GEO (35,786km)', value: 35786 },
    ],
    rain: [
      { name: 'Clear', value: 0 },
      { name: 'Light (5 mm/h)', value: 5 },
      { name: 'Heavy (25 mm/h)', value: 25 },
      { name: 'Monsoon (90 mm/h)', value: 90 },
    ],
  };

  return (
    <>
      <Navbar />
      <div className="gradient-wash"></div>

      <main className="page-content min-h-screen relative z-10" style={{ paddingTop: '120px' }}>
        <section className="section-container">
          <div className="section-header">
            <div className="section-eyebrow-container">
              <span className="eyebrow-dot"></span>
              <span className="badge-chip">Atmospheric Product</span>
            </div>
            <h2 className="section-title">Link Budget Calculator</h2>
            <p className="section-subtitle">Real-time analytical propagation solver showing path attenuation and noise levels live.</p>
          </div>

          {/* Interactive Calculator Glass Card */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="calculator-card glass-card"
          >
            {/* Inputs Column */}
            <div className="calc-inputs-panel">
              <h3 className="panel-title">Link Parameters</h3>
              
              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-frequency">Carrier Frequency</label>
                  <span className="input-value">{store.frequency.toFixed(1)} GHz</span>
                </div>
                <input
                  type="range"
                  id="input-frequency"
                  min="10"
                  max="30"
                  step="0.5"
                  value={store.frequency}
                  onChange={(e) => store.setFrequency(parseFloat(e.target.value))}
                />
                <div className="preset-buttons">
                  {presets.frequency.map((p) => (
                    <button
                      key={p.name}
                      type="button"
                      className={`btn-preset ${store.frequency === p.value ? 'active' : ''}`}
                      onClick={() => store.setFrequency(p.value)}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-eirp">Transmitter EIRP</label>
                  <span className="input-value">{store.eirp.toFixed(0)} dBW</span>
                </div>
                <input
                  type="range"
                  id="input-eirp"
                  min="30"
                  max="70"
                  step="1"
                  value={store.eirp}
                  onChange={(e) => store.setEirp(parseFloat(e.target.value))}
                />
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-distance">Slant Range (Distance)</label>
                  <span className="input-value">{store.distance.toLocaleString()} km</span>
                </div>
                <input
                  type="range"
                  id="input-distance"
                  min="500"
                  max="40000"
                  step="100"
                  value={store.distance}
                  onChange={(e) => store.setDistance(parseFloat(e.target.value))}
                />
                <div className="preset-buttons">
                  {presets.distance.map((p) => (
                    <button
                      key={p.name}
                      type="button"
                      className={`btn-preset ${store.distance === p.value ? 'active' : ''}`}
                      onClick={() => store.setDistance(p.value)}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-elevation">Elevation Angle</label>
                  <span className="input-value">{store.elevation.toFixed(0)}°</span>
                </div>
                <input
                  type="range"
                  id="input-elevation"
                  min="5"
                  max="90"
                  step="1"
                  value={store.elevation}
                  onChange={(e) => store.setElevation(parseFloat(e.target.value))}
                />
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-diameter">Receiver Antenna Diameter</label>
                  <span className="input-value">{store.diameter.toFixed(1)} m</span>
                </div>
                <input
                  type="range"
                  id="input-diameter"
                  min="0.5"
                  max="5.0"
                  step="0.1"
                  value={store.diameter}
                  onChange={(e) => store.setDiameter(parseFloat(e.target.value))}
                />
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-rain">Rain Intensity</label>
                  <span className="input-value">{store.rainRate.toFixed(0)} mm/h</span>
                </div>
                <input
                  type="range"
                  id="input-rain"
                  min="0"
                  max="150"
                  step="1"
                  value={store.rainRate}
                  onChange={(e) => store.setRainRate(parseFloat(e.target.value))}
                />
                <div className="preset-buttons">
                  {presets.rain.map((p) => (
                    <button
                      key={p.name}
                      type="button"
                      className={`btn-preset ${store.rainRate === p.value ? 'active' : ''}`}
                      onClick={() => store.setRainRate(p.value)}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-temp">System Noise Temperature</label>
                  <span className="input-value">{store.temperature.toFixed(0)} K</span>
                </div>
                <input
                  type="range"
                  id="input-temp"
                  min="100"
                  max="800"
                  step="10"
                  value={store.temperature}
                  onChange={(e) => store.setTemperature(parseFloat(e.target.value))}
                />
              </div>

              <div className="input-group">
                <div className="input-header">
                  <label htmlFor="input-bandwidth">Signal Bandwidth</label>
                  <span className="input-value">{store.bandwidth.toFixed(0)} MHz</span>
                </div>
                <input
                  type="range"
                  id="input-bandwidth"
                  min="1"
                  max="500"
                  step="5"
                  value={store.bandwidth}
                  onChange={(e) => store.setBandwidth(parseFloat(e.target.value))}
                />
              </div>
            </div>

            {/* Outputs Column */}
            <div className="calc-outputs-panel">
              <div className="summary-row">
                <div className="summary-metric">
                  <span className="metric-label">RECEIVED SNR</span>
                  <div className="metric-value-container">
                    <span className="metric-value font-mono">{derived.snr.toFixed(2)} dB</span>
                    <span className={`status-badge ${derived.status.toLowerCase()}`}>
                      {derived.status}
                    </span>
                  </div>
                </div>
                <div className="summary-metric">
                  <span className="metric-label">PACKET LOSS RATE</span>
                  <span className="metric-value font-mono">{derived.lossPercent.toFixed(2)}%</span>
                </div>
              </div>

              {/* Live Signal Visualizer (SVG Dynamic Illustration) */}
              <div className="visualizer-container">
                <svg className="signal-svg" viewBox="0 0 400 160">
                  {/* Background grid lines */}
                  <line x1="0" y1="40" x2="400" y2="40" stroke="rgba(237, 255, 254, 0.03)" strokeDasharray="2,2" />
                  <line x1="0" y1="80" x2="400" y2="80" stroke="rgba(237, 255, 254, 0.03)" strokeDasharray="2,2" />
                  <line x1="0" y1="120" x2="400" y2="120" stroke="rgba(237, 255, 254, 0.03)" strokeDasharray="2,2" />
                  
                  {/* Rain cloud layers */}
                  <g className="rain-layer" opacity={store.rainRate > 0 ? Math.min(0.2 + (store.rainRate / 100), 1.0) : 0}>
                    <path d="M 280 20 Q 300 10 320 20 Q 340 10 350 25 Q 365 25 365 40 Q 365 50 350 50 L 250 50 Q 235 50 235 40 Q 235 25 255 25 Q 265 15 280 20 Z" fill="var(--color-tide-pool-teal)" opacity="0.8" />
                    {/* Rain lines */}
                    <line x1="260" y1="55" x2="245" y2="140" stroke="#56c2ff" strokeWidth="1.5" strokeDasharray="4,6" className="falling-rain-line" />
                    <line x1="285" y1="55" x2="270" y2="140" stroke="#56c2ff" strokeWidth="1.5" strokeDasharray="4,6" className="falling-rain-line" style={{ animationDelay: '0.1s' }} />
                    <line x1="310" y1="55" x2="295" y2="140" stroke="#56c2ff" strokeWidth="1.5" strokeDasharray="4,6" className="falling-rain-line" style={{ animationDelay: '0.2s' }} />
                    <line x1="335" y1="55" x2="320" y2="140" stroke="#56c2ff" strokeWidth="1.5" strokeDasharray="4,6" className="falling-rain-line" style={{ animationDelay: '0.15s' }} />
                    <line x1="360" y1="55" x2="345" y2="140" stroke="#56c2ff" strokeWidth="1.5" strokeDasharray="4,6" className="falling-rain-line" style={{ animationDelay: '0.05s' }} />
                  </g>

                  {/* Satellite representation */}
                  <g transform="translate(40, 30)">
                    <circle cx="0" cy="0" r="14" fill="#011d1c" stroke="rgba(237, 255, 254, 0.08)" strokeWidth="2" />
                    <rect x="-35" y="-5" width="20" height="10" rx="1" fill="#00827c" stroke="#cbfffc" strokeWidth="1" />
                    <rect x="15" y="-5" width="20" height="10" rx="1" fill="#00827c" stroke="#cbfffc" stroke-width="1" />
                    <path d="M -8 -8 Q -14 0 -8 8 L -4 4 Q -8 0 -4 -4 Z" fill="var(--color-ice-mist)" />
                    <line x1="-12" y1="0" x2="-2" y2="0" stroke="var(--color-ice-mist)" strokeWidth="1.5" />
                  </g>
                  <text x="40" y="60" fill="var(--color-fog-veil)" fontSize="10" fontFamily="Matter" textAnchor="middle">SATELLITE</text>

                  {/* Ground Station representation */}
                  <g transform="translate(340, 120)">
                    <line x1="0" y1="0" x2="0" y2="20" stroke="var(--color-ice-mist)" strokeWidth="3" />
                    <line x1="-10" y1="20" x2="10" y2="20" stroke="var(--color-ice-mist)" strokeWidth="2" />
                    <path d="M -15 -12 Q 0 -5 15 -12" fill="none" stroke="var(--color-ice-mist)" strokeWidth="3" />
                    <line x1="0" y1="-7" x2="0" y2="-15" stroke="var(--color-teal-accent)" strokeWidth="1.5" />
                    <circle cx="0" cy="-15" r="2.5" fill="var(--color-teal-accent)" />
                  </g>
                  <text x="340" y="155" fill="var(--color-fog-veil)" fontSize="10" fontFamily="Matter" textAnchor="middle">GATEWAY</text>

                  {/* Dynamic Signal Propagation Path */}
                  <path
                    d="M 40 30 L 340 120"
                    stroke={derived.status === 'Excellent' ? '#59d499' : derived.status === 'Marginal' ? '#56c2ff' : '#ff6363'}
                    strokeWidth="2.5"
                    strokeDasharray={derived.status === 'Excellent' ? 'none' : derived.status === 'Marginal' ? '5,5' : '2,5'}
                    fill="none"
                  />
                  <circle r="4" fill={derived.status === 'Excellent' ? '#ffffff' : derived.status === 'Marginal' ? '#56c2ff' : '#ff6363'}>
                    <animateMotion dur="2s" repeatCount="indefinite" path="M 40 30 L 340 120" />
                  </circle>
                </svg>
                <div className="visualizer-status">
                  <span className={`indicator-dot ${derived.status === 'Outage' ? 'outage' : 'active'}`}></span>
                  <span>{derived.status === 'Excellent' ? 'Link Healthy (Lock)' : derived.status === 'Marginal' ? 'Link Degraded' : 'Link Outage'}</span>
                </div>
              </div>

              {/* Link Budget Table (Auros Pressed Tonal Structure) */}
              <div className="link-budget-table-container">
                <table className="link-budget-table font-mono">
                  <thead>
                    <tr>
                      <th>Parameter</th>
                      <th className="text-right">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Transmitter EIRP</td>
                      <td className="text-right text-success">+{store.eirp.toFixed(2)} dBW</td>
                    </tr>
                    <tr>
                      <td>Free-Space Path Loss (FSPL)</td>
                      <td className="text-right text-danger">-{derived.fspl.toFixed(2)} dB</td>
                    </tr>
                    <tr>
                      <td>Atmospheric Gaseous Loss</td>
                      <td className="text-right text-danger">-{derived.gaseousLoss.toFixed(2)} dB</td>
                    </tr>
                    <tr>
                      <td>Rain Attenuation</td>
                      <td className={`text-right ${derived.rainLoss > 0 ? 'text-danger' : ''}`}>
                        {derived.rainLoss > 0 ? `-${derived.rainLoss.toFixed(2)} dB` : '0.00 dB'}
                      </td>
                    </tr>
                    <tr>
                      <td>Tropospheric Scintillation Loss</td>
                      <td className="text-right text-danger">-{derived.scintLoss.toFixed(2)} dB</td>
                    </tr>
                    <tr>
                      <td>{"Receiver Antenna Gain ($G_{rx}$)"}</td>
                      <td className="text-right text-success">+{derived.rxGain.toFixed(2)} dBi</td>
                    </tr>
                    <tr>
                      <td>Receiver Noise Floor ($N$)</td>
                      <td className="text-right text-muted">{derived.noise.toFixed(2)} dBW</td>
                    </tr>
                    <tr className="table-total-row">
                      <td>Calculated SNR</td>
                      <td className="text-right text-highlight">{derived.snr.toFixed(2)} dB</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        </section>
      </main>

      <footer className="footer-container">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="brand-name">Raincast</span>
            <span className="footer-meta font-mono">v2.1.0-API</span>
          </div>
          <div className="footer-copy">
            &copy; 2026 Raincast. Interactive Link Modeler.
          </div>
        </div>
      </footer>
    </>
  );
}
