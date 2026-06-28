'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';

export default function Home() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: [0.23, 1, 0.32, 1], // outQuint
      },
    },
  };

  const sphereVariants = {
    hidden: { scale: 0.8, opacity: 0 },
    visible: {
      scale: 1,
      opacity: 1,
      transition: {
        delay: 0.4,
        duration: 1.2,
        ease: 'easeOut',
      },
    },
  };

  return (
    <>
      <Navbar />
      <div className="gradient-wash"></div>

      <main className="page-content min-h-screen relative z-10">
        {/* Hero Section */}
        <section className="hero-section flex flex-col items-center text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="flex flex-col items-center"
          >
            <motion.div variants={itemVariants} className="announcement-badge">
              <span className="badge-text"><span className="badge-version">AUROS</span> Platform Engine v2.1.0</span>
            </motion.div>

            <motion.h1 
              variants={itemVariants} 
              className="hero-title select-none"
            >
              Weather-Aware<br />Satellite Links.
            </motion.h1>

            <motion.p 
              variants={itemVariants} 
              className="hero-subtitle max-w-[600px] text-color-fog-veil mb-9"
            >
              A physics-first satellite communication simulator and machine learning narrowcasting framework for reconstructing rain rates from observed SNR telemetry.
            </motion.p>

            <motion.div variants={itemVariants} className="hero-ctas flex gap-5 mb-16">
              <Link href="/products" className="btn-primary btn-large">
                Interactive Calculator
              </Link>
              <Link href="/docs" className="btn-secondary btn-large">
                Read Documentation
              </Link>
            </motion.div>
          </motion.div>

          {/* Bioluminescent Particle Sphere centerpiece */}
          <motion.div
            variants={sphereVariants}
            initial="hidden"
            animate="visible"
            className="particle-sphere-wrapper"
          >
            <svg className="particle-sphere-svg w-full h-full" viewBox="0 0 300 300">
              <circle cx="150" cy="150" r="110" fill="none" stroke="rgba(203, 255, 252, 0.15)" strokeWidth="0.75" strokeDasharray="2,12" className="rotator" style={{ animation: 'rotate 24s linear infinite' }} />
              <circle cx="150" cy="150" r="85" fill="none" stroke="rgba(0, 130, 124, 0.25)" strokeWidth="0.75" strokeDasharray="6,8" className="rotator" style={{ animation: 'rotate-back 18s linear infinite' }} />
              <circle cx="150" cy="150" r="60" fill="none" stroke="rgba(203, 255, 252, 0.1)" strokeWidth="0.5" strokeDasharray="4,4" className="rotator" style={{ animation: 'rotate 12s linear infinite' }} />
              
              <g fill="#cbfffc" opacity="0.95">
                <circle cx="150" cy="150" r="3" className="pulse-particle" />
                <circle cx="120" cy="110" r="1.5" className="pulse-particle" style={{ animationDelay: '0.3s' }} />
                <circle cx="180" cy="190" r="1.5" className="pulse-particle" style={{ animationDelay: '0.6s' }} />
                <circle cx="100" cy="150" r="2" className="pulse-particle" style={{ animationDelay: '0.15s' }} />
                <circle cx="200" cy="150" r="2" className="pulse-particle" style={{ animationDelay: '0.45s' }} />
                <circle cx="160" cy="90" r="1.5" className="pulse-particle" style={{ animationDelay: '0.75s' }} />
                <circle cx="140" cy="210" r="1.5" className="pulse-particle" style={{ animationDelay: '0.35s' }} />
              </g>
              
              <path d="M 120 110 L 150 150 L 180 190 M 100 150 L 150 150 L 200 150 M 160 90 L 150 150 L 140 210" stroke="rgba(203, 255, 252, 0.2)" strokeWidth="0.75" fill="none" />
            </svg>
          </motion.div>
        </section>

        {/* What Problem Does It Solve? */}
        <section className="section-container">
          <div className="section-header">
            <div className="section-eyebrow-container">
              <span className="eyebrow-dot"></span>
              <span className="badge-chip">The Problem</span>
            </div>
            <h2 className="section-title">Tropospheric Rain Attenuation</h2>
            <p className="section-subtitle">Satellite communication systems observe continuous signal degradation caused by atmospheric rain fade. Can we reconstruct rainfall intensity directly from receiver telemetry?</p>
          </div>

          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#edfffe" strokeWidth="2">
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                </svg>
              </div>
              <h3 className="feature-card-title">High-Fidelity Physics</h3>
              <p className="feature-card-desc">SGP4 orbital propagation, dynamic slant ranges, and ITU-R P.618/P.676 atmospheric models evaluate link path fading on-the-fly.</p>
            </div>

            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#edfffe" strokeWidth="2">
                  <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
                  <line x1="4" y1="22" x2="4" y2="15" />
                </svg>
              </div>
              <h3 className="feature-card-title">Stochastic Dynamics</h3>
              <p className="feature-card-desc">Integrates the Maseng-Bakken AR(1) lognormal model compiled with Numba JIT to synthesize temporally-correlated rain events.</p>
            </div>

            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#edfffe" strokeWidth="2">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                </svg>
              </div>
              <h3 className="feature-card-title">ML Narrowcasting</h3>
              <p className="feature-card-desc">A physics-aware XGBoost cascade isolates scintillation noise from rain fades, reconstructing rain rates within 0.28 mm/h accuracy.</p>
            </div>
          </div>
        </section>

        {/* Highlights & Results */}
        <section className="section-container">
          <div className="section-header">
            <div className="section-eyebrow-container">
              <span className="eyebrow-dot"></span>
              <span className="badge-chip">Research Contributions</span>
            </div>
            <h2 className="section-title">Key Performance Benchmarks</h2>
            <p className="section-subtitle">Evaluating narrowcasting stages and simulator corrections against theoretical targets.</p>
          </div>

          <div className="glass-card table-panel p-10">
            <h3 className="panel-title mb-6">Narrowcaster Regression Accuracy</h3>
            <div className="link-budget-table-container">
              <table className="link-budget-table font-mono">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="text-center">Stage A (Analytical)</th>
                    <th className="text-center">Stage B (XGBoost)</th>
                    <th className="text-center text-success">Stage C (Frequency-Aware)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>F1 Score (Rain Detection)</td>
                    <td className="text-center">0.163</td>
                    <td className="text-center">0.999</td>
                    <td className="text-center text-highlight">0.999</td>
                  </tr>
                  <tr>
                    <td>RMSE (mm/h)</td>
                    <td className="text-center text-danger">2.10</td>
                    <td className="text-center">0.49</td>
                    <td className="text-center text-success">0.28</td>
                  </tr>
                  <tr>
                    <td>R² Coefficient</td>
                    <td className="text-center">0.111</td>
                    <td className="text-center">0.995</td>
                    <td className="text-center text-highlight">0.998</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="table-caption text-xs text-color-fog-veil mt-4 leading-relaxed tracking-wider">
              *Stage A relies on pure mathematical inversion of the path attenuation. Stage B utilizes rolling window statistical features. Stage C incorporates specific frequency physics coefficients ($k$ and $\alpha$) to generalize across the 10–30 GHz bands.
            </p>
          </div>
        </section>
      </main>

      <footer className="footer-container relative z-10">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="brand-name">Raincast</span>
            <span className="footer-meta font-mono">v2.1.0-API</span>
          </div>
          <div className="footer-copy">
            &copy; 2026 Raincast. Open-source satellite link narrowcasting.
          </div>
        </div>
      </footer>
    </>
  );
}
