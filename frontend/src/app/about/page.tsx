'use client';

import Navbar from '@/components/Navbar';
import { motion } from 'framer-motion';

export default function AboutPage() {
  return (
    <>
      <Navbar />
      <div className="gradient-wash"></div>

      <main className="page-content min-h-screen relative z-10" style={{ paddingTop: '120px' }}>
        <motion.section 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="section-container"
        >
          <div className="section-header">
            <div className="section-eyebrow-container">
              <span className="eyebrow-dot"></span>
              <span className="badge-chip">Science & Research</span>
            </div>
            <h2 className="section-title">Physics-Informed Narrowcasting</h2>
            <p className="section-subtitle">Bridging atmospheric physics and machine learning to infer rainfall rates from satellite telemetry.</p>
          </div>

          <div className="glass-card p-10 mb-10">
            <h3 className="panel-title text-xl mb-4">The Core Research</h3>
            <p className="text-color-fog-veil text-sm leading-relaxed mb-5">
              Tropospheric precipitation causes substantial microwave signal attenuation in satellite links, especially at carrier bands above 10 GHz (Ku and Ka-bands). Traditional precipitation monitoring relies on expensive rain gauges or spatial weather radars. 
            </p>
            <p className="text-color-fog-veil text-sm leading-relaxed mb-5">
              <strong>Raincast</strong> investigates an alternative: can we leverage satellite receivers as distributed, real-time weather sensors? By mapping signal-to-noise ratio (SNR) decays, our framework reconstructs rainfall intensities along propagation slant paths.
            </p>
            
            <h3 className="panel-title text-xl mt-8 mb-4">Identifying & Correcting Simulator Biases</h3>
            <p className="text-color-fog-veil text-sm leading-relaxed mb-5">
              During the validation of our stochastic propagation simulator, we discovered and corrected two crucial statistical biases that caused the generator to underestimate extreme rain exceedances by 15% to 50%:
            </p>
            <ul className="pl-6 text-color-fog-veil text-sm mb-5 leading-relaxed list-disc">
              <li className="mb-2">
                <strong>Quantile Probit Fitting Error</strong>: Replaced static 10% annual rain fraction assumptions with local, latitude-dependent standard normal CDF quantiles.
              </li>
              <li className="mb-2">
                <strong>Temporal Markov Reset Bias</strong>: Corrected event onset initializations to draw random noise scaled by local variance parameters rather than resetting events to the lognormal median.
              </li>
            </ul>
            <p className="text-color-fog-veil text-sm leading-relaxed">
              Delhi exceedance target replication accuracy improved from 57.5% to 98.0% (41.16 mm/h modeled vs. 42.00 mm/h theoretical ITU target) following these corrections.
            </p>
          </div>
        </motion.section>
      </main>

      <footer className="footer-container relative z-10">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="brand-name">Raincast</span>
            <span className="footer-meta font-mono">v2.1.0-API</span>
          </div>
          <div className="footer-copy">
            &copy; 2026 Raincast. Physics-first propagation research.
          </div>
        </div>
      </footer>
    </>
  );
}
