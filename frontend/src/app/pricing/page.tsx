'use client';

import Navbar from '@/components/Navbar';
import { motion } from 'framer-motion';

export default function PricingPage() {
  const tiers = [
    {
      name: 'Free',
      price: '$0',
      frequency: 'per month',
      features: [
        '100 requests / hour',
        '5 concurrent simulations',
        'Max 24h simulation window',
        'Standard community support',
      ],
      isHighlight: false,
    },
    {
      name: 'Developer',
      price: '$99',
      frequency: 'per month',
      features: [
        '2,000 requests / hour',
        '20 concurrent simulations',
        'Max 168h simulation window',
        'Live telemetry API access',
        '24h support response',
      ],
      isHighlight: true,
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      frequency: 'annual contract',
      features: [
        'Unlimited API request limits',
        'Custom concurrency limits',
        'Unlimited simulation duration',
        'Custom orbit propagations',
        'Dedicated engineer SLA',
      ],
      isHighlight: false,
    },
  ];

  return (
    <>
      <Navbar />
      <div className="gradient-wash"></div>

      <main className="page-content min-h-screen relative z-10" style={{ paddingTop: '120px' }}>
        <section className="section-container">
          <div className="section-header">
            <div className="section-eyebrow-container">
              <span className="eyebrow-dot"></span>
              <span className="badge-chip">Pricing Models</span>
            </div>
            <h2 className="section-title">Flexible Pricing Plans</h2>
            <p className="section-subtitle">Choose a plan configured for your operational link density and API workload.</p>
          </div>

          <div className="pricing-grid">
            {tiers.map((tier, index) => (
              <motion.div
                key={tier.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                className={`pricing-card ${tier.isHighlight ? 'card-highlight' : 'card-base'}`}
              >
                <div className="pricing-tier-name">{tier.name}</div>
                <div className="pricing-price">{tier.price}</div>
                <div className="pricing-frequency">{tier.frequency}</div>
                <ul className="pricing-features">
                  {tier.features.map((feature) => (
                    <li key={feature}>{feature}</li>
                  ))}
                </ul>
                <Link
                  href="/products"
                  className={tier.isHighlight ? 'btn-primary text-center w-full' : 'btn-secondary text-center w-full'}
                >
                  {tier.price === 'Custom' ? 'Contact Sales' : tier.price === '$0' ? 'Start Free' : 'Get Developer'}
                </Link>
              </motion.div>
            ))}
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
            &copy; 2026 Raincast. Scalable telemetry narrowcasting tiers.
          </div>
        </div>
      </footer>
    </>
  );
}

import Link from 'next/link';
