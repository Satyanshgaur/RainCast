import { create } from 'zustand';

interface LinkState {
  // Inputs
  frequency: number;   // GHz
  eirp: number;        // dBW
  distance: number;    // km
  elevation: number;   // degrees
  diameter: number;    // meters
  rainRate: number;    // mm/h
  temperature: number; // K
  bandwidth: number;   // MHz

  // Actions
  setFrequency: (val: number) => void;
  setEirp: (val: number) => void;
  setDistance: (val: number) => void;
  setElevation: (val: number) => void;
  setDiameter: (val: number) => void;
  setRainRate: (val: number) => void;
  setTemperature: (val: number) => void;
  setBandwidth: (val: number) => void;
}

export const useLinkStore = create<LinkState>((set) => ({
  // Defaults
  frequency: 14.0,
  eirp: 54.0,
  distance: 35786,
  elevation: 35.0,
  diameter: 1.2,
  rainRate: 0.0,
  temperature: 290,
  bandwidth: 250,

  // Setters
  setFrequency: (val) => set({ frequency: val }),
  setEirp: (val) => set({ eirp: val }),
  setDistance: (val) => set({ distance: val }),
  setElevation: (val) => set({ elevation: val }),
  setDiameter: (val) => set({ diameter: val }),
  setRainRate: (val) => set({ rainRate: val }),
  setTemperature: (val) => set({ temperature: val }),
  setBandwidth: (val) => set({ bandwidth: val }),
}));

// Helper to interpolate specific attenuation coefficients (ITU-R P.838 circular polarization)
export function getITUCoefficients(f: number) {
  const freqs = [10, 14, 20, 30];
  const ks = [0.012, 0.031, 0.075, 0.220];
  const alphas = [1.25, 1.19, 1.10, 1.00];
  
  if (f <= 10) return { k: ks[0], alpha: alphas[0] };
  if (f >= 30) return { k: ks[3], alpha: alphas[3] };
  
  for (let i = 0; i < 3; i++) {
    if (f >= freqs[i] && f <= freqs[i + 1]) {
      const t = (f - freqs[i]) / (freqs[i + 1] - freqs[i]);
      const k = Math.exp(Math.log(ks[i]) + t * (Math.log(ks[i + 1]) - Math.log(ks[i])));
      const alpha = alphas[i] + t * (alphas[i + 1] - alphas[i]);
      return { k, alpha };
    }
  }
  return { k: 0.03, alpha: 1.15 };
}

// Derived properties selector
export function selectLinkBudget(state: LinkState) {
  const { frequency, eirp, distance, elevation, diameter, rainRate, temperature, bandwidth } = state;

  const c = 299792458; // m/s
  const lambda = c / (frequency * 1e9);
  
  // 1. Receiver Antenna Gain
  const rxGain = 10 * Math.log10(0.6 * Math.pow((Math.PI * diameter) / lambda, 2));

  // 2. Free-Space Path Loss
  const fspl = 20 * Math.log10(distance) + 20 * Math.log10(frequency) + 92.45;

  // 3. Gaseous Loss
  const elevRad = (Math.max(elevation, 5) * Math.PI) / 180;
  const gaseousZenith = 0.05 + 0.015 * (frequency - 10);
  const gaseousLoss = gaseousZenith / Math.sin(elevRad);

  // 4. Rain Attenuation on Path
  let rainLoss = 0;
  if (rainRate > 0) {
    const coeffs = getITUCoefficients(frequency);
    const specificAttn = coeffs.k * Math.pow(rainRate, coeffs.alpha);
    const rainHeight = 4.5; // km
    const slantPath = rainHeight / Math.sin(elevRad);
    const pathReduction = 1 / (1 + slantPath / 10.0);
    rainLoss = specificAttn * slantPath * pathReduction;
  }

  // 5. Tropospheric Scintillation Loss
  const sigmaScint = 0.1 * Math.pow(frequency / 12, 7 / 12) * Math.pow(Math.sin(elevRad), -11 / 12) * Math.pow(diameter / 1.2, -5 / 6);
  const scintLoss = 2.33 * sigmaScint;

  // 6. Thermal Noise Floor (k_B * T * B)
  const noise = -228.6 + 10 * Math.log10(temperature) + 10 * Math.log10(bandwidth * 1e6);

  // 7. Received SNR
  const snr = eirp - fspl - gaseousLoss - rainLoss - scintLoss + rxGain - noise;

  // 8. Packet Loss Rate
  const lossPercent = 100 / (1 + Math.exp(0.8 * (snr - 10)));

  // Link Status
  let status: 'Excellent' | 'Marginal' | 'Outage' = 'Excellent';
  if (snr < 10) {
    status = 'Outage';
  } else if (snr < 14) {
    status = 'Marginal';
  }

  return {
    rxGain,
    fspl,
    gaseousLoss,
    rainLoss,
    scintLoss,
    noise,
    snr,
    lossPercent,
    status
  };
}
