document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const elements = {
        // Inputs
        freqSlider: document.getElementById('input-frequency'),
        eirpSlider: document.getElementById('input-eirp'),
        distSlider: document.getElementById('input-distance'),
        elevSlider: document.getElementById('input-elevation'),
        diamSlider: document.getElementById('input-diameter'),
        rainSlider: document.getElementById('input-rain'),
        tempSlider: document.getElementById('input-temp'),
        bwSlider: document.getElementById('input-bandwidth'),

        // Input Value Displays
        freqVal: document.getElementById('val-frequency'),
        eirpVal: document.getElementById('val-eirp'),
        distVal: document.getElementById('val-distance'),
        elevVal: document.getElementById('val-elevation'),
        diamVal: document.getElementById('val-diameter'),
        rainVal: document.getElementById('val-rain'),
        tempVal: document.getElementById('val-temp'),
        bwVal: document.getElementById('val-bandwidth'),

        // Live Outputs
        outSnr: document.getElementById('out-snr'),
        outStatus: document.getElementById('out-status'),
        outLoss: document.getElementById('out-loss'),

        // Budget Table Cells
        tableEirp: document.getElementById('table-eirp'),
        tableFspl: document.getElementById('table-fspl'),
        tableGas: document.getElementById('table-gas'),
        tableRain: document.getElementById('table-rain'),
        tableScint: document.getElementById('table-scint'),
        tableRxGain: document.getElementById('table-rx-gain'),
        tableNoise: document.getElementById('table-noise'),
        tableSnr: document.getElementById('table-snr'),

        // Visualizer Elements
        signalPath: document.getElementById('signal-path'),
        signalPulse: document.getElementById('signal-pulse'),
        indicatorDot: document.getElementById('indicator-status-dot'),
        statusText: document.getElementById('visualizer-status-text'),
        rainClouds: document.getElementById('svg-rain-clouds')
    };

    // Preset Buttons Listener Setup
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetId = btn.getAttribute('data-target');
            const targetVal = btn.getAttribute('data-value');
            
            // Update input and trigger change
            const slider = document.getElementById(targetId);
            if (slider) {
                slider.value = targetVal;
                
                // Toggle active class inside the parent preset group
                btn.parentNode.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                calculateLinkBudget();
            }
        });
    });

    // Helper to interpolate specific attenuation coefficients (ITU-R P.838 circular polarization)
    function getITUCoefficients(f) {
        const freqs = [10, 14, 20, 30];
        const ks = [0.012, 0.031, 0.075, 0.220];
        const alphas = [1.25, 1.19, 1.10, 1.00];
        
        if (f <= 10) return { k: ks[0], alpha: alphas[0] };
        if (f >= 30) return { k: ks[3], alpha: alphas[3] };
        
        for (let i = 0; i < 3; i++) {
            if (f >= freqs[i] && f <= freqs[i+1]) {
                const t = (f - freqs[i]) / (freqs[i+1] - freqs[i]);
                const k = Math.exp(Math.log(ks[i]) + t * (Math.log(ks[i+1]) - Math.log(ks[i])));
                const alpha = alphas[i] + t * (alphas[i+1] - alphas[i]);
                return { k, alpha };
            }
        }
        return { k: 0.03, alpha: 1.15 };
    }

    // Main Link Budget Calculation Function
    function calculateLinkBudget() {
        // Read active inputs
        const freq = parseFloat(elements.freqSlider.value); // GHz
        const eirp = parseFloat(elements.eirpSlider.value); // dBW
        const dist = parseFloat(elements.distSlider.value); // km
        const elev = parseFloat(elements.elevSlider.value); // degrees
        const diam = parseFloat(elements.diamSlider.value); // meters
        const rain = parseFloat(elements.rainSlider.value); // mm/h
        const temp = parseFloat(elements.tempSlider.value); // K
        const bw = parseFloat(elements.bwSlider.value); // MHz

        // Update Slider Label Values
        elements.freqVal.textContent = `${freq.toFixed(1)} GHz`;
        elements.eirpVal.textContent = `${eirp.toFixed(0)} dBW`;
        elements.distVal.textContent = `${dist.toLocaleString()} km`;
        elements.elevVal.textContent = `${elev.toFixed(0)}°`;
        elements.diamVal.textContent = `${diam.toFixed(1)} m`;
        elements.rainVal.textContent = `${rain.toFixed(0)} mm/h`;
        elements.tempVal.textContent = `${temp.toFixed(0)} K`;
        elements.bwVal.textContent = `${bw.toFixed(0)} MHz`;

        // 1. Wavelength & Rx Antenna Gain calculation
        const c = 299792458; // Speed of light m/s
        const lambda = c / (freq * 1e9);
        const rxGain = 10 * Math.log10(0.6 * Math.pow((Math.PI * diam) / lambda, 2));

        // 2. Free-Space Path Loss (FSPL)
        const fspl = 20 * Math.log10(dist) + 20 * Math.log10(freq) + 92.45;

        // 3. Atmospheric Gaseous Loss
        const elevRad = (Math.max(elev, 5) * Math.PI) / 180;
        const gaseousZenith = 0.05 + 0.015 * (freq - 10);
        const gaseousLoss = gaseousZenith / Math.sin(elevRad);

        // 4. Rain Attenuation (ITU-R P.618/P.838 specific model)
        let rainLoss = 0;
        if (rain > 0) {
            const coeffs = getITUCoefficients(freq);
            const specificAttn = coeffs.k * Math.pow(rain, coeffs.alpha); // dB/km
            const rainHeight = 4.5; // km
            const slantPath = rainHeight / Math.sin(elevRad);
            const pathReduction = 1 / (1 + slantPath / 10.0);
            rainLoss = specificAttn * slantPath * pathReduction;
        }

        // 5. Tropospheric Scintillation Loss
        const sigmaScint = 0.1 * Math.pow(freq / 12, 7/12) * Math.pow(Math.sin(elevRad), -11/12) * Math.pow(diam / 1.2, -5/6);
        const scintLoss = 2.33 * sigmaScint;

        // 6. Thermal Noise Floor (k * T * B)
        const noise = -228.6 + 10 * Math.log10(temp) + 10 * Math.log10(bw * 1e6);

        // 7. Received SNR
        const snr = eirp - fspl - gaseousLoss - rainLoss - scintLoss + rxGain - noise;

        // 8. Packet Loss Rate
        const lossPercent = 100 / (1 + Math.exp(0.8 * (snr - 10)));

        // Output Display Formatting
        elements.outSnr.textContent = `${snr.toFixed(2)} dB`;
        elements.outLoss.textContent = `${lossPercent.toFixed(2)}%`;

        // Update Link Status Badges & Colors
        let status = 'Excellent';
        let statusClass = 'excellent';
        
        if (snr < 10) {
            status = 'Outage';
            statusClass = 'outage';
        } else if (snr < 14) {
            status = 'Marginal';
            statusClass = 'marginal';
        }

        elements.outStatus.textContent = status;
        elements.outStatus.className = `status-badge ${statusClass}`;

        // Table updates
        elements.tableEirp.textContent = `+${eirp.toFixed(2)} dBW`;
        elements.tableFspl.textContent = `-${fspl.toFixed(2)} dB`;
        elements.tableGas.textContent = `-${gaseousLoss.toFixed(2)} dB`;
        elements.tableRain.textContent = rainLoss > 0 ? `-${rainLoss.toFixed(2)} dB` : `0.00 dB`;
        elements.tableRain.className = rainLoss > 0 ? 'text-right text-danger' : 'text-right';
        elements.tableScint.textContent = `-${scintLoss.toFixed(2)} dB`;
        elements.tableRxGain.textContent = `+${rxGain.toFixed(2)} dBi`;
        elements.tableNoise.textContent = `${noise.toFixed(2)} dBW`;
        elements.tableSnr.textContent = `${snr.toFixed(2)} dB`;

        // 9. Interactive Visualizer SVG/Status Animations
        // Update rain cloud opacity
        if (rain > 0) {
            elements.rainClouds.setAttribute('opacity', Math.min(0.2 + (rain / 100), 1.0).toString());
        } else {
            elements.rainClouds.setAttribute('opacity', '0');
        }

        // Animate path stroke color and status indicators
        if (status === 'Excellent') {
            elements.signalPath.setAttribute('stroke', '#59d499'); // Green
            elements.signalPath.setAttribute('stroke-dasharray', 'none');
            elements.signalPulse.setAttribute('fill', '#ffffff');
            elements.indicatorDot.className = 'indicator-dot active';
            elements.statusText.textContent = 'Link Healthy (Lock)';
        } else if (status === 'Marginal') {
            elements.signalPath.setAttribute('stroke', '#56c2ff'); // Sky blue
            elements.signalPath.setAttribute('stroke-dasharray', '5,5');
            elements.signalPulse.setAttribute('fill', '#56c2ff');
            elements.indicatorDot.className = 'indicator-dot active';
            elements.statusText.textContent = 'Link Degraded';
        } else {
            elements.signalPath.setAttribute('stroke', '#ff6363'); // Red
            elements.signalPath.setAttribute('stroke-dasharray', '2,5');
            elements.signalPulse.setAttribute('fill', '#ff6363');
            elements.indicatorDot.className = 'indicator-dot outage';
            elements.statusText.textContent = 'Link Outage';
        }
    }

    // Attach Input Event Listeners for Live Updates
    const inputControls = [
        elements.freqSlider, elements.eirpSlider, elements.distSlider,
        elements.elevSlider, elements.diamSlider, elements.rainSlider,
        elements.tempSlider, elements.bwSlider
    ];

    inputControls.forEach(control => {
        if (control) {
            control.addEventListener('input', () => {
                // Clear active status on presets if input slider is adjusted directly
                const targetId = control.id;
                document.querySelectorAll(`.btn-preset[data-target="${targetId}"]`).forEach(btn => {
                    btn.classList.remove('active');
                });
                calculateLinkBudget();
            });
        }
    });

    // Developer Tab View Actions
    document.querySelectorAll('.btn-doc-tab').forEach(tabBtn => {
        tabBtn.addEventListener('click', () => {
            const targetTab = tabBtn.getAttribute('data-tab');
            
            // Switch tabs active classes
            document.querySelectorAll('.btn-doc-tab').forEach(b => b.classList.remove('active'));
            tabBtn.classList.add('active');

            // Switch content active classes
            document.querySelectorAll('.docs-tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Perform Initial Run
    calculateLinkBudget();
});
