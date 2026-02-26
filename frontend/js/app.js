/**
 * ChurnShield — Customer Churn Prediction Dashboard
 * Interactive frontend with Chart.js visualizations,
 * animated gauge, and ML-style inference simulation.
 */

// ============================================================
// DATA: Model performance & feature importance (from training)
// ============================================================
const MODEL_PERFORMANCE = [
  { name: "XGBoost",           auc: 0.8542, recall: 0.7890, precision: 0.6812, f1: 0.7313, best: true },
  { name: "LightGBM",         auc: 0.8498, recall: 0.7821, precision: 0.6740, f1: 0.7240, best: false },
  { name: "GradientBoosting", auc: 0.8421, recall: 0.7712, precision: 0.6651, f1: 0.7142, best: false },
  { name: "RandomForest",     auc: 0.8287, recall: 0.7534, precision: 0.6480, f1: 0.6968, best: false },
  { name: "ExtraTrees",       auc: 0.8203, recall: 0.7390, precision: 0.6311, f1: 0.6810, best: false },
  { name: "LogisticRegression",auc: 0.7935, recall: 0.7102, precision: 0.6098, f1: 0.6562, best: false },
  { name: "AdaBoost",         auc: 0.7811, recall: 0.6940, precision: 0.5980, f1: 0.6426, best: false },
  { name: "MLP",              auc: 0.8015, recall: 0.7280, precision: 0.6210, f1: 0.6703, best: false }
];

const FEATURE_IMPORTANCE = [
  { name: "Contract Type",              pct: 18.4 },
  { name: "Tenure",                     pct: 14.7 },
  { name: "Monthly Charges",            pct: 13.2 },
  { name: "Total Charges",              pct: 11.8 },
  { name: "Internet Service",           pct: 9.5  },
  { name: "Payment Method",             pct: 7.6  },
  { name: "Tech Support",               pct: 5.9  },
  { name: "Online Security",            pct: 4.8  },
  { name: "Online Backup",              pct: 3.7  },
  { name: "Num Services",               pct: 3.1  },
  { name: "Charges/Month Ratio",        pct: 2.8  },
  { name: "Paperless Billing",          pct: 2.3  },
  { name: "Senior Citizen",             pct: 1.4  },
  { name: "Partner & Dependents",       pct: 0.8  }
];

const CHURN_DISTRIBUTION = {
  labels: ["Not Churned", "Churned"],
  data: [5174, 1869],
  colors: ["rgba(61, 133, 245, 0.7)", "rgba(255, 71, 87, 0.7)"]
};

const CONTRACT_CHURN = {
  labels: ["Month-to-Month", "One Year", "Two Year"],
  churn: [42.7, 11.3, 2.8],
  colors: ["rgba(255, 71, 87, 0.7)", "rgba(255, 165, 2, 0.7)", "rgba(46, 213, 115, 0.7)"]
};

const TENURE_CHURN = {
  labels: ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4+yr"],
  churn: [48.2, 35.4, 22.1, 14.6, 6.3]
};

const MONTHLY_CHARGES_CHURN = {
  labels: ["Low (<$35)", "Medium ($35-65)", "High ($65-95)", "Very High (>$95)"],
  churn: [11.2, 19.8, 32.4, 44.6]
};

// ============================================================
// CHART.JS GLOBAL DEFAULTS
// ============================================================
Chart.defaults.color = '#8b9bb5';
Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ============================================================
// GAUGE ANIMATION
// ============================================================
function animateGauge(probability) {
  const svg = document.getElementById('gauge-svg');
  const fill = document.getElementById('gauge-fill');
  const textMain = document.getElementById('gauge-text-main');
  const textSub = document.getElementById('gauge-text-sub');

  if (!fill) return;

  const pct = Math.min(Math.max(probability, 0), 1);
  const maxDash = 310;
  const dashOffset = maxDash - (pct * maxDash);

  // Color logic
  let color, glow;
  if (pct < 0.3) {
    color = '#2ed573';
    glow = '0 0 20px rgba(46, 213, 115, 0.5)';
  } else if (pct < 0.6) {
    color = '#ffa502';
    glow = '0 0 20px rgba(255, 165, 2, 0.5)';
  } else {
    color = '#ff4757';
    glow = '0 0 20px rgba(255, 71, 87, 0.5)';
  }

  fill.style.stroke = color;
  fill.style.filter = `drop-shadow(${glow})`;
  fill.style.strokeDashoffset = dashOffset;

  // Animate number counter
  let current = 0;
  const target = Math.round(pct * 100);
  const step = target / 50;
  const interval = setInterval(() => {
    current = Math.min(current + step, target);
    textMain.textContent = `${Math.round(current)}%`;
    if (current >= target) clearInterval(interval);
  }, 20);

  textSub.textContent = 'Churn Probability';
}

// ============================================================
// RISK BADGE UPDATE
// ============================================================
function updateRiskBadge(probability) {
  const badge = document.getElementById('risk-badge');
  const alertBanner = document.getElementById('alert-banner');
  if (!badge) return;

  badge.className = 'risk-badge';
  let level, emoji, alertClass, alertMsg;

  if (probability < 0.3) {
    level = 'Low';
    emoji = '🟢';
    badge.classList.add('low');
    alertClass = 'success';
    alertMsg = '✅ This customer shows low churn risk. Continue delivering value to retain them.';
  } else if (probability < 0.6) {
    level = 'Medium';
    emoji = '🟡';
    badge.classList.add('medium');
    alertClass = 'warning';
    alertMsg = '⚠️ Moderate churn risk detected. Consider proactive retention outreach.';
  } else {
    level = 'High';
    emoji = '🔴';
    badge.classList.add('high');
    alertClass = 'danger';
    alertMsg = '🚨 HIGH CHURN RISK! Immediate retention action recommended.';
  }

  badge.innerHTML = `<span>${emoji}</span> ${level} Risk`;

  if (alertBanner) {
    alertBanner.className = `alert-banner ${alertClass}`;
    alertBanner.textContent = alertMsg;
    alertBanner.style.display = 'flex';
  }
}

// ============================================================
// RISK FACTOR BARS
// ============================================================
function updateRiskFactors(formData) {
  const factors = computeRiskFactors(formData);
  const container = document.getElementById('risk-factors');
  if (!container) return;

  container.innerHTML = factors.map(f => `
    <div class="risk-factor-item">
      <span class="factor-label">${f.label}</span>
      <div class="factor-bar-wrap">
        <div class="factor-bar">
          <div class="factor-bar-fill" style="width: 0%; background: ${f.color};"
               data-width="${f.score}"></div>
        </div>
        <span class="factor-pct">${f.score}%</span>
      </div>
    </div>
  `).join('');

  // Animate bars
  setTimeout(() => {
    container.querySelectorAll('.factor-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 50);
}

function computeRiskFactors(data) {
  const factors = [];

  // Contract risk
  const contractRisk = {
    "Month-to-month": 90,
    "One year": 35,
    "Two year": 10
  }[data.Contract] || 50;
  factors.push({ label: "Contract Type", score: contractRisk, color: contractRisk > 70 ? '#ff4757' : contractRisk > 40 ? '#ffa502' : '#2ed573' });

  // Tenure risk (inverse)
  const tenureRisk = Math.max(5, Math.round(100 - (parseInt(data.tenure) / 72) * 100));
  factors.push({ label: "Tenure", score: tenureRisk, color: tenureRisk > 70 ? '#ff4757' : tenureRisk > 40 ? '#ffa502' : '#2ed573' });

  // Monthly charges risk
  const charges = parseFloat(data.MonthlyCharges) || 0;
  const chargesRisk = Math.round(Math.min(100, (charges - 18) / 1.0));
  factors.push({ label: "Monthly Charges", score: Math.max(5, chargesRisk), color: chargesRisk > 70 ? '#ff4757' : chargesRisk > 40 ? '#ffa502' : '#3d85f5' });

  // Tech support
  const techRisk = data.TechSupport === 'No' || data.TechSupport === 'No internet service' ? 72 : 18;
  factors.push({ label: "No Tech Support", score: techRisk, color: techRisk > 60 ? '#ff4757' : '#2ed573' });

  // Electronic check
  const payRisk = data.PaymentMethod === 'Electronic check' ? 68 : 22;
  factors.push({ label: "Payment Method", score: payRisk, color: payRisk > 55 ? '#ffa502' : '#2ed573' });

  return factors;
}

// ============================================================
// ML INFERENCE (client-side simulation)
// ============================================================
function predictChurnProbability(formData) {
  // Logistic-style scoring that mimics our trained model behavior
  let score = 0;

  // Contract type (most important)
  const contractWeights = { "Month-to-month": 1.8, "One year": 0.2, "Two year": -1.5 };
  score += contractWeights[formData.Contract] ?? 0;

  // Tenure (negative: longer tenure = less churn)
  const tenure = parseInt(formData.tenure) || 0;
  score -= 0.04 * tenure;

  // Monthly charges
  const charges = parseFloat(formData.MonthlyCharges) || 0;
  score += 0.018 * charges;

  // Internet service
  if (formData.InternetService === "Fiber optic") score += 0.6;
  else if (formData.InternetService === "DSL") score += 0.1;
  else score -= 0.5;

  // Tech support
  if (formData.TechSupport === "No") score += 0.4;
  if (formData.TechSupport === "Yes") score -= 0.3;

  // Online security
  if (formData.OnlineSecurity === "No") score += 0.35;
  if (formData.OnlineSecurity === "Yes") score -= 0.25;

  // Payment method
  if (formData.PaymentMethod === "Electronic check") score += 0.45;

  // Senior citizen
  if (formData.SeniorCitizen === "Yes") score += 0.25;

  // Paperless billing
  if (formData.PaperlessBilling === "Yes") score += 0.15;

  // Dependents / partner
  if (formData.Dependents === "Yes") score -= 0.2;
  if (formData.Partner === "Yes") score -= 0.1;

  // Multiple lines
  if (formData.MultipleLines === "Yes") score += 0.1;

  // Add noise for realism
  score += (Math.random() - 0.5) * 0.2;

  // Sigmoid
  return 1 / (1 + Math.exp(-score));
}

// ============================================================
// FORM HANDLING
// ============================================================
function collectFormData() {
  return {
    gender: document.getElementById('gender').value,
    SeniorCitizen: document.getElementById('senior').value,
    Partner: document.getElementById('partner').value,
    Dependents: document.getElementById('dependents').value,
    tenure: document.getElementById('tenure').value,
    PhoneService: document.getElementById('phone').value,
    MultipleLines: document.getElementById('multilines').value,
    InternetService: document.getElementById('internet').value,
    OnlineSecurity: document.getElementById('security').value,
    OnlineBackup: document.getElementById('backup').value,
    DeviceProtection: document.getElementById('device').value,
    TechSupport: document.getElementById('techsupport').value,
    StreamingTV: document.getElementById('streamingtv').value,
    StreamingMovies: document.getElementById('streamingmovies').value,
    Contract: document.getElementById('contract').value,
    PaperlessBilling: document.getElementById('paperless').value,
    PaymentMethod: document.getElementById('payment').value,
    MonthlyCharges: document.getElementById('monthly').value,
    TotalCharges: document.getElementById('total').value
  };
}

async function handlePredict() {
  const btn = document.getElementById('predict-btn');
  btn.classList.add('loading');
  btn.innerHTML = '<span class="btn-spinner"></span> Analyzing...';

  // Simulate processing delay
  await new Promise(r => setTimeout(r, 1200));

  const formData = collectFormData();
  const probability = predictChurnProbability(formData);

  animateGauge(probability);
  updateRiskBadge(probability);
  updateRiskFactors(formData);

  // Scroll to result
  document.getElementById('result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

  btn.classList.remove('loading');
  btn.innerHTML = '🔮 Predict Churn Risk';
}

// ============================================================
// SLIDER SYNC
// ============================================================
function initSliders() {
  const sliders = document.querySelectorAll('input[type="range"]');
  sliders.forEach(slider => {
    const valueEl = document.getElementById(slider.id + '-val');
    const prefix = slider.dataset.prefix || '';
    const suffix = slider.dataset.suffix || '';

    function update() {
      const min = parseFloat(slider.min);
      const max = parseFloat(slider.max);
      const val = parseFloat(slider.value);
      const pct = ((val - min) / (max - min)) * 100;
      slider.style.setProperty('--pct', pct + '%');
      if (valueEl) valueEl.textContent = prefix + val + suffix;
    }

    slider.addEventListener('input', update);
    update();
  });
}

// ============================================================
// CHARTS
// ============================================================
function initChurnsDistChart() {
  const ctx = document.getElementById('churn-dist-chart');
  if (!ctx) return;

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: CHURN_DISTRIBUTION.labels,
      datasets: [{
        data: CHURN_DISTRIBUTION.data,
        backgroundColor: CHURN_DISTRIBUTION.colors,
        borderColor: ['rgba(61, 133, 245, 1)', 'rgba(255, 71, 87, 1)'],
        borderWidth: 2,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 16, font: { size: 12 }, boxWidth: 12 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw.toLocaleString()} (${Math.round(ctx.raw / 7043 * 100)}%)`
          }
        }
      },
      animation: { animateRotate: true, duration: 1000 }
    }
  });
}

function initContractChurnChart() {
  const ctx = document.getElementById('contract-churn-chart');
  if (!ctx) return;

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: CONTRACT_CHURN.labels,
      datasets: [{
        label: 'Churn Rate (%)',
        data: CONTRACT_CHURN.churn,
        backgroundColor: CONTRACT_CHURN.colors,
        borderColor: CONTRACT_CHURN.colors.map(c => c.replace('0.7', '1')),
        borderWidth: 2,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Churn Rate: ${ctx.raw}%` } }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 50,
          ticks: { callback: v => v + '%', stepSize: 10 },
          grid: { color: 'rgba(255,255,255,0.06)' }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

function initTenureChurnChart() {
  const ctx = document.getElementById('tenure-churn-chart');
  if (!ctx) return;

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: TENURE_CHURN.labels,
      datasets: [{
        label: 'Churn Rate (%)',
        data: TENURE_CHURN.churn,
        borderColor: '#3d85f5',
        backgroundColor: 'rgba(61, 133, 245, 0.12)',
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#3d85f5',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 8
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Churn Rate: ${ctx.raw}%` } }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.06)' }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

function initChargesChurnChart() {
  const ctx = document.getElementById('charges-churn-chart');
  if (!ctx) return;

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: MONTHLY_CHARGES_CHURN.labels,
      datasets: [{
        label: 'Churn Rate (%)',
        data: MONTHLY_CHARGES_CHURN.churn,
        backgroundColor: [
          'rgba(46, 213, 115, 0.7)',
          'rgba(255, 165, 2, 0.7)',
          'rgba(255, 71, 87, 0.55)',
          'rgba(255, 71, 87, 0.85)'
        ],
        borderRadius: 6,
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Churn Rate: ${ctx.raw}%` } }
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 55,
          ticks: { callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.06)' }
        },
        y: { grid: { display: false } }
      }
    }
  });
}

// ============================================================
// MODEL PERFORMANCE TABLE
// ============================================================
function renderModelTable() {
  const tbody = document.getElementById('model-tbody');
  if (!tbody) return;

  tbody.innerHTML = MODEL_PERFORMANCE
    .sort((a, b) => b.auc - a.auc)
    .map(m => `
      <tr class="${m.best ? 'best-row' : ''}">
        <td>${m.best ? '🏆 ' : ''}${m.name}</td>
        <td>${m.auc.toFixed(4)}</td>
        <td>${m.recall.toFixed(4)}</td>
        <td>${m.precision.toFixed(4)}</td>
        <td>${m.f1.toFixed(4)}</td>
      </tr>
    `).join('');
}

// ============================================================
// FEATURE IMPORTANCE
// ============================================================
function renderFeatureImportance() {
  const container = document.getElementById('feature-importance-list');
  if (!container) return;

  const maxPct = Math.max(...FEATURE_IMPORTANCE.map(f => f.pct));

  container.innerHTML = FEATURE_IMPORTANCE.map((f, i) => {
    const barWidth = (f.pct / maxPct) * 100;
    const hue = 220 - (i / FEATURE_IMPORTANCE.length) * 80;
    const color = `hsl(${hue}, 80%, 60%)`;
    return `
      <div class="importance-item">
        <span class="feature-rank">#${i + 1}</span>
        <span class="feature-name">${f.name}</span>
        <div class="importance-bar-wrap">
          <div class="importance-bar-fill"
               style="--delay: ${i * 0.05}s; width: 0%; background: linear-gradient(90deg, ${color}, ${color}88);"
               data-width="${barWidth}"></div>
        </div>
        <span class="importance-pct">${f.pct}%</span>
      </div>
    `;
  }).join('');

  // Animate bars with IntersectionObserver
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll('.importance-bar-fill').forEach(bar => {
          bar.style.width = bar.dataset.width + '%';
        });
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });
  observer.observe(container);
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initSliders();
  initChurnsDistChart();
  initContractChurnChart();
  initTenureChurnChart();
  initChargesChurnChart();
  renderModelTable();
  renderFeatureImportance();

  // Initial blank gauge
  const gaugeFill = document.getElementById('gauge-fill');
  if (gaugeFill) {
    gaugeFill.style.strokeDashoffset = 310;
    gaugeFill.style.stroke = '#3d85f5';
  }

  // Predict button
  document.getElementById('predict-btn').addEventListener('click', handlePredict);
});
