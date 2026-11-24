const form = document.getElementById('scan-form');
const input = document.getElementById('url-input');
const btn = document.getElementById('scan-btn');
const btnLabel = btn.querySelector('.btn-label');

const resultPanel = document.getElementById('result');
const verdictBanner = document.getElementById('verdict-banner');
const verdictIcon = document.getElementById('verdict-icon');
const verdictLabel = document.getElementById('verdict-label');
const verdictSub = document.getElementById('verdict-sub');
const verdictUrl = document.getElementById('verdict-url');
const scoreValue = document.getElementById('score-value');
const scoreFill = document.getElementById('score-fill');
const signalsGrid = document.getElementById('signals-grid');
const errorBox = document.getElementById('error');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const url = input.value.trim();
  if (!url) return;

  resultPanel.classList.add('hidden');
  errorBox.classList.add('hidden');
  btn.disabled = true;
  btnLabel.textContent = 'Scanning…';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Something went wrong.');
    }

    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btnLabel.textContent = 'Check URL';
  }
});

function showResult(data) {
  const isLegit = data.result === 'LEGITIMATE';
  const cls = isLegit ? 'safe' : 'danger';

  // Verdict banner
  verdictBanner.className = `verdict-banner ${cls}`;
  verdictIcon.textContent = isLegit ? '✓' : '✕';
  verdictLabel.textContent = isLegit ? 'No red flags detected' : 'Phishing patterns detected';
  verdictSub.textContent = isLegit
    ? 'Structural analysis found no significant risk signals.'
    : 'Structural analysis matches known phishing patterns.';

  // Trust score card
  const scoreCard = document.querySelector('.score-card');
  scoreCard.className = `score-card ${cls}`;
  scoreValue.textContent = data.trust_score;
  scoreFill.style.width = `${data.trust_score}%`;

  // Detection card
  verdictUrl.textContent = data.url;

  // Heuristic signals
  signalsGrid.innerHTML = '';
  data.signals.forEach((signal) => {
    const item = document.createElement('div');
    item.className = `signal-item ${signal.risky ? 'risky' : ''}`;
    item.innerHTML = `
      <span class="signal-label">${signal.label}</span>
      <span class="signal-value">${signal.value}</span>
    `;
    signalsGrid.appendChild(item);
  });

  resultPanel.classList.remove('hidden');
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
}