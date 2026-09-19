/**
 * script.js – Dog Breed Classifier Frontend Logic
 *
 * Responsibilities:
 *  1. Health-check poll on load → update header status badge
 *  2. Drag-and-drop + click-to-browse image upload with preview
 *  3. Form submission → POST /predict (multipart/form-data)
 *  4. Render top-3 breed cards with animated confidence bars
 *  5. Graceful error display and "try again" flow
 */

'use strict';

/* ─────────────────────────────────────────────────────────────
   DOM References
───────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const dropIdle       = document.getElementById('dropIdle');
const dropPreview    = document.getElementById('dropPreview');
const previewImg     = document.getElementById('previewImg');
const previewLabel   = document.getElementById('previewLabel');
const clearBtn       = document.getElementById('clearBtn');
const uploadForm     = document.getElementById('uploadForm');
const classifyBtn    = document.getElementById('classifyBtn');
const errorMsg       = document.getElementById('errorMsg');
const loadingCard    = document.getElementById('loadingCard');
const resultsSection = document.getElementById('resultsSection');
const topBreedName   = document.getElementById('topBreedName');
const topBreedConf   = document.getElementById('topBreedConf');
const breedCards     = document.getElementById('breedCards');
const tryAgainBtn    = document.getElementById('tryAgainBtn');
const statusDot      = document.getElementById('statusDot');
const statusLabel    = document.getElementById('statusLabel');

/* ─────────────────────────────────────────────────────────────
   State
───────────────────────────────────────────────────────────── */
let selectedFile = null;

/* ─────────────────────────────────────────────────────────────
   Health Check
───────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    const data = await res.json();

    if (data.model_loaded) {
      statusDot.className  = 'status-dot online';
      statusLabel.textContent = 'Model ready';
    } else {
      statusDot.className  = 'status-dot offline';
      statusLabel.textContent = 'Model not loaded';
    }
  } catch {
    statusDot.className  = 'status-dot offline';
    statusLabel.textContent = 'API offline';
  }
}

/* Poll every 30 seconds */
checkHealth();
setInterval(checkHealth, 30_000);

/* ─────────────────────────────────────────────────────────────
   File Handling Helpers
───────────────────────────────────────────────────────────── */
const ALLOWED_TYPES = new Set([
  'image/jpeg', 'image/jpg', 'image/png',
  'image/webp', 'image/gif', 'image/bmp', 'image/tiff',
]);

/**
 * Validate and stage a File object for upload.
 * @param {File} file
 */
function stageFile(file) {
  if (!file) return;

  // Type check
  if (!ALLOWED_TYPES.has(file.type)) {
    showError(`"${file.name}" is not a supported image type. Please use JPEG, PNG, or WebP.`);
    return;
  }

  // Size limit: 20 MB
  const MAX_MB = 20;
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max size: ${MAX_MB} MB.`);
    return;
  }

  selectedFile = file;
  hideError();
  showPreview(file);
  enableClassify();
}

/** Render image preview in the drop zone. */
function showPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url); // free memory
  previewLabel.textContent = file.name;
  dropIdle.hidden    = true;
  dropPreview.hidden = false;
}

/** Reset drop zone to idle state. */
function clearPreview() {
  selectedFile = null;
  fileInput.value  = '';
  previewImg.src   = '';
  dropIdle.hidden    = false;
  dropPreview.hidden = true;
  disableClassify();
  hideError();
}

function enableClassify() {
  classifyBtn.disabled = false;
  classifyBtn.setAttribute('aria-disabled', 'false');
}

function disableClassify() {
  classifyBtn.disabled = true;
  classifyBtn.setAttribute('aria-disabled', 'true');
}

/* ─────────────────────────────────────────────────────────────
   Error Display
───────────────────────────────────────────────────────────── */
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.textContent = '';
  errorMsg.hidden = true;
}

/* ─────────────────────────────────────────────────────────────
   UI State Switches
───────────────────────────────────────────────────────────── */
function showUploadUI() {
  loadingCard.hidden    = true;
  resultsSection.hidden = true;
  document.querySelector('.upload-card').hidden = false;
}

function showLoadingUI() {
  document.querySelector('.upload-card').hidden = false; // keep visible
  loadingCard.hidden    = false;
  resultsSection.hidden = true;
  classifyBtn.disabled  = true;
  classifyBtn.querySelector('.btn-text').textContent = 'Classifying…';
}

function showResultsUI() {
  loadingCard.hidden    = true;
  resultsSection.hidden = false;
  document.querySelector('.upload-card').hidden = true;
}

/* ─────────────────────────────────────────────────────────────
   Render Breed Cards
───────────────────────────────────────────────────────────── */
/**
 * @param {{ rank: number, breed: string, confidence: float, confidence_pct: float }[]} predictions
 */
function renderResults(predictions) {
  // Update headline
  const top = predictions[0];
  topBreedName.textContent = top.breed;
  topBreedConf.textContent =
    `${top.confidence_pct.toFixed(1)}% confidence · Top match from 120 breeds`;

  const topBreedInfoBox = document.getElementById('topBreedInfoBox');
  if (top.info && typeof top.info === 'object' && Object.keys(top.info).length > 0) {
    topBreedInfoBox.innerHTML = `
      <div class="breed-specs-grid">
        <div class="spec-item"><span class="spec-label">🌍 Origin:</span> ${top.info.origin || 'Unknown'}</div>
        <div class="spec-item"><span class="spec-label">🏷️ Group:</span> ${top.info.group || 'N/A'}</div>
        <div class="spec-item"><span class="spec-label">❤️ Temperament:</span> ${top.info.temperament || 'N/A'}</div>
        <div class="spec-item"><span class="spec-label">🧬 Physical Traits:</span> ${top.info.physical_traits || 'N/A'}</div>
        <div class="spec-item"><span class="spec-label">✨ Specialty:</span> ${top.info.specialty || 'N/A'}</div>
        <div class="spec-item"><span class="spec-label">⚠️ Care Notes:</span> ${top.info.care_notes || 'N/A'}</div>
      </div>
    `;
    topBreedInfoBox.hidden = false;
  } else if (top.info && typeof top.info === 'string') {
    topBreedInfoBox.textContent = top.info;
    topBreedInfoBox.hidden = false;
  } else {
    topBreedInfoBox.hidden = true;
  }

  // Clear previous cards
  breedCards.innerHTML = '';

  predictions.forEach((pred) => {
    const card = document.createElement('div');
    card.className = `breed-card rank-${pred.rank}`;
    card.setAttribute('role', 'listitem');

    card.innerHTML = `
      <div class="rank-badge rank-${pred.rank}" aria-label="Rank ${pred.rank}">
        ${pred.rank === 1 ? '🥇' : pred.rank === 2 ? '🥈' : '🥉'}
      </div>
      <div class="breed-info">
        <p class="breed-name" title="${pred.breed}">${pred.breed}</p>
        <div class="confidence-bar-wrapper" role="progressbar"
             aria-label="${pred.breed} confidence"
             aria-valuenow="${pred.confidence_pct.toFixed(1)}"
             aria-valuemin="0" aria-valuemax="100">
          <div class="confidence-bar" data-width="${pred.confidence_pct.toFixed(2)}"></div>
        </div>
      </div>
      <span class="confidence-pct">${pred.confidence_pct.toFixed(1)}%</span>
    `;

    breedCards.appendChild(card);
  });

  // Animate bars after paint (requestAnimationFrame for CSS transition)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.confidence-bar').forEach((bar) => {
        bar.style.width = `${bar.dataset.width}%`;
      });
    });
  });
}

/* ─────────────────────────────────────────────────────────────
   API Call
───────────────────────────────────────────────────────────── */
async function classifyImage() {
  if (!selectedFile) return;

  showLoadingUI();
  hideError();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60_000), // 60 s timeout
    });

    const data = await response.json();

    if (!response.ok) {
      // Server returned 4xx / 5xx
      const detail = data.detail || `Server error (${response.status})`;
      if (response.status === 503) {
        throw new Error(
          '⚠️ Model not ready. ' + detail +
          '\n\nOpen EXPORT_MODEL_FROM_COLAB.md for step-by-step instructions.'
        );
      }
      throw new Error(detail);
    }

    if (!data.success || !Array.isArray(data.predictions) || data.predictions.length === 0) {
      throw new Error('Invalid response from server. Please try again.');
    }

    renderResults(data.predictions);
    showResultsUI();

  } catch (err) {
    // Restore upload UI on error
    classifyBtn.disabled = false;
    classifyBtn.querySelector('.btn-text').textContent = 'Classify Breed';
    loadingCard.hidden = true;

    const message = err.name === 'TimeoutError'
      ? 'Request timed out. The model may still be loading — please try again.'
      : err.message || 'An unexpected error occurred.';

    showError(message);
  }
}

/* ─────────────────────────────────────────────────────────────
   Event Listeners – Drop Zone
───────────────────────────────────────────────────────────── */

// Click anywhere in the drop zone triggers the hidden file input
dropZone.addEventListener('click', (e) => {
  if (e.target === clearBtn || clearBtn.contains(e.target)) return;
  fileInput.click();
});

// Keyboard access for the drop zone
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

// File input change
fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    stageFile(fileInput.files[0]);
  }
});

// Drag events
['dragenter', 'dragover'].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
  });
});

['dragleave', 'dragend', 'drop'].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
  });
});

dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) stageFile(file);
});

// Clear / remove image
clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearPreview();
});

/* ─────────────────────────────────────────────────────────────
   Event Listeners – Form
───────────────────────────────────────────────────────────── */
uploadForm.addEventListener('submit', (e) => {
  e.preventDefault();
  classifyImage();
});

/* ─────────────────────────────────────────────────────────────
   Event Listeners – Results
───────────────────────────────────────────────────────────── */
tryAgainBtn.addEventListener('click', () => {
  clearPreview();
  breedCards.innerHTML = '';
  classifyBtn.querySelector('.btn-text').textContent = 'Classify Breed';
  showUploadUI();
});

/* ─────────────────────────────────────────────────────────────
   Global paste support (Ctrl+V to paste an image)
───────────────────────────────────────────────────────────── */
document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) stageFile(file);
      break;
    }
  }
});
