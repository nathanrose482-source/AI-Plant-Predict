/* ── Plant Vision — vanilla JS client ────────────────────────────────────── */
"use strict";

const API = "";   // same origin

// ── Utility ─────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function showToast(msg, duration = 3500) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  t.classList.remove("hidden");
  setTimeout(() => { t.classList.remove("show"); }, duration);
}

function setLoading(btn, isLoading, label) {
  if (isLoading) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>${label}`;
  } else {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── Health polling ──────────────────────────────────────────────────────────

async function checkHealth() {
  const badge = $("healthBadge");
  const dot   = badge.querySelector(".badge-dot");
  const text  = badge.querySelector(".badge-text");
  try {
    const r = await fetch(`${API}/api/v1/health`);
    const d = await r.json();
    const s = d.status;
    dot.className = "badge-dot " + (s === "healthy" ? "ok" : s === "degraded" ? "warn" : "err");
    text.textContent = s.charAt(0).toUpperCase() + s.slice(1);
  } catch {
    dot.className = "badge-dot err";
    text.textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ── Image upload helpers ─────────────────────────────────────────────────────

function bindUploadZone(zoneId, inputId, previewId, btnId) {
  const zone    = $(zoneId);
  const input   = $(inputId);
  const preview = $(previewId);
  const btn     = $(btnId);

  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showToast("Please select a valid image file.");
      return;
    }
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove("hidden");
    btn.disabled = false;
    btn._file = file;
  }

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => loadFile(input.files[0]));

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    loadFile(e.dataTransfer.files[0]);
  });
}

bindUploadZone("classifyDropZone", "classifyInput", "classifyPreview", "classifyBtn");
bindUploadZone("explainDropZone",  "explainInput",  "explainPreview",  "explainBtn");

// ── Threshold slider ─────────────────────────────────────────────────────────

const slider = $("thresholdSlider");
const thVal  = $("thresholdValue");
slider.addEventListener("input", () => { thVal.textContent = parseFloat(slider.value).toFixed(2); });

// ── Classify ─────────────────────────────────────────────────────────────────

$("classifyBtn").addEventListener("click", async () => {
  const btn  = $("classifyBtn");
  const file = btn._file;
  if (!file) return;

  setLoading(btn, true, "Classifying…");
  $("classifyResult").classList.add("hidden");

  const fd = new FormData();
  fd.append("image", file);
  fd.append("threshold", slider.value);
  fd.append("force_fallback", $("forceFallback").checked ? "true" : "false");

  try {
    const r = await fetch(`${API}/api/v1/classify`, { method: "POST", body: fd });
    const d = await r.json();

    if (!r.ok) {
      showToast(`Error ${r.status}: ${d.detail ?? "Unknown error"}`);
      return;
    }

    renderClassifyResult(d);
    } catch (err) {
    showToast(`Classification failed: ${err.message}`);
    console.error("Classification error:", err);
  } finally {
    setLoading(btn, false, "Classify");
  }
});

function renderClassifyResult(d) {
  const box  = $("classifyResult");
  const tier = d.prediction_tier === "tier1_local" ? "tier1" : "tier2";
  const tierLabel = d.prediction_tier === "tier1_local" ? "Tier-1 Local ONNX" : "Tier-2 Gemini VLM";
  const confPct = (d.raw_confidence * 100).toFixed(1);
  const card = d.knowledge_card || {};

  const rows = [
    ["Scientific name",      card.scientific_name],
    ["Family",               card.family],
    ["Edibility",            card.edibility],
    ["Cultivation",          card.cultivation],
    ["Nutritional highlights", card.nutritional_highlights],
    ["Traditional uses",     Array.isArray(card.traditional_uses) ? card.traditional_uses.join(", ") : card.traditional_uses],
    ["Caution",              card.caution],
    ["FGVC notes",           card.fgvc_notes],
    ["Description",          card.description],
  ].filter(([, v]) => v);

  const tableRows = rows.map(([k, v]) =>
    `<tr><th>${escHtml(k)}</th><td>${escHtml(v)}</td></tr>`
  ).join("");

  box.innerHTML = `
    <span class="tier-badge ${tier}">${escHtml(tierLabel)}</span>
    <div style="font-size:1.4rem;font-weight:700;margin-bottom:.25rem">${escHtml(card.common_name || d.predicted_class)}</div>
    <div style="color:var(--slate-600);font-size:.9rem;margin-bottom:.5rem">
      Confidence: <strong>${confPct}%</strong>
      &nbsp;·&nbsp; Latency: <strong>${d.elapsed_latency_ms.toFixed(1)} ms</strong>
      &nbsp;·&nbsp; Threshold: <strong>${(d.threshold_used * 100).toFixed(0)}%</strong>
    </div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:${confPct}%"></div></div>
    </div>
    ${rows.length ? `<table class="knowledge-table"><thead><tr><th>Property</th><th>Details</th></tr></thead><tbody>${tableRows}</tbody></table>` : ""}
    ${d.is_ood ? `<p style="margin-top:.75rem;color:var(--amber-500);font-size:.85rem">⚠️ Out-of-distribution image detected — results may be unreliable.</p>` : ""}
  `;
  box.classList.remove("hidden");
}

// ── Explain ──────────────────────────────────────────────────────────────────

$("explainBtn").addEventListener("click", async () => {
  const btn  = $("explainBtn");
  const file = btn._file;
  if (!file) return;

  setLoading(btn, true, "Generating…");
  $("explainResult").classList.add("hidden");

  const fd = new FormData();
  fd.append("image", file);

  try {
    const r = await fetch(`${API}/api/v1/explain`, { method: "POST", body: fd });
    const d = await r.json();

    if (!r.ok) {
      showToast(`Error ${r.status}: ${d.detail ?? "Unknown error"}`);
      return;
    }

    $("heatmapImg").src = `data:${d.mime_type};base64,${d.heatmap_base64}`;
    $("explainMeta").textContent =
      `Predicted: ${d.predicted_class} · Confidence: ${(d.raw_confidence * 100).toFixed(1)}%`;
    $("explainResult").classList.remove("hidden");
    } catch (err) {
    showToast(`Explanation failed: ${err.message}`);
    console.error("Explanation error:", err);
  } finally {
    setLoading(btn, false, "Generate Heatmap");
  }
});
