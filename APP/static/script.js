/**
 * NexusLend AI – Frontend Controller
 * Handles form serialisation → API call → animated result rendering
 */

const form = document.getElementById("loanForm");
const btn = document.getElementById("analyzeBtn");
const resultSection = document.getElementById("resultSection");

// ─── FORM SUBMIT ────────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Loading state
  btn.classList.add("loading");
  btn.querySelector(".btn-text").textContent = "Analysing…";
  resultSection.classList.add("hidden");

  // Collect all form fields
  const formData = new FormData(form);
  const payload = {};
  formData.forEach((val, key) => {
    payload[key] = val;
  });

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      alert("Error: " + (data.error || "Unknown server error"));
      return;
    }

    renderResults(data);

    // Smooth scroll to results
    setTimeout(
      () =>
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" }),
      100,
    );
  } catch (err) {
    alert("Network error: " + err.message);
  } finally {
    btn.classList.remove("loading");
    btn.querySelector(".btn-text").textContent = "Run AI Analysis";
  }
});

// ─── RENDER RESULTS ─────────────────────────────────────────────────────────
function renderResults(d) {
  // Risk class helpers
  const riskClass = {
    "Low Risk": "low",
    "Medium Risk": "medium",
    "High Risk": "high",
  };
  const tier = riskClass[d.risk_level] || "medium";

  // ── Verdict banner ──
  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict-banner verdict-" + tier;

  document.getElementById("verdictOutcome").textContent = d.outcome;
  document.getElementById("verdictDecision").textContent = d.decision;

  // Outcome colour
  const outcomeEl = document.getElementById("verdictOutcome");
  outcomeEl.style.color =
    d.outcome === "No Default" ? "var(--green)" : "var(--coral)";

  // ── Risk ring ──
  const prob = d.default_probability; // 0-100
  const ringFill = document.getElementById("ringFill");
  const circumf = 314; // 2πr where r=50
  const offset = circumf - (circumf * prob) / 100;
  const ringColor =
    tier === "low" ? "#34d399" : tier === "medium" ? "#ffc857" : "#ff6b6b";

  ringFill.style.stroke = ringColor;
  ringFill.style.strokeDashoffset = circumf; // start at 0 fill
  setTimeout(() => {
    ringFill.style.strokeDashoffset = offset;
  }, 50);

  document.getElementById("ringProb").textContent = prob + "%";
  document.getElementById("ringProb").style.color = ringColor;

  const badge = document.getElementById("riskBadge");
  badge.textContent = d.risk_level;
  badge.className = "risk-badge risk-" + tier;

  // ── Metric cards ──
  setCard("emiVal", formatCurrency(d.emi));
  setCard("ratioVal", d.emi_ratio + "%");
  setCard("safeVal", formatCurrency(d.safe_loan_amount));
  setCard("probVal", d.default_probability + "%");

  // Ratio card colour
  const ratioEl = document.getElementById("ratioVal");
  if (d.emi_ratio > 50) ratioEl.style.color = "var(--coral)";
  else if (d.emi_ratio > 35) ratioEl.style.color = "var(--amber)";
  else ratioEl.style.color = "var(--green)";

  // ── Reasons ──
  const list = document.getElementById("reasonsList");
  list.innerHTML = "";
  (d.reasons || []).forEach((r, i) => {
    const li = document.createElement("li");
    li.textContent = r;
    li.style.animationDelay = i * 0.08 + "s";
    list.appendChild(li);
  });

  // ── Recommendation ──
  const recoText = buildRecommendation(d);
  document.getElementById("recoText").innerHTML = recoText;

  // ── Mock notice ──
  const mockNotice = document.getElementById("mockNotice");
  d.mock_mode
    ? mockNotice.classList.remove("hidden")
    : mockNotice.classList.add("hidden");

  // ── Show section ──
  resultSection.classList.remove("hidden");
}

// ─── HELPERS ────────────────────────────────────────────────────────────────
function setCard(id, val) {
  const el = document.getElementById(id);
  el.textContent = val;
}

function formatCurrency(val) {
  if (!val && val !== 0) return "—";
  return (
    "₹" + Number(val).toLocaleString("en-IN", { maximumFractionDigits: 0 })
  );
}

function buildRecommendation(d) {
  const safeStr = formatCurrency(d.safe_loan_amount);
  const ratio = d.emi_ratio;

  if (d.risk_level === "Low Risk") {
    return `<strong>✅ Approve Loan.</strong> The applicant demonstrates strong creditworthiness. EMI-to-income ratio of <strong>${ratio}%</strong> is well within safe limits. Proceed with standard documentation.`;
  }
  if (d.risk_level === "High Risk") {
    return `<strong>🚫 Reject Loan.</strong> The risk profile is unfavourable. Consider recommending a maximum loan of <strong>${safeStr}</strong> (based on 35% EMI-to-income threshold) if the applicant reapplies with improved financial standing or additional collateral.`;
  }
  // Medium
  if (ratio > 50) {
    return `<strong>⚠ Conditional Review.</strong> EMI burden of <strong>${ratio}%</strong> is too high. Recommend reducing the loan to approximately <strong>${safeStr}</strong> to keep EMI within 35% of monthly income. Request additional income proof and guarantor.`;
  }
  return `<strong>⚠ Approve with Conditions.</strong> The applicant qualifies but with enhanced monitoring. Safe loan ceiling is <strong>${safeStr}</strong>. Require post-disbursement quarterly reviews and ensure co-applicant documentation is complete.`;
}
