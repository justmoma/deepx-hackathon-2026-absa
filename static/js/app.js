// ═══ Arabic ABSA — Frontend Logic ═══

const EXAMPLES = [
    "الأكل كان ممتاز بس الخدمة بطيئة والموظفين مو متعاونين",
    "التوصيل اتأخر ساعة كاملة والأكل وصل بارد",
    "التطبيق معلق كل ما أطلب يعطيني خطأ والخدمة ما ترد",
    "الاسعار غالية جداً بس الأكل لذيذ والخدمة ممتازة"
];

const ASPECT_ICONS = {
    food: "🍽️", service: "🤝", price: "💰", cleanliness: "✨",
    delivery: "🚚", ambiance: "🎶", app_experience: "📱", general: "📋", none: "—"
};

const ASPECT_LABELS_AR = {
    food: "الطعام", service: "الخدمة", price: "السعر", cleanliness: "النظافة",
    delivery: "التوصيل", ambiance: "الأجواء", app_experience: "التطبيق", general: "عام", none: "لا يوجد"
};

// ─── Init ────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadStats();
    setupNavScroll();
    animateOnScroll();
});

// ─── Health Check ────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const dot = document.querySelector(".status-dot");
        const text = document.querySelector(".status-text");
        if (data.mode === "live") {
            dot.className = "status-dot live";
            text.textContent = "Live Model";
        } else {
            dot.className = "status-dot demo";
            text.textContent = "Demo Mode";
        }
    } catch (e) {
        console.error("Health check failed:", e);
    }
}

// ─── Analyze ─────────────────────────────────────────────────────────
async function analyzeReview() {
    const text = document.getElementById("reviewInput").value.trim();
    if (!text) return;

    const btn = document.getElementById("analyzeBtn");
    const loading = document.getElementById("loadingSkeleton");
    const empty = document.getElementById("emptyState");
    const results = document.getElementById("resultsContainer");

    btn.classList.add("loading");
    btn.innerHTML = '<span class="btn-icon">⏳</span><span>Analyzing...</span>';
    empty.style.display = "none";
    results.style.display = "none";
    loading.style.display = "flex";

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        renderResults(data);
    } catch (e) {
        console.error("Analysis failed:", e);
        document.getElementById("emptyState").innerHTML =
            '<div class="empty-icon">⚠️</div><p>Analysis failed. Please check if the server is running.</p>';
        document.getElementById("emptyState").style.display = "flex";
    } finally {
        loading.style.display = "none";
        btn.classList.remove("loading");
        btn.innerHTML = '<span class="btn-icon">⚡</span><span>Analyze</span>';
    }
}

function renderResults(data) {
    const container = document.getElementById("resultsContainer");
    const cards = document.getElementById("aspectCards");
    const badge = document.getElementById("modeBadge");

    // Mode badge
    badge.style.display = "inline-block";
    badge.className = `mode-badge ${data.mode}`;
    badge.textContent = data.mode === "live" ? "🟢 Live" : "🟡 Demo";

    // Aspect cards with confidence
    cards.innerHTML = "";
    data.aspects.forEach((aspect, idx) => {
        const sentiment = data.aspect_sentiments[aspect];
        const icon = ASPECT_ICONS[aspect] || "📋";
        const arLabel = ASPECT_LABELS_AR[aspect] || aspect;
        const confidence = data.confidences?.[aspect] || null;

        const card = document.createElement("div");
        card.className = "aspect-card";
        card.style.animationDelay = `${idx * 0.08}s`;

        const confBar = confidence ? `
            <div class="confidence-wrap">
                <div class="confidence-bar">
                    <div class="confidence-fill confidence-${sentiment}" style="width:${confidence}%"></div>
                </div>
                <span class="confidence-label">${confidence}%</span>
            </div>` : '';

        card.innerHTML = `
            <div class="aspect-emoji">${icon}</div>
            <div class="aspect-info">
                <div class="aspect-name">${capitalize(aspect.replace("_", " "))}</div>
                <div class="aspect-name-ar">${arLabel}</div>
                ${confBar}
            </div>
            <span class="sentiment-badge ${sentiment}">${sentiment}</span>
        `;
        cards.appendChild(card);
    });

    // JSON output
    document.getElementById("jsonOutput").textContent = JSON.stringify({
        aspects: data.aspects,
        aspect_sentiments: data.aspect_sentiments,
        confidences: data.confidences || {}
    }, null, 2);

    container.style.display = "block";
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function toggleJSON() {
    const el = document.getElementById("jsonOutput");
    const btn = document.getElementById("jsonToggleText");
    if (el.style.display === "none") {
        el.style.display = "block";
        btn.textContent = "Hide JSON Output";
    } else {
        el.style.display = "none";
        btn.textContent = "Show JSON Output";
    }
}

function clearInput() {
    document.getElementById("reviewInput").value = "";
    document.getElementById("resultsContainer").style.display = "none";
    document.getElementById("emptyState").style.display = "flex";
    document.getElementById("emptyState").innerHTML =
        '<div class="empty-icon">🔍</div><p>Enter an Arabic review and click <strong>Analyze</strong> to see the results.</p>';
    document.getElementById("modeBadge").style.display = "none";
}

function loadExample(idx) {
    document.getElementById("reviewInput").value = EXAMPLES[idx];
}

// ─── Stats / Charts ──────────────────────────────────────────────────
async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();
        renderAspectChart(data.aspect_counts);
        renderSentimentChart(data.sentiment_counts);
        renderBreakdownChart(data.aspect_sentiment_breakdown);
    } catch (e) {
        console.error("Stats load failed:", e);
    }
}

function renderAspectChart(counts) {
    const container = document.getElementById("aspectChart");
    const max = Math.max(...Object.values(counts));
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);

    let html = '<div class="bar-chart">';
    sorted.forEach(([aspect, count]) => {
        const pct = (count / max) * 100;
        html += `
            <div class="bar-row">
                <span class="bar-label">${ASPECT_ICONS[aspect] || ""} ${capitalize(aspect.replace("_"," "))}</span>
                <div class="bar-track">
                    <div class="bar-fill aspect-bar" style="width:${pct}%">
                        <span class="bar-value">${count}</span>
                    </div>
                </div>
            </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderSentimentChart(counts) {
    const container = document.getElementById("sentimentChart");
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const pos = counts.positive || 0;
    const neg = counts.negative || 0;
    const neu = counts.neutral || 0;

    const circumference = 2 * Math.PI * 60;
    const posLen = (pos / total) * circumference;
    const negLen = (neg / total) * circumference;
    const neuLen = (neu / total) * circumference;

    container.innerHTML = `
        <div class="donut-chart">
            <div class="donut-visual">
                <svg viewBox="0 0 160 160">
                    <circle cx="80" cy="80" r="60" stroke="var(--positive)" 
                        stroke-dasharray="${posLen} ${circumference}" stroke-dashoffset="0"/>
                    <circle cx="80" cy="80" r="60" stroke="var(--negative)" 
                        stroke-dasharray="${negLen} ${circumference}" stroke-dashoffset="${-posLen}"/>
                    <circle cx="80" cy="80" r="60" stroke="var(--neutral)" 
                        stroke-dasharray="${neuLen} ${circumference}" stroke-dashoffset="${-(posLen + negLen)}"/>
                </svg>
                <div class="donut-center">
                    <span class="donut-total">${total}</span>
                    <span class="donut-label">Total</span>
                </div>
            </div>
            <div class="donut-legend">
                <div class="legend-item"><span class="legend-dot pos"></span> Positive: ${pos} (${((pos/total)*100).toFixed(1)}%)</div>
                <div class="legend-item"><span class="legend-dot neg"></span> Negative: ${neg} (${((neg/total)*100).toFixed(1)}%)</div>
                <div class="legend-item"><span class="legend-dot neu"></span> Neutral: ${neu} (${((neu/total)*100).toFixed(1)}%)</div>
            </div>
        </div>`;
}

function renderBreakdownChart(breakdown) {
    const container = document.getElementById("breakdownChart");
    const entries = Object.entries(breakdown).sort((a, b) => {
        const totalA = a[1].positive + a[1].negative + a[1].neutral;
        const totalB = b[1].positive + b[1].negative + b[1].neutral;
        return totalB - totalA;
    });

    let maxTotal = 0;
    entries.forEach(([, v]) => {
        const t = v.positive + v.negative + v.neutral;
        if (t > maxTotal) maxTotal = t;
    });

    let html = '<div class="stacked-chart">';
    entries.forEach(([aspect, sentiments]) => {
        const total = sentiments.positive + sentiments.negative + sentiments.neutral;
        const pctFull = (total / maxTotal) * 100;
        const posW = total ? (sentiments.positive / total) * 100 : 0;
        const negW = total ? (sentiments.negative / total) * 100 : 0;
        const neuW = total ? (sentiments.neutral / total) * 100 : 0;

        html += `
            <div class="stacked-row">
                <span class="stacked-label">${ASPECT_ICONS[aspect] || ""} ${capitalize(aspect.replace("_"," "))}</span>
                <div class="stacked-track" style="width:${pctFull}%">
                    ${posW > 0 ? `<div class="stacked-segment seg-positive" style="width:${posW}%">${sentiments.positive}</div>` : ''}
                    ${negW > 0 ? `<div class="stacked-segment seg-negative" style="width:${negW}%">${sentiments.negative}</div>` : ''}
                    ${neuW > 0 ? `<div class="stacked-segment seg-neutral" style="width:${neuW}%">${sentiments.neutral}</div>` : ''}
                </div>
            </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

// ─── Nav Scroll ──────────────────────────────────────────────────────
function setupNavScroll() {
    const links = document.querySelectorAll(".nav-link");
    const sections = document.querySelectorAll("section[id]");

    window.addEventListener("scroll", () => {
        let current = "";
        sections.forEach(section => {
            if (window.scrollY >= section.offsetTop - 200) {
                current = section.getAttribute("id");
            }
        });
        links.forEach(link => {
            link.classList.remove("active");
            if (link.getAttribute("href") === "#" + current) {
                link.classList.add("active");
            }
        });
    });
}

// ─── Scroll Animations ──────────────────────────────────────────────
function animateOnScroll() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                }
            });
        },
        { threshold: 0.1 }
    );

    document.querySelectorAll(".metric-card, .arch-step, .taxonomy-card, .insight-card").forEach(el => {
        el.style.opacity = "0";
        el.style.transform = "translateY(20px)";
        el.style.transition = "opacity 0.6s ease-out, transform 0.6s ease-out";
        observer.observe(el);
    });
}

// Add CSS for visible class
const style = document.createElement("style");
style.textContent = ".visible { opacity: 1 !important; transform: translateY(0) !important; }";
document.head.appendChild(style);

// Enter key to analyze
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
        analyzeReview();
    }
});
