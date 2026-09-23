const MODEL = "lstm";

const RANDOM_KEYWORDS = [
    "নদী", "বৃষ্টি", "ভালোবাসা", "আকাশ", "হৃদয়", "একা",
    "মেঘ", "বসন্ত", "স্মৃতি", "সাগর", "পাখি", "ফুল",
    "প্রকৃতি", "জীবন", "মৃত্যু", "স্বপ্ন", "রাত", "ভোর"
];
const CHIP_KEYWORDS = RANDOM_KEYWORDS.slice(0, 10);
const SKELETON_WIDTHS = [88, 72, 80, 64, 76, 68];
const NARROW_SCREEN = "(max-width: 860px)";

const $ = (id) => document.getElementById(id);

const el = {
    keyword: $("keyword-input"),
    chips: $("chips"),
    random: $("random-btn"),
    tuning: $("tuning"),
    lines: $("lines-input"),
    linesValue: $("lines-value"),
    temp: $("temp-input"),
    tempValue: $("temp-value"),
    topk: $("topk-input"),
    topkValue: $("topk-value"),
    rhymeScheme: $("rhymeScheme"),
    generate: $("generate-btn"),

    sheet: $("sheet"),
    empty: $("empty-state"),
    loading: $("loading"),
    skeleton: $("skeleton"),
    error: $("error-box"),
    poem: $("poem-container"),
    title: $("poem-title"),
    body: $("poem-body"),

    words: $("stat-words"),
    ttr: $("stat-ttr"),
    ppl: $("stat-ppl"),
    time: $("stat-time"),

    copy: $("copy-btn"),
    download: $("download-btn"),
    regenerate: $("regenerate-btn"),
};

let busy = false;
let lastPayload = null;
let currentPoem = "";

function setState(name) {
    const views = { empty: el.empty, loading: el.loading, error: el.error, poem: el.poem };
    Object.entries(views).forEach(([key, node]) => node.classList.toggle("hidden", key !== name));
}

function bindSlider(input, output, format = (v) => v) {
    const sync = () => {
        const min = Number(input.min);
        const max = Number(input.max);
        input.style.setProperty("--fill", ((input.value - min) / (max - min)) * 100 + "%");
        output.textContent = format(input.value);
    };
    input.addEventListener("input", sync);
    sync();
}

function buildChips() {
    CHIP_KEYWORDS.forEach((word) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = word;
        chip.dataset.word = word;
        chip.setAttribute("aria-pressed", "false");
        chip.addEventListener("click", () => {
            el.keyword.value = word;
            syncChips();
            generatePoem();
        });
        el.chips.insertBefore(chip, el.random);
    });
}

function syncChips() {
    const current = el.keyword.value.trim();
    el.chips.querySelectorAll(".chip[data-word]").forEach((chip) => {
        chip.setAttribute("aria-pressed", String(chip.dataset.word === current));
    });
}

function pickRandomKeyword() {
    const current = el.keyword.value.trim();
    const pool = RANDOM_KEYWORDS.filter((word) => word !== current);
    return pool[Math.floor(Math.random() * pool.length)];
}

function buildSkeleton(lineCount) {
    el.skeleton.replaceChildren();
    for (let i = 0; i <= lineCount; i++) {
        const bar = document.createElement("span");
        bar.style.width = i === 0 ? "45%" : SKELETON_WIDTHS[(i - 1) % SKELETON_WIDTHS.length] + "%";
        el.skeleton.appendChild(bar);
    }
}

function formatNumber(value, digits) {
    if (value === undefined || value === null || value === "") return "—";
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : String(value);
}

function showError(heading, message) {
    el.error.replaceChildren();
    const h = document.createElement("h2");
    h.textContent = heading;
    const p = document.createElement("p");
    p.textContent = message;
    el.error.append(h, p);
    setState("error");
}

async function generatePoem(payload) {
    if (busy) return;

    if (!payload) {
        const keyword = el.keyword.value.trim();
        if (!keyword) {
            showError("কোনো শব্দ দেওয়া হয়নি", "কবিতার বিষয় হিসেবে একটি শব্দ লিখুন, তারপর আবার চেষ্টা করুন।");
            el.keyword.focus();
            return;
        }
        payload = {
            model: MODEL,
            keyword: keyword,
            num_lines: parseInt(el.lines.value, 10),
            temperature: parseFloat(el.temp.value),
            top_k: parseInt(el.topk.value, 10),
            top_p: 0.90,
            rhyme_scheme: el.rhymeScheme ? el.rhymeScheme.value : "auto",
        };
    }

    busy = true;
    el.generate.disabled = true;
    el.generate.textContent = "লেখা হচ্ছে…";
    buildSkeleton(payload.num_lines);
    setState("loading");

    // On phones the controls sit above the sheet, so bring the result into view.
    if (window.matchMedia(NARROW_SCREEN).matches) {
        el.sheet.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    try {
        const resp = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json().catch(() => ({}));

        if (!resp.ok || data.error) {
            showError("কবিতা তৈরি করা যায়নি", data.error || "সার্ভার থেকে সঠিক উত্তর আসেনি। একটু পরে আবার চেষ্টা করুন।");
            return;
        }

        lastPayload = payload;
        renderPoem(data, payload);
    } catch (e) {
        showError("সার্ভারে পৌঁছানো যায়নি", "সার্ভার চালু আছে কিনা দেখুন, তারপর আবার চেষ্টা করুন। (" + e.message + ")");
    } finally {
        busy = false;
        el.generate.disabled = false;
        el.generate.textContent = "কবিতা লিখুন";
    }
}

function renderPoem(data, payload) {
    const lines = data.lines && data.lines.length
        ? data.lines
        : (data.poem_text || "").split("\n").filter(Boolean);

    if (!lines.length) {
        showError("কোনো লাইন তৈরি হয়নি", "অন্য একটি শব্দ দিন অথবা Temperature একটু বাড়িয়ে আবার চেষ্টা করুন।");
        return;
    }

    el.title.textContent = data.keyword || payload.keyword;

    el.body.replaceChildren();
    lines.forEach((line, i) => {
        const p = document.createElement("p");
        p.className = "poem-line";
        p.style.setProperty("--i", i);
        p.textContent = line;
        el.body.appendChild(p);
    });

    const m = data.metrics || {};
    el.words.textContent = formatNumber(m.total_words, 0);
    el.ttr.textContent = formatNumber(m.type_token_ratio ?? m.ttr, 2);
    el.ppl.textContent = formatNumber(m.average_perplexity, 1);
    el.time.textContent = data.elapsed_seconds !== undefined
        ? formatNumber(data.elapsed_seconds, 2) + " সে"
        : "—";

    currentPoem = data.poem_text || lines.join("\n");
    setState("poem");
}

async function copyPoem() {
    if (!currentPoem) return;
    try {
        await navigator.clipboard.writeText(currentPoem);
    } catch {
        // Clipboard API needs a secure context; fall back for plain-HTTP setups.
        const area = document.createElement("textarea");
        area.value = currentPoem;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
    }
    const label = el.copy.querySelector(".label");
    label.textContent = "কপি হয়েছে";
    setTimeout(() => { label.textContent = "কপি"; }, 1600);
}

function downloadPoem() {
    if (!currentPoem) return;
    const blob = new Blob([currentPoem], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bangla_poem.txt";
    a.click();
    URL.revokeObjectURL(url);
}

function init() {
    bindSlider(el.lines, el.linesValue);
    bindSlider(el.temp, el.tempValue, (v) => parseFloat(v).toFixed(2));
    bindSlider(el.topk, el.topkValue);

    buildChips();
    syncChips();

    if (window.matchMedia(NARROW_SCREEN).matches) el.tuning.open = false;

    el.keyword.addEventListener("input", syncChips);
    el.keyword.addEventListener("keydown", (e) => {
        if (e.key === "Enter") generatePoem();
    });

    el.random.addEventListener("click", () => {
        el.keyword.value = pickRandomKeyword();
        syncChips();
        generatePoem();
    });

    el.generate.addEventListener("click", () => generatePoem());
    el.regenerate.addEventListener("click", () => generatePoem(lastPayload));
    el.copy.addEventListener("click", copyPoem);
    el.download.addEventListener("click", downloadPoem);

    setState("empty");
}

init();