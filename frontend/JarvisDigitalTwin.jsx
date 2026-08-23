import { useState, useEffect, useCallback, useRef } from "react";

// ── Theme definitions ──────────────────────────────────────────
const themes = {
  dark: {
    bg: "#080c14",
    surface: "#0f1319",
    surfaceLight: "#161b25",
    border: "rgba(99,102,241,0.15)",
    borderHover: "rgba(99,102,241,0.35)",
    text: "#e5e7eb",
    textMuted: "#6b7280",
    accent: "#6366f1",
    accentLight: "#818cf8",
    accentBg: "rgba(99,102,241,0.12)",
    green: "#22c55e",
    greenBg: "rgba(34,197,94,0.12)",
    orange: "#f39c12",
    orangeBg: "rgba(243,156,18,0.12)",
    blue: "#60a5fa",
    blueBg: "rgba(96,165,250,0.12)",
    red: "#ef4444",
    redBg: "rgba(239,68,68,0.12)",
    gridColor: "rgba(99,102,241,0.04)",
    nodeInactive: "rgba(255,255,255,0.03)",
    connectionInactive: "rgba(255,255,255,0.06)",
    inputBg: "rgba(99,102,241,0.05)",
    bubbleUser: "#6366f1",
    bubbleTwin: "#1e2130",
    bubbleTwinBorder: "rgba(99,102,241,0.15)",
    statsBar: "rgba(99,102,241,0.08)",
    cardBg: "rgba(15,19,25,0.95)",
  },
  light: {
    bg: "#f5f0e8",
    surface: "#ede8df",
    surfaceLight: "#e8e2d8",
    border: "rgba(99,102,241,0.2)",
    borderHover: "rgba(99,102,241,0.45)",
    text: "#1f2937",
    textMuted: "#6b7280",
    accent: "#6366f1",
    accentLight: "#4f46e5",
    accentBg: "rgba(99,102,241,0.1)",
    green: "#16a34a",
    greenBg: "rgba(22,163,74,0.1)",
    orange: "#d97706",
    orangeBg: "rgba(217,119,6,0.1)",
    blue: "#2563eb",
    blueBg: "rgba(37,99,235,0.1)",
    red: "#dc2626",
    redBg: "rgba(220,38,38,0.1)",
    gridColor: "rgba(99,102,241,0.06)",
    nodeInactive: "rgba(0,0,0,0.03)",
    connectionInactive: "rgba(0,0,0,0.08)",
    inputBg: "rgba(99,102,241,0.06)",
    bubbleUser: "#6366f1",
    bubbleTwin: "#e8e2d8",
    bubbleTwinBorder: "rgba(99,102,241,0.2)",
    statsBar: "rgba(99,102,241,0.06)",
    cardBg: "rgba(245,240,232,0.95)",
  },
};

// ── Pipeline node definitions (wider layout) ──────────────────
const NODES = [
  { id: "intent",     label: "Intent",    sub: "classifier", x: 200, y: 50,  r: 28 },
  { id: "rag",        label: "RAG",       sub: "corpus",     x: 60,  y: 155, r: 22 },
  { id: "identity",   label: "Identity",  sub: "yaml",       x: 340, y: 155, r: 22 },
  { id: "calendar",   label: "Calendar",  sub: "events",     x: 145, y: 175, r: 19 },
  { id: "weather",    label: "Weather",   sub: "API",        x: 255, y: 175, r: 19 },
  { id: "gmail",      label: "Gmail",     sub: "search",     x: 60,  y: 250, r: 18 },
  { id: "news",       label: "News",      sub: "API",        x: 145, y: 265, r: 17 },
  { id: "github",     label: "GitHub",    sub: "API",        x: 340, y: 250, r: 18 },
  { id: "krikri",     label: "Krikri",    sub: "LLM",        x: 200, y: 330, r: 30 },
  { id: "factcheck",  label: "Fact",      sub: "check",      x: 115, y: 420, r: 20 },
  { id: "guardrails", label: "Guard",     sub: "rails",      x: 285, y: 420, r: 20 },
  { id: "output",     label: "Output",    sub: "response",   x: 200, y: 500, r: 25 },
];

const CONNECTIONS = [
  { from: "intent", to: "rag" },
  { from: "intent", to: "identity" },
  { from: "intent", to: "calendar" },
  { from: "intent", to: "gmail" },
  { from: "intent", to: "weather" },
  { from: "intent", to: "news" },
  { from: "intent", to: "github" },
  { from: "rag", to: "krikri" },
  { from: "identity", to: "krikri" },
  { from: "gmail", to: "krikri" },
  { from: "calendar", to: "krikri" },
  { from: "weather", to: "krikri" },
  { from: "news", to: "krikri" },
  { from: "github", to: "krikri" },
  { from: "krikri", to: "factcheck" },
  { from: "krikri", to: "guardrails" },
  { from: "factcheck", to: "output" },
  { from: "guardrails", to: "output" },
];

// ── Node descriptions (shown on click) ───────────────────────
const NODE_INFO = {
  intent: {
    title: "Intent Classifier",
    desc: "Αναλύει το μήνυμά σου και το κατηγοριοποιεί σε 9 intents: personal, knowledge, casual, memory, schedule, weather, news, devops, sensitive. Καθορίζει ποια data sources θα ενεργοποιηθούν.",
    tech: "Python · Keyword Matching · Confidence Scoring",
  },
  rag: {
    title: "RAG Search",
    desc: "Αναζητά στο αρχείο των 13.289 προσωπικών συνομιλιών με υβριδική αναζήτηση: BM25 για ακριβείς λέξεις και πυκνά διανύσματα για παραφράσεις, συντηγμένα με RRF. Βρίσκει τι έχει ήδη ειπωθεί, όχι πώς θα απαντούσε.",
    tech: "ChromaDB · BM25 + dense · RRF k=60",
  },
  identity: {
    title: "Identity Lookup",
    desc: "Φορτώνει το identity profile: ηλικία, σπουδές, εργασία, skills, projects. Εξασφαλίζει ότι το digital twin γνωρίζει ποιος είναι ο Γιώργος.",
    tech: "YAML · Static Profile",
  },
  gmail: {
    title: "Gmail Search",
    desc: "Αναζητά στα πρόσφατα emails μέσω Gmail API. Χρήσιμο για ερωτήσεις τύπου «τι μου έστειλε ο Χ;» ή «πότε έρχεται η παραγγελία;».",
    tech: "Gmail API · n8n OAuth2",
  },
  calendar: {
    title: "Calendar Events",
    desc: "Ελέγχει το Google Calendar για meetings, events και ελεύθερες ώρες. Απαντά σε «πότε είμαι free αύριο;» ή «τι έχω σήμερα;».",
    tech: "Google Calendar API · n8n Node",
  },
  weather: {
    title: "Weather API",
    desc: "Φέρνει τρέχοντα καιρικά δεδομένα — θερμοκρασία, υγρασία, περιγραφή — για την τοποθεσία του Γιώργου.",
    tech: "OpenWeatherMap · REST API",
  },
  news: {
    title: "News Headlines",
    desc: "Φέρνει τα 5 πιο πρόσφατα ελληνικά νέα μέσω NewsAPI. Για ερωτήσεις τύπου «τι γίνεται;» ή «τι νέα;».",
    tech: "NewsAPI · REST API",
  },
  github: {
    title: "GitHub Activity",
    desc: "Ελέγχει πρόσφατα commits, PRs και events στο GitHub profile (giotros). Για «τι δούλεψα σήμερα;» ή «πόσα commits;».",
    tech: "GitHub API · REST · giotros",
  },
  krikri: {
    title: "Krikri LLM",
    desc: "Το fine-tuned language model — Llama-Krikri-8B-Instruct (ΙΕΛ/Αθηνά) με QLoRA adapter εκπαιδευμένο σε 13.289 ζεύγη από προσωπικές συνομιλίες. Το Mistral-7B εξετάστηκε και απορρίφθηκε: έσπαγε τις ελληνικές λέξεις σε πολύ περισσότερα tokens.",
    tech: "Krikri-8B · QLoRA r=64 · 4-bit NF4 · Ray · Colab T4/A100",
  },
  factcheck: {
    title: "Fact Check",
    desc: "Τέσσερις έλεγχοι για επινοήσεις: λίστα γνωστών λαθών, λίστα τεκμηριωμένων εργαλείων, αναπτύγματα ακρωνυμίων, και παραφθαρμένα ονόματα. Αν βρεθεί αντίφαση, η απάντηση ξαναπαράγεται μία φορά — και αν ξαναποτύχει, το σύστημα αρνείται αντί να πει ψέμα.",
    tech: "Denylist + allowlist · retry · refusal"
  },
  guardrails: {
    title: "Guardrails",
    desc: "Post-processing: αφαιρεί PII τρίτων (ονόματα, τηλέφωνα, IBAN), emojis, markup. GDPR compliance — κανένα προσωπικό δεδομένο τρίτου δεν διαρρέει.",
    tech: "Regex · PII Detection · GDPR",
  },
  output: {
    title: "Final Output",
    desc: "Η τελική απάντηση μετά από: intent routing → context injection → LLM generation → fact-checking → guardrails. Ασφαλής και πιστή στο πρόσωπο του Γιώργου.",
    tech: "JSON Response · n8n Webhook",
  },
};

// ── Intent → active nodes mapping ──────────────────────────────
// Εφεδρικό μόνο: χρησιμοποιείται όταν το backend δεν αναφέρει πηγές.
const intentSourceFallback = {
  PERSONAL: "Identity", KNOWLEDGE: "RAG Corpus", CASUAL: "Krikri LLM",
  SENSITIVE: "Human Review", MEMORY: "Gmail", SCHEDULE: "Calendar",
  DEVOPS: "GitHub", WEATHER: "Weather API", NEWS: "News API",
};

const INTENT_NODES = {
  PERSONAL: ["intent", "identity", "krikri", "factcheck", "guardrails", "output"],
  KNOWLEDGE: ["intent", "rag", "krikri", "factcheck", "guardrails", "output"],
  CASUAL: ["intent", "krikri", "guardrails", "output"],
  SENSITIVE: ["intent", "output"],
  MEMORY: ["intent", "gmail", "rag", "krikri", "factcheck", "guardrails", "output"],
  SCHEDULE: ["intent", "calendar", "krikri", "factcheck", "guardrails", "output"],
  DEVOPS: ["intent", "github", "krikri", "factcheck", "guardrails", "output"],
  WEATHER: ["intent", "weather", "krikri", "factcheck", "guardrails", "output"],
  NEWS: ["intent", "news", "rag", "krikri", "factcheck", "guardrails", "output"],
};

// ── Node color mapping ─────────────────────────────────────────
function getNodeColor(nodeId, theme) {
  const t = themes[theme];
  const map = {
    intent: { fill: t.accentBg, stroke: t.accent, text: t.accent, sub: t.accentLight },
    rag: { fill: t.greenBg, stroke: t.green, text: t.green, sub: t.green },
    identity: { fill: t.orangeBg, stroke: t.orange, text: t.orange, sub: t.orange },
    gmail: { fill: t.redBg, stroke: t.red, text: t.red, sub: t.red },
    calendar: { fill: t.blueBg, stroke: t.blue, text: t.blue, sub: t.blue },
    weather: { fill: t.blueBg, stroke: t.blue, text: t.blue, sub: t.blue },
    news: { fill: t.orangeBg, stroke: t.orange, text: t.orange, sub: t.orange },
    github: { fill: t.greenBg, stroke: t.green, text: t.green, sub: t.green },
    krikri: { fill: t.accentBg, stroke: t.accent, text: t.accent, sub: t.accentLight },
    factcheck: { fill: t.greenBg, stroke: t.green, text: t.green, sub: t.green },
    guardrails: { fill: t.orangeBg, stroke: t.orange, text: t.orange, sub: t.orange },
    output: { fill: t.accentBg, stroke: t.accent, text: "#fff", sub: t.accentLight },
  };
  return map[nodeId] || map.intent;
}

// ── Demo conversations ─────────────────────────────────────────
const DEMO_CONVERSATIONS = {
  CASUAL: {
    reply: "Καλά, δουλεύω πάνω στη διπλωματική μου — ένα digital twin project με LLMs. Εσύ;",
    source: "Krikri LLM",
    confidence: 94,
    time: "0.8s",
  },
  PERSONAL: {
    reply: "Γιώργος Τροχίδης, 26 χρονών, από Γιαννιτσά. Cloud engineer στην e-avenue.",
    source: "Identity",
    confidence: 98,
    time: "0.7s",
  },
  MEMORY: {
    reply: "Τα παρήγγειλα στις 8/7 και σύμφωνα με το tracking έρχονται αύριο.",
    source: "Gmail",
    confidence: 92,
    time: "1.2s",
  },
  SCHEDULE: {
    reply: "Έχεις meeting 10-11 με e-avenue, μετά ελεύθερος μέχρι τις 5.",
    source: "Calendar",
    confidence: 95,
    time: "0.9s",
  },
  WEATHER: {
    reply: "30°C στην Τρίπολη, ηλιοφάνεια. Πάρε αντηλιακό!",
    source: "Weather API",
    confidence: 97,
    time: "0.6s",
  },
  DEVOPS: {
    reply: "3 commits στο jarvis repo: intent classifier, API routes, tests.",
    source: "GitHub",
    confidence: 90,
    time: "1.1s",
  },
  NEWS: {
    reply: "Κυριότερα νέα: νέα μέτρα οικονομίας, πυρκαγιές στην Αττική, Euroleague αποτελέσματα.",
    source: "News API",
    confidence: 91,
    time: "0.9s",
  },
  KNOWLEDGE: {
    reply: "Σπούδασα Πληροφορική στο Ιόνιο Πανεπιστήμιο (Κέρκυρα), τώρα μεταπτυχιακό στο ΠαΠελ.",
    source: "RAG Corpus",
    confidence: 96,
    time: "1.0s",
  },
};

// Simple intent detection for demo fallback
function detectDemoIntent(msg) {
  const m = msg.toLowerCase();
  if (/καιρ[οό]|θερμοκρασ|βρ[εέ]χ|ζ[εέ]στ|κρ[υύ]ο/.test(m)) return "WEATHER";
  if (/νέα|ειδ[ηή]σ|news|τι γίνεται/.test(m)) return "NEWS";
  if (/ποι[οό]ς (είσαι|ε[ιί]σαι)|ηλικ|σπούδ|πτυχ|δουλε[υύ]|εργ[αά]ζ/.test(m)) return "PERSONAL";
  if (/πότε|free|meeting|ραντεβ|calendar|πρόγραμμα/.test(m)) return "SCHEDULE";
  if (/email|mail|παραγγε[λι]|tracking|στειλ/.test(m)) return "MEMORY";
  if (/commit|github|deploy|push|code|repo/.test(m)) return "DEVOPS";
  if (/κέρκυρα|πανεπιστ[ηή]μ|ιόνιο|thesis|διπλωματ/.test(m)) return "KNOWLEDGE";
  return "CASUAL";
}

// ── Neural Network SVG Component ───────────────────────────────
function NeuralNetwork({ activeNodes, animatingNodes, theme, hoveredNode, onNodeHover, onNodeClick }) {
  const t = themes[theme];
  const getNodeById = (id) => NODES.find((n) => n.id === id);

  return (
    <svg viewBox="0 0 400 540" style={{ width: "100%", height: "100%" }}>
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="glowStrong">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="glowHover">
          <feGaussianBlur stdDeviation="8" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Connections */}
      {CONNECTIONS.map((conn, i) => {
        const from = getNodeById(conn.from);
        const to = getNodeById(conn.to);
        const isActive =
          activeNodes.includes(conn.from) && activeNodes.includes(conn.to);
        const fromColors = getNodeColor(conn.from, theme);

        return (
          <line
            key={i}
            x1={from.x}
            y1={from.y}
            x2={to.x}
            y2={to.y}
            stroke={isActive ? fromColors.stroke : t.connectionInactive}
            strokeWidth={isActive ? 2 : 0.7}
            opacity={isActive ? 0.7 : 0.25}
            style={{
              transition: "all 0.6s ease",
              filter: isActive ? "url(#glow)" : "none",
            }}
          />
        );
      })}

      {/* Nodes */}
      {NODES.map((node) => {
        const isActive = activeNodes.includes(node.id);
        const isAnimating = animatingNodes.includes(node.id);
        const isHovered = hoveredNode === node.id;
        const colors = getNodeColor(node.id, theme);

        return (
          <g
            key={node.id}
            style={{
              cursor: "pointer",
              transition: "all 0.4s ease",
              transform: isAnimating ? "scale(1.15)" : isHovered ? "scale(1.08)" : "scale(1)",
              transformOrigin: `${node.x}px ${node.y}px`,
            }}
            onMouseEnter={() => onNodeHover(node.id)}
            onMouseLeave={() => onNodeHover(null)}
            onClick={() => onNodeClick(node.id)}
          >
            {/* Pulse ring */}
            {(isActive || isHovered) && (
              <circle
                cx={node.x}
                cy={node.y}
                r={node.r + 8}
                fill="none"
                stroke={colors.stroke}
                strokeWidth="0.7"
                opacity={isHovered ? 0.6 : 0.3}
                style={{
                  animation: isAnimating
                    ? "pulse-ring 1.5s ease-in-out infinite"
                    : "none",
                }}
              />
            )}

            {/* Hover outer ring */}
            {isHovered && (
              <circle
                cx={node.x}
                cy={node.y}
                r={node.r + 14}
                fill="none"
                stroke={colors.stroke}
                strokeWidth="0.4"
                opacity="0.2"
                strokeDasharray="3 3"
              />
            )}

            {/* Main circle */}
            <circle
              cx={node.x}
              cy={node.y}
              r={node.r}
              fill={isActive || isHovered ? colors.fill : t.nodeInactive}
              stroke={isActive || isHovered ? colors.stroke : t.connectionInactive}
              strokeWidth={isHovered ? 2.5 : isActive ? 2 : 0.8}
              style={{
                transition: "all 0.3s ease",
                filter: isHovered ? "url(#glowHover)" : isAnimating ? "url(#glowStrong)" : isActive ? "url(#glow)" : "none",
              }}
            />

            {/* Label */}
            <text
              x={node.x}
              y={node.y - 4}
              textAnchor="middle"
              fill={isActive || isHovered ? colors.text : t.textMuted}
              fontSize="10"
              fontWeight="600"
              fontFamily="inherit"
              style={{ transition: "fill 0.3s ease", pointerEvents: "none" }}
            >
              {node.label}
            </text>
            <text
              x={node.x}
              y={node.y + 8}
              textAnchor="middle"
              fill={isActive || isHovered ? colors.sub : t.textMuted}
              fontSize="7"
              fontFamily="inherit"
              opacity={isActive || isHovered ? 0.8 : 0.4}
              style={{ transition: "all 0.3s ease", pointerEvents: "none" }}
            >
              {node.sub}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Node Info Card ────────────────────────────────────────────
function NodeInfoCard({ nodeId, theme, onClose }) {
  if (!nodeId || !NODE_INFO[nodeId]) return null;
  const t = themes[theme];
  const info = NODE_INFO[nodeId];
  const colors = getNodeColor(nodeId, theme);

  return (
    <div
      style={{
        background: t.cardBg,
        border: `1px solid ${colors.stroke}44`,
        borderRadius: 12,
        padding: "14px 16px",
        backdropFilter: "blur(12px)",
        animation: "fadeIn 0.25s ease",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: colors.stroke }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: colors.text }}>{info.title}</span>
        </div>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: t.textMuted,
            cursor: "pointer",
            fontSize: 16,
            padding: "0 4px",
            lineHeight: 1,
          }}
        >
          ×
        </button>
      </div>
      <p style={{ fontSize: 11, color: t.text, lineHeight: 1.5, margin: "0 0 8px 0", opacity: 0.85 }}>
        {info.desc}
      </p>
      <div
        style={{
          fontSize: 9,
          color: colors.stroke,
          background: `${colors.stroke}11`,
          padding: "4px 8px",
          borderRadius: 6,
          display: "inline-block",
          letterSpacing: 0.3,
        }}
      >
        {info.tech}
      </div>
    </div>
  );
}

// ── Intent Badge Component ─────────────────────────────────────
function IntentBadge({ intent, theme }) {
  const t = themes[theme];
  const intentColors = {
    PERSONAL: { bg: t.orangeBg, text: t.orange, border: t.orange },
    KNOWLEDGE: { bg: t.accentBg, text: t.accent, border: t.accent },
    CASUAL: { bg: t.greenBg, text: t.green, border: t.green },
    SENSITIVE: { bg: t.redBg, text: t.red, border: t.red },
    MEMORY: { bg: t.greenBg, text: t.green, border: t.green },
    SCHEDULE: { bg: t.blueBg, text: t.blue, border: t.blue },
    DEVOPS: { bg: t.accentBg, text: t.accentLight, border: t.accent },
    WEATHER: { bg: t.blueBg, text: t.blue, border: t.blue },
    NEWS: { bg: t.accentBg, text: t.accent, border: t.accent },
  };
  const c = intentColors[intent] || intentColors.CASUAL;

  return (
    <span
      style={{
        fontSize: 10,
        padding: "2px 8px",
        borderRadius: 10,
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.border}33`,
        fontWeight: 500,
      }}
    >
      {intent}
    </span>
  );
}

// ── Who am I talking to? ───────────────────────────────────────
// Shown once per session, before the first message. Both fields are
// optional: skipping them selects the neutral register, which is the
// reserved one. Nothing typed here is stored — it lives in React state for
// the lifetime of the tab and is sent with each request so the backend can
// pick a tone, never to be written down.
function SpeakerPrompt({ t, onSubmit, onSkip }) {
  const [name, setName] = useState("");
  const [role, setRole] = useState("");

  const field = {
    width: "100%",
    padding: "10px 12px",
    marginTop: 6,
    borderRadius: 8,
    border: `1px solid ${t.border}`,
    background: t.cardBg || t.bg,
    color: t.text,
    fontSize: 14,
    outline: "none",
    boxSizing: "border-box",
  };
  const label = { fontSize: 12, opacity: 0.7, fontWeight: 600 };

  const submit = (e) => {
    e.preventDefault();
    onSubmit({ name: name.trim(), role: role.trim() });
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <form
        onSubmit={submit}
        style={{
          background: t.cardBg || t.bg,
          border: `1px solid ${t.border}`,
          borderRadius: 14,
          padding: 28,
          width: "min(420px, 90vw)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
        }}
      >
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
          Ποιος μου μιλάει;
        </div>
        <div style={{ fontSize: 13, opacity: 0.65, marginBottom: 20, lineHeight: 1.5 }}>
          Προσαρμόζω το ύφος μου ανάλογα. Δεν αποθηκεύεται τίποτα — μόλις
          κλείσεις τη σελίδα, ξεχνιέται.
        </div>

        <label style={label}>
          Όνομα
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="π.χ. Παναγιώτης"
            style={field}
          />
        </label>

        <div style={{ height: 14 }} />

        <label style={label}>
          Ιδιότητα
          <input
            value={role}
            onChange={(e) => setRole(e.target.value)}
            placeholder="π.χ. συνάδελφος, καθηγητής, φίλος"
            style={field}
          />
        </label>

        <div style={{ display: "flex", gap: 10, marginTop: 24 }}>
          <button
            type="submit"
            style={{
              flex: 1,
              padding: "10px 16px",
              borderRadius: 8,
              border: "none",
              background: t.accent || "#6366f1",
              color: "#fff",
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Ξεκίνα
          </button>
          <button
            type="button"
            onClick={onSkip}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              border: `1px solid ${t.border}`,
              background: "transparent",
              color: t.text,
              fontSize: 14,
              cursor: "pointer",
              opacity: 0.75,
            }}
          >
            Παράλειψη
          </button>
        </div>
      </form>
    </div>
  );
}

// ── Main App Component ─────────────────────────────────────────
export default function JarvisDigitalTwin() {
  const [theme, setTheme] = useState("dark");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [activeNodes, setActiveNodes] = useState([]);
  const [animatingNodes, setAnimatingNodes] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [twinEnabled, setTwinEnabled] = useState(true);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [isLive, setIsLive] = useState(false);
  const [stats, setStats] = useState({ messages: 0, sourcesUsed: 0, avgTime: 0, sources: 9 });

  // Who is talking, asked once per session. Held in component state only —
  // no localStorage, no cookie, nothing written to the server. Reloading the
  // page forgets it, which is the intended lifetime for someone else's name.
  const [speaker, setSpeaker] = useState({ name: "", role: "" });
  const [askSpeaker, setAskSpeaker] = useState(true);
  const chatEndRef = useRef(null);
  const t = themes[theme];

  const scrollToBottom = useCallback(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Animate nodes sequentially.
  //
  // Δέχεται προαιρετικά τις ΠΡΑΓΜΑΤΙΚΕΣ πηγές που χρησιμοποιήθηκαν. Ο
  // στατικός χάρτης INTENT_NODES μένει μόνο ως εφεδρικό, για όταν το
  // backend δεν τις αναφέρει.
  //
  // Ο χάρτης ήταν το μοναδικό κριτήριο: το CASUAL άναβε πάντα τέσσερις
  // κόμβους και το SCHEDULE πάντα πέντε, ανεξάρτητα από το τι εκτελέστηκε.
  // Ένα διάγραμμα που μοιάζει να δείχνει εκτέλεση ενώ δείχνει πρόθεση δεν
  // είναι απλώς ανακριβές — είναι πειστικό, και η πειστικότητα είναι το
  // πρόβλημα. Ο εξεταστής βλέπει έναν κόμβο να ανάβει και συμπεραίνει ότι
  // κάτι έτρεξε.
  const animatePipeline = useCallback((intent, usedSources) => {
    const fallback = INTENT_NODES[intent] || INTENT_NODES.CASUAL;
    let nodes = fallback;

    if (Array.isArray(usedSources) && usedSources.length > 0) {
      // Τα ονόματα του API στα id του διαγράμματος.
      const nodeFor = { rag: "rag", calendar: "calendar", email: "gmail",
                        weather: "weather", news: "news", github: "github",
                        identity: "identity" };
      const middle = usedSources.map((s) => nodeFor[s]).filter(Boolean);
      nodes = ["intent", ...middle, "krikri", "factcheck", "guardrails",
               "output"];
    }

    setActiveNodes([]);
    setAnimatingNodes([]);

    nodes.forEach((nodeId, i) => {
      setTimeout(() => {
        setActiveNodes((prev) => [...prev, nodeId]);
        setAnimatingNodes((prev) => [...prev, nodeId]);
        setTimeout(() => {
          setAnimatingNodes((prev) => prev.filter((n) => n !== nodeId));
        }, 600);
      }, i * 200);
    });
  }, []);

  const handleNodeClick = useCallback((nodeId) => {
    setSelectedNode((prev) => (prev === nodeId ? null : nodeId));
  }, []);

  // ── Send message ──
  const handleSend = useCallback(async () => {
    const userMsg = input.trim();
    if (!userMsg) return;

    setMessages((prev) => [...prev, { type: "user", text: userMsg }]);
    setInput("");
    setIsTyping(true);

    const startTime = performance.now();

    try {
      // Build conversation history from previous messages
      const history = messages
        .filter((m) => m.type === "user" || m.type === "twin")
        .slice(-10)
        .map((m) => ({
          role: m.type === "user" ? "user" : "assistant",
          content: m.text,
        }));

      const resp = await fetch("/webhook/twin-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg,
          history,
          speaker_name: speaker.name,
          speaker_role: speaker.role,
        }),
      });

      const data = await resp.json();
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

      const intent = (data.intent || "CASUAL").toUpperCase();
      // sources_used: ό,τι όντως συνεισέφερε context. Αν το backend είναι
      // παλιότερο και δεν το στέλνει, η animatePipeline πέφτει στον χάρτη.
      animatePipeline(intent, data.sources_used);

      // Η ετικέτα δείχνει ποιες πηγές ΑΠΑΝΤΗΣΑΝ, όχι ποιες υποτίθεται
      // ότι αντιστοιχούν στο intent.
      //
      // Ο παλιός χάρτης έλεγε «via Calendar» για κάθε ερώτηση
      // δρομολογημένη σε SCHEDULE — ακόμα και όταν το ημερολόγιο δεν είχε
      // διαπιστευτήρια και δεν είχε επιστρέψει τίποτα. Ο χρήστης έβλεπε
      // την ετικέτα, έβλεπε τον κόμβο αναμμένο, και συμπέραινε ότι η
      // απάντηση στηρίζεται σε δεδομένα. Στηριζόταν σε επινόηση.
      const sourceLabels = {
        rag: "RAG Corpus", calendar: "Calendar", email: "Gmail",
        weather: "Weather API", news: "News API", github: "GitHub",
        identity: "Identity",
      };
      const used = Array.isArray(data.sources_used) ? data.sources_used : null;
      const sourceText = used && used.length
        ? used.map((s) => sourceLabels[s] || s).join(" + ")
        : (used ? "Krikri LLM (χωρίς πηγές)" : intentSourceFallback[intent]);

      setTimeout(() => {
        setIsTyping(false);
        setMessages((prev) => [
          ...prev,
          {
            type: "twin",
            text: data.reply || data.error || "Δεν μπόρεσα να απαντήσω.",
            intent,
            source: sourceText || "Unknown",
            confidence: Math.round((data.confidence || 0) * 100),
            time: `${elapsed}s`,
          },
        ]);
        setStats((s) => ({
          ...s,
          sourcesUsed: Array.isArray(data.sources_used) ? data.sources_used.length : 0,
          messages: s.messages + 1,
          avgTime: ((parseFloat(s.avgTime || 0) * s.messages + parseFloat(elapsed)) / (s.messages + 1)).toFixed(1),
        }));
        setIsLive(true);
      }, 800);
    } catch {
      // Fallback: smart demo matching
      const intent = detectDemoIntent(userMsg);
      const demo = DEMO_CONVERSATIONS[intent] || DEMO_CONVERSATIONS.CASUAL;
      animatePipeline(intent);

      setTimeout(() => {
        setIsTyping(false);
        setMessages((prev) => [
          ...prev,
          {
            type: "twin",
            text: demo.reply,
            intent,
            source: demo.source,
            confidence: demo.confidence,
            time: demo.time,
            isDemo: true,
          },
        ]);
        setStats((s) => ({ ...s, messages: s.messages + 1 }));
      }, 1200);
    }
  }, [input, animatePipeline, messages, speaker]);

  // ── Feedback handlers ───────────────────────────────────────
  const handleFeedback = useCallback(async (msgIndex, rating) => {
    const msg = messages[msgIndex];
    // Find the user message that preceded this twin response
    let userMsg = "";
    for (let j = msgIndex - 1; j >= 0; j--) {
      if (messages[j].type === "user") { userMsg = messages[j].text; break; }
    }

    if (rating === -1) {
      // Show correction input
      setMessages((prev) =>
        prev.map((m, i) => (i === msgIndex ? { ...m, showCorrection: true, rated: -1 } : m))
      );
    } else {
      setMessages((prev) =>
        prev.map((m, i) => (i === msgIndex ? { ...m, rated: 1 } : m))
      );
    }

    try {
      await fetch("/orchestration/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg, reply: msg.text,
          intent: (msg.intent || "").toLowerCase(), rating,
        }),
      });
    } catch { /* silent */ }
  }, [messages]);

  const submitCorrection = useCallback(async (msgIndex, correction) => {
    const msg = messages[msgIndex];
    let userMsg = "";
    for (let j = msgIndex - 1; j >= 0; j--) {
      if (messages[j].type === "user") { userMsg = messages[j].text; break; }
    }
    setMessages((prev) =>
      prev.map((m, i) => (i === msgIndex ? { ...m, showCorrection: false, rated: -1 } : m))
    );
    try {
      await fetch("/orchestration/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg, reply: msg.text,
          intent: (msg.intent || "").toLowerCase(), rating: -1, correction,
        }),
      });
    } catch { /* silent */ }
  }, [messages]);

  return (
    <div
      style={{
        background: t.bg,
        minHeight: "100vh",
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        color: t.text,
        transition: "background 0.4s ease, color 0.4s ease",
      }}
    >
      {askSpeaker && (
        <SpeakerPrompt
          t={t}
          onSubmit={(s) => { setSpeaker(s); setAskSpeaker(false); }}
          onSkip={() => setAskSpeaker(false)}
        />
      )}
      <style>{`
        @keyframes pulse-ring {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }
        @keyframes typing-dot {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .msg-enter { animation: fadeIn 0.3s ease; }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: ${t.accent}33; border-radius: 2px; }
        ::-webkit-scrollbar-track { background: transparent; }
      `}</style>

      {/* ── Header ──────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "14px 24px",
          borderBottom: `1px solid ${t.border}`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 40, height: 40, borderRadius: "50%",
              border: `2px solid ${t.accent}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: t.accentBg,
            }}
          >
            <span style={{ fontSize: 18, fontWeight: 600, color: t.accent }}>J</span>
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: 0.5 }}>Jarvis George</div>
            <div style={{ fontSize: 11, color: t.accent, letterSpacing: 1, textTransform: "uppercase" }}>
              Digital Twin v5.0{isLive ? " · LIVE" : ""}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            style={{
              background: t.surfaceLight, border: `1px solid ${t.border}`,
              borderRadius: 20, padding: "6px 14px", cursor: "pointer",
              fontSize: 12, color: t.text, display: "flex", alignItems: "center", gap: 6,
              transition: "all 0.3s ease",
            }}
          >
            {theme === "dark" ? "☀" : "☾"} {theme === "dark" ? "Light" : "Dark"}
          </button>

          <div onClick={() => setTwinEnabled(!twinEnabled)} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
            <div
              style={{
                width: 40, height: 22, borderRadius: 11,
                background: twinEnabled ? `${t.green}44` : `${t.red}44`,
                padding: 2, transition: "background 0.3s ease",
                display: "flex", alignItems: "center",
              }}
            >
              <div
                style={{
                  width: 18, height: 18, borderRadius: "50%",
                  background: twinEnabled ? t.green : t.red,
                  transform: twinEnabled ? "translateX(18px)" : "translateX(0)",
                  transition: "all 0.3s ease",
                }}
              />
            </div>
            <span style={{ fontSize: 11, color: twinEnabled ? t.green : t.red }}>
              {twinEnabled ? "ACTIVE" : "OFF"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Main Content ────────────────────────────────────── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          height: "calc(100vh - 120px)",
          gap: 0,
        }}
      >
        {/* ── Left: Pipeline ───────────────────────────────── */}
        <div
          style={{
            borderRight: `1px solid ${t.border}`,
            padding: "12px 16px",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <div style={{
            fontSize: 11, color: t.textMuted, textTransform: "uppercase",
            letterSpacing: 1.5, marginBottom: 6, fontWeight: 500,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span>Pipeline Network</span>
            {hoveredNode && (
              <span style={{ color: getNodeColor(hoveredNode, theme).stroke, letterSpacing: 0, textTransform: "none", fontSize: 10 }}>
                Click for details
              </span>
            )}
          </div>

          <div style={{ flex: 1, minHeight: 0 }}>
            <NeuralNetwork
              activeNodes={activeNodes}
              animatingNodes={animatingNodes}
              theme={theme}
              hoveredNode={hoveredNode}
              onNodeHover={setHoveredNode}
              onNodeClick={handleNodeClick}
            />
          </div>

          {/* Node info card */}
          {selectedNode && (
            <NodeInfoCard
              nodeId={selectedNode}
              theme={theme}
              onClose={() => setSelectedNode(null)}
            />
          )}

          {/* Data sources */}
          {!selectedNode && (
            <div
              style={{
                display: "flex", flexWrap: "wrap", gap: 4,
                justifyContent: "center", padding: "8px 0",
                borderTop: `1px solid ${t.border}`,
              }}
            >
              {["Gmail", "Calendar", "GitHub", "Weather", "News", "RAG", "Identity", "Krikri", "Guardrails"].map(
                (src) => (
                  <span
                    key={src}
                    style={{
                      fontSize: 9, padding: "2px 8px", borderRadius: 10,
                      border: `1px solid ${t.border}`, color: t.textMuted, background: t.surfaceLight,
                    }}
                  >
                    {src}
                  </span>
                )
              )}
            </div>
          )}
        </div>

        {/* ── Right: Chat ───────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          {/* Messages */}
          <div
            style={{
              flex: 1, overflowY: "auto", padding: "20px 24px",
              display: "flex", flexDirection: "column", gap: 14,
            }}
          >
            {messages.length === 0 && (
              <div
                style={{
                  display: "flex", flexDirection: "column", alignItems: "center",
                  justifyContent: "center", height: "100%", gap: 12, opacity: 0.5,
                }}
              >
                <div
                  style={{
                    width: 60, height: 60, borderRadius: "50%",
                    border: `2px solid ${t.accent}`, display: "flex",
                    alignItems: "center", justifyContent: "center", background: t.accentBg,
                  }}
                >
                  <span style={{ fontSize: 28, fontWeight: 700, color: t.accent }}>J</span>
                </div>
                <div style={{ fontSize: 14, color: t.textMuted }}>
                  Ρώτα το digital twin σου ό,τι θέλεις
                </div>
                <div style={{ fontSize: 11, color: t.textMuted }}>
                  Δοκίμασε: "Τι κάνεις;" ή "Τι καιρό κάνει;"
                </div>
              </div>
            )}

            {messages.map((msg, i) =>
              msg.type === "user" ? (
                <div key={i} className="msg-enter" style={{ alignSelf: "flex-end", maxWidth: "75%" }}>
                  <div
                    style={{
                      background: t.bubbleUser, borderRadius: "18px 18px 4px 18px",
                      padding: "10px 16px", color: "#fff", fontSize: 13,
                    }}
                  >
                    {msg.text}
                  </div>
                </div>
              ) : (
                <div key={i} className="msg-enter" style={{ alignSelf: "flex-start", maxWidth: "85%" }}>
                  <div style={{ display: "flex", gap: 5, marginBottom: 4, flexWrap: "wrap" }}>
                    <IntentBadge intent={msg.intent} theme={theme} />
                    <span
                      style={{
                        fontSize: 10, padding: "2px 8px", borderRadius: 10,
                        background: t.accentBg, color: t.accentLight,
                        border: `1px solid ${t.accent}33`,
                      }}
                    >
                      via {msg.source}
                    </span>
                    {msg.isDemo && (
                      <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 10, background: t.orangeBg, color: t.orange }}>
                        demo
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      background: t.bubbleTwin, border: `1px solid ${t.bubbleTwinBorder}`,
                      borderRadius: "4px 18px 18px 18px", padding: "10px 16px",
                    }}
                  >
                    <div style={{ fontSize: 13 }}>{msg.text}</div>
                    <div
                      style={{
                        fontSize: 10, color: t.textMuted, marginTop: 6,
                        display: "flex", gap: 8, alignItems: "center",
                      }}
                    >
                      <span>{msg.time}</span>
                      <span style={{ color: t.green, display: "flex", alignItems: "center", gap: 3 }}>
                        ✓ {msg.confidence}%
                      </span>
                      {!msg.isDemo && !msg.rated && (
                        <span style={{ display: "flex", gap: 4, marginLeft: 8 }}>
                          <button
                            onClick={() => handleFeedback(i, 1)}
                            style={{
                              background: "none", border: "none", cursor: "pointer",
                              fontSize: 14, opacity: 0.6, padding: "0 2px",
                            }}
                            title="Σωστή απάντηση"
                          >
                            👍
                          </button>
                          <button
                            onClick={() => handleFeedback(i, -1)}
                            style={{
                              background: "none", border: "none", cursor: "pointer",
                              fontSize: 14, opacity: 0.6, padding: "0 2px",
                            }}
                            title="Λάθος απάντηση — γράψε τη σωστή"
                          >
                            👎
                          </button>
                        </span>
                      )}
                      {msg.rated === 1 && <span style={{ fontSize: 12, opacity: 0.7 }}>👍</span>}
                      {msg.rated === -1 && <span style={{ fontSize: 12, opacity: 0.7 }}>👎</span>}
                    </div>
                    {msg.showCorrection && (
                      <div style={{ marginTop: 6, display: "flex", gap: 4 }}>
                        <input
                          type="text"
                          placeholder="Γράψε τη σωστή απάντηση..."
                          style={{
                            flex: 1, fontSize: 11, padding: "4px 8px", borderRadius: 8,
                            border: `1px solid ${t.accent}44`, background: t.bg,
                            color: t.text, outline: "none",
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && e.target.value.trim()) {
                              submitCorrection(i, e.target.value.trim());
                              e.target.value = "";
                            }
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )
            )}

            {isTyping && (
              <div className="msg-enter" style={{ alignSelf: "flex-start" }}>
                <div
                  style={{
                    background: t.bubbleTwin, border: `1px solid ${t.bubbleTwinBorder}`,
                    borderRadius: "4px 18px 18px 18px", padding: "12px 20px",
                    display: "flex", gap: 4,
                  }}
                >
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      style={{
                        width: 6, height: 6, borderRadius: "50%",
                        background: t.accent,
                        animation: `typing-dot 1s ease-in-out ${i * 0.2}s infinite`,
                      }}
                    />
                  ))}
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div
            style={{
              padding: "12px 24px", borderTop: `1px solid ${t.border}`,
              display: "flex", gap: 10, alignItems: "center",
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ρώτα το twin σου... (Enter για αποστολή)"
              disabled={!twinEnabled}
              style={{
                flex: 1, background: t.inputBg, border: `1px solid ${t.border}`,
                borderRadius: 20, padding: "10px 18px", color: t.text,
                fontSize: 13, outline: "none", transition: "border-color 0.3s ease",
              }}
              onFocus={(e) => (e.target.style.borderColor = t.accent)}
              onBlur={(e) => (e.target.style.borderColor = t.border)}
            />
            <button
              onClick={handleSend}
              disabled={!twinEnabled}
              style={{
                width: 38, height: 38, borderRadius: "50%", background: t.accent,
                border: "none", cursor: twinEnabled ? "pointer" : "not-allowed",
                display: "flex", alignItems: "center", justifyContent: "center",
                opacity: twinEnabled ? 1 : 0.4, transition: "opacity 0.3s ease, transform 0.15s ease",
              }}
              onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.92)")}
              onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>

          {/* Stats bar */}
          <div
            style={{
              display: "flex", justifyContent: "space-around",
              padding: "10px 24px", borderTop: `1px solid ${t.border}`, background: t.statsBar,
            }}
          >
            {/*
              Το «Accuracy 96%» ήταν σταθερά στον κώδικα και δεν άλλαζε
              ποτέ. Ένα ποσοστό ακρίβειας που δεν μετριέται είναι ακριβώς η
              ψευδοακρίβεια που το ίδιο το σύστημα ελέγχεται να μην παράγει
              — και βρισκόταν μόνιμα στην οθόνη, μπροστά σε όποιον κάνει
              την επίδειξη. Η ερώτηση «τι σημαίνει αυτό το 96%;» δεν έχει
              καλή απάντηση.

              Αντικαταστάθηκε από το πόσες πηγές ΑΠΑΝΤΗΣΑΝ στο τελευταίο
              μήνυμα, που είναι μετρημένο και αλλάζει.
            */}
            {[
              { value: stats.messages, label: "Messages", color: t.accent },
              { value: `${stats.sourcesUsed ?? 0}/${stats.sources}`,
                label: "Πηγές που απάντησαν",
                color: (stats.sourcesUsed ?? 0) > 0 ? t.green : t.textMuted },
              { value: `${stats.avgTime || 0}s`, label: "Avg response", color: t.orange },
              { value: stats.sources, label: "Data sources", color: t.accentLight },
            ].map((stat, i) => (
              <div key={i} style={{ textAlign: "center" }}>
                <div style={{ fontSize: 16, fontWeight: 600, color: stat.color }}>{stat.value}</div>
                <div style={{ fontSize: 9, color: t.textMuted, textTransform: "uppercase", letterSpacing: 1 }}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
