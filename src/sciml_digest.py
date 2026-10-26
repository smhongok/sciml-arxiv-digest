"""
SciML arXiv Digest: daily digest of new arXiv papers on AI for CFD / neural PDE solvers.

Pipeline:
  1. Fetch today's announcements for several arXiv categories from the arXiv RSS feed.
  2. Cheap keyword prefilter on title + abstract (configurable).
  3. LLM screening: relevancy score 1-10 from title + abstract, in batches.
  4. LLM deep analysis of top papers using the arXiv HTML full text.
  5. Render an HTML digest (digest/ + ./digest.html) and optionally send it by email.

Usage:
  python src/sciml_digest.py                    # full run
  python src/sciml_digest.py --no-llm           # fetch + prefilter only (no API key needed)
  python src/sciml_digest.py --no-email --limit 30
"""
import argparse
import datetime
import html
import json
import os
import re
import smtplib
import ssl
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import DEFAULT_MODELS, LLMClient, detect_provider
from paths import DATA_DIR, DIGEST_DIR, ROOT_DIR

USER_AGENT = "Mozilla/5.0 (compatible; SciML-arXiv-Digest/1.0)"
RSS_URL = "https://rss.arxiv.org/rss/{}"
NS = {"arxiv": "http://arxiv.org/schemas/atom", "dc": "http://purl.org/dc/elements/1.1/"}


# ---------------------------------------------------------------------------
# 1. Fetch papers
# ---------------------------------------------------------------------------
def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_papers(categories, include_cross_lists=True, include_replacements=False):
    """Fetch today's arXiv announcements for the given categories (deduplicated by arXiv id)."""
    allowed_types = {"new"}
    if include_cross_lists:
        allowed_types.add("cross")
    if include_replacements:
        allowed_types.update({"replace", "replace-cross"})

    url = RSS_URL.format("+".join(categories))
    print(f"Fetching {url}")
    root = ET.fromstring(http_get(url))

    papers = {}
    for item in root.iter("item"):
        announce_type = (item.findtext("arxiv:announce_type", default="", namespaces=NS) or "").strip()
        if announce_type not in allowed_types:
            continue
        link = item.findtext("link", default="").strip()
        arxiv_id = link.rstrip("/").split("/")[-1]
        if not arxiv_id or arxiv_id in papers:
            continue
        description = item.findtext("description", default="")
        abstract = description.split("Abstract:", 1)[-1].strip()
        papers[arxiv_id] = {
            "id": arxiv_id,
            "title": " ".join(item.findtext("title", default="").split()),
            "authors": " ".join(item.findtext("dc:creator", default="", namespaces=NS).split()),
            "abstract": " ".join(abstract.split()),
            "categories": [c.text for c in item.findall("category") if c.text],
            "announce_type": announce_type,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
            "html": f"https://arxiv.org/html/{arxiv_id}",
        }
    return list(papers.values())


# ---------------------------------------------------------------------------
# 2. Keyword prefilter
# ---------------------------------------------------------------------------
def _compile(keywords):
    if not keywords:
        return None
    # Whole-word match (optional plural "s"); a trailing "*" makes the keyword a prefix, e.g. "turbulen*".
    alts = [re.escape(k[:-1]) if k.endswith("*") else re.escape(k) + r"s?\b" for k in keywords]
    return re.compile(r"\b(?:" + "|".join(alts) + ")", re.IGNORECASE)


def keyword_prefilter(papers, cfg):
    if not cfg or not cfg.get("enabled", True):
        return papers
    strong, ml, domain = (_compile(cfg.get(k)) for k in ("strong", "ml", "domain"))
    kept = []
    for p in papers:
        text = f"{p['title']} {p['abstract']}"
        if (strong and strong.search(text)) or (ml and domain and ml.search(text) and domain.search(text)):
            kept.append(p)
    return kept


# ---------------------------------------------------------------------------
# 3. LLM screening
# ---------------------------------------------------------------------------
SCREEN_SYSTEM = (
    "You are an expert in scientific machine learning, computational fluid dynamics and numerical "
    "methods for PDEs. You screen new arXiv papers for a researcher and judge how relevant each one "
    "is to their stated interests. Be calibrated: 9-10 = core to the interests, 7-8 = clearly relevant, "
    "5-6 = partially related / worth a glance, 1-4 = not relevant."
)

SCREEN_PROMPT = """Researcher's interests:
{interest}

Score each paper below. Respond with ONLY a JSON array, one object per paper, in this exact form:
[{{"id": "<arXiv id>", "score": <integer 1-10>, "reason": "<one short sentence in {language}{reason_style}>", "tags": ["<2-4 short English topic tags, e.g. 'neural operator', 'turbulence', 'PINN'>"]}}]

Papers:
{papers}"""

# Writing style for summaries, keyed by config `language`. Other languages fall back to "default".
STYLE_GUIDES = {
    "Korean": {
        "reason": " 음슴체(~함/~임로 끝남), 문장 속 핵심 표현 하나를 **굵게**, 예: '**neural operator**로 CFD surrogate를 학습하여 관심사와 직접 관련됨'",
        "rules": """작성 규칙 (반드시 지킬 것):
- 모든 내용을 한국어로 작성함. 논문 본문이 영어여도 마찬가지임.
- 모델명, PDE 이름, 지표, 기술 용어는 영어 그대로 둠 (예: FNO, Navier-Stokes, relative L2 error, PINN).
- 음슴체로 씀: 모든 문장을 "~함", "~임", "~됨", "~있음", "~보임" 등으로 끝냄. "~합니다", "~한다"는 쓰지 않음.
- 불릿 하나에 한 문장만 씀. 짧고 간결하게 (대략 80자 이내).
- 근거, 수치, 예시는 하위 불릿으로 분리함.
- 불릿마다 가장 중요한 핵심 표현 하나를 마크다운 **굵게**로 강조함. 문장 전체나 고유명사만 굵게 하지 않음.
- 논문에 없는 내용은 지어내지 말고 "논문에 명시 없음"이라고 씀.""",
        "example": """{"tldr": "**multi-head FNO** 기반 autoregressive surrogate로 4D 저류층 history matching을 가속함",
 "method": ["유정 데이터를 예측하는 **autoregressive multi-head FNO**를 제안함",
            ["학습 시 **truncated BPTT**로 장기 rollout 안정성을 확보함"],
            "물리 잔차를 loss에 넣는 **PINO 방식**으로 데이터 의존도를 낮춤"]}""",
    },
    "default": {
        "reason": "",
        "rules": """Writing rules:
- Write everything in {language}; keep standard technical terms (model names, PDE names, metrics) in English.
- One short sentence per bullet; put evidence, numbers and examples in sub-bullets.
- In every bullet, mark the single most important phrase in markdown **bold**.
- Do not invent details: if something is not stated in the paper, say so.""",
        "example": """{"tldr": "An **autoregressive multi-head FNO** surrogate speeds up 4D reservoir history matching",
 "method": ["Proposes an **autoregressive multi-head FNO** that predicts well data",
            ["Uses **truncated BPTT** to stabilise long rollouts"]]}""",
    },
}


def style_guide(language):
    guide = STYLE_GUIDES.get(language, STYLE_GUIDES["default"])
    return {k: v.replace("{language}", language) for k, v in guide.items()}


def screen_papers(papers, client, interest, language, batch_size=10):
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        print(f"Screening papers {start + 1}-{start + len(batch)} / {len(papers)} with {client.model}")
        listing = "\n\n".join(
            f"[id: {p['id']}]\nTitle: {p['title']}\nAbstract: {p['abstract']}" for p in batch
        )
        prompt = SCREEN_PROMPT.format(interest=interest.strip(), language=language,
                                      reason_style=style_guide(language)["reason"], papers=listing)
        try:
            results = client.complete_json(SCREEN_SYSTEM, prompt, max_tokens=8000, effort="low")
        except Exception as e:
            print(f"  ! Screening batch failed: {e}")
            results = []
        by_id = {str(r.get("id", "")).strip(): r for r in results if isinstance(r, dict)}
        for p in batch:
            r = by_id.get(p["id"])
            if r is None:
                p.update(score=0, reason="(스크리닝 결과 없음)", tags=[])
                continue
            try:
                p["score"] = int(r.get("score", 0))
            except (TypeError, ValueError):
                p["score"] = 0
            p["reason"] = r.get("reason", "")
            p["tags"] = r.get("tags", []) or []
    return papers


# ---------------------------------------------------------------------------
# 4. Deep analysis with full text
# ---------------------------------------------------------------------------
ANALYSIS_SYSTEM = """You are an expert reviewer in scientific machine learning (neural operators, PINNs, ML for CFD, learned PDE solvers). You write concise, technically precise paper briefings in {language} for a researcher in this field.

{rules}"""

ANALYSIS_PROMPT = """Researcher's interests:
{interest}

Paper:
Title: {title}
Authors: {authors}
Abstract: {abstract}

Full text (may be truncated or unavailable):
{fulltext}

---
Write a briefing on the paper above. Respond with ONLY a JSON object with these keys.
"tldr" is one sentence. Every other key is an array of 1-4 bullet strings; a nested array right after a bullet holds its sub-bullets.
{{
  "tldr": "one-sentence summary",
  "problem": ["target PDE / physical system / task, and why it is hard"],
  "method": ["core idea and architecture"],
  "physics": ["how physics or numerical structure is used (PDE residual loss, conservation, symmetry, hybrid solver coupling, ...), or that it is not used"],
  "data": ["datasets / benchmarks / simulation setups (e.g. PDEBench, The Well, Re numbers, resolutions)"],
  "results": ["main quantitative results vs baselines (speed-up, error)"],
  "limitations": ["limitations and open questions (stated or apparent)"],
  "relevance": ["why this matters for the researcher's interests, and what they might take from it"]
}}

Style example (format only, not content):
{example}

Remember: write in {language}, following the writing rules exactly."""


def hangul_ratio(obj):
    """Share of Hangul among letters in a JSON-able object (Korean briefings with English terms are ~0.5)."""
    text = json.dumps(obj, ensure_ascii=False)
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return hangul / max(1, hangul + latin)


def fetch_fulltext(paper, max_chars):
    """Fetch the arXiv HTML rendering of a paper; returns (text, code_links)."""
    try:
        soup = BeautifulSoup(http_get(paper["html"]), "html.parser")
    except Exception as e:
        print(f"  ! No HTML full text for {paper['id']}: {e}")
        return "", []
    content = soup.find("div", class_="ltx_page_content") or soup.body
    if content is None:
        return "", []
    repo_re = re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co)/[\w.-]+/[\w.-]+")
    code_links = sorted({
        m.group(0).removesuffix(".git").rstrip(".")
        for a in content.find_all("a", href=True)
        if (m := repo_re.match(a["href"]))
    })
    text = " ".join(p.get_text(" ", strip=True) for p in content.find_all(["h1", "h2", "h3", "p", "figcaption"]))
    return text[:max_chars], code_links[:5]


def analyze_paper(paper, client, interest, language, max_chars):
    fulltext, code_links = fetch_fulltext(paper, max_chars)
    paper["code_links"] = code_links
    guide = style_guide(language)
    system = ANALYSIS_SYSTEM.format(language=language, rules=guide["rules"])
    prompt = ANALYSIS_PROMPT.format(
        interest=interest.strip(), language=language, title=paper["title"], authors=paper["authors"],
        abstract=paper["abstract"], fulltext=fulltext or "(not available - rely on the abstract)",
        example=guide["example"],
    )
    try:
        analysis = client.complete_json(system, prompt, max_tokens=12000)
        # Models occasionally drift into the paper's language (English); retry once if so.
        if language == "Korean" and hangul_ratio(analysis) < 0.15:
            print(f"  ! {paper['id']} came back in English; retrying")
            analysis = client.complete_json(
                system, prompt + "\n\n이전 응답이 영어로 작성되었음. 반드시 한국어 음슴체로 다시 작성할 것.",
                max_tokens=12000,
            )
        paper["analysis"] = analysis
        print(f"  Analyzed {paper['id']}: {paper['title'][:70]}")
    except Exception as e:
        print(f"  ! Analysis failed for {paper['id']}: {e}")
    return paper


# ---------------------------------------------------------------------------
# 5. HTML rendering
# ---------------------------------------------------------------------------
ANALYSIS_LABELS = [
    ("problem", "문제 / 대상 PDE"),
    ("method", "방법"),
    ("physics", "물리 정보 활용"),
    ("data", "데이터 / 벤치마크"),
    ("results", "주요 결과"),
    ("limitations", "한계"),
    ("relevance", "나에게 의미"),
]

CSS = """
body{margin:0;padding:24px 12px;background:#f4f5f7;font-family:-apple-system,'Segoe UI','Apple SD Gothic Neo','Malgun Gothic',sans-serif;color:#1f2328;line-height:1.55}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.meta{color:#656d76;font-size:13px;margin-bottom:20px}
h2{font-size:16px;margin:28px 0 10px;color:#3d444d}
.card{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:16px 18px;margin-bottom:14px}
.top{display:flex;gap:12px;align-items:flex-start}
.score{flex:none;min-width:34px;text-align:center;font-weight:700;font-size:15px;border-radius:8px;padding:5px 6px;color:#fff;background:#8c959f}
.s9{background:#1a7f37}.s7{background:#2f81f7}.s5{background:#9a6700}
.title{font-size:16px;font-weight:600;color:#0b57d0;text-decoration:none}
.authors{color:#656d76;font-size:13px;margin-top:2px}
.tags{margin-top:6px}
.tag{display:inline-block;font-size:12px;background:#eef1f5;color:#3d444d;border-radius:999px;padding:1px 9px;margin:2px 4px 0 0}
.cat{background:#fff4e5;color:#7a4b00}
.reason{margin-top:8px;font-size:14px}
.tldr{margin-top:12px;padding:10px 12px;background:#f0f6ff;border-left:3px solid #2f81f7;border-radius:4px;font-size:14.5px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:14px}
td{vertical-align:top;padding:6px 8px;border-top:1px solid #eef1f5}
td.k{width:110px;color:#656d76;white-space:nowrap;font-weight:600}
td ul{margin:0;padding-left:18px}
td ul ul{margin-top:2px;color:#3d444d;font-size:13.5px}
li{margin:2px 0}
b{color:#0d1117;font-weight:700}
.links{margin-top:10px;font-size:13px}
.links a{color:#0b57d0;margin-right:12px}
details{margin-top:8px;font-size:13.5px;color:#3d444d}
"""


def _esc(x):
    if isinstance(x, list):
        x = "; ".join(str(i) for i in x)
    return html.escape(str(x or ""))


def _md(x):
    """Escape text and render markdown **bold** as <b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(x))


def _bullets(value):
    """Render a bullet list; a nested list right after an item becomes that item's sub-bullets."""
    if not isinstance(value, list):
        return _md(value)
    items = []
    for v in value:
        if isinstance(v, list) and items:
            items[-1] = items[-1].removesuffix("</li>") + _bullets(v) + "</li>"
        elif isinstance(v, list):
            items.append(f"<li>{_bullets(v)}</li>")
        else:
            items.append(f"<li>{_md(v)}</li>")
    return f'<ul>{"".join(items)}</ul>'


def _score_class(s):
    return "s9" if s >= 9 else "s7" if s >= 7 else "s5" if s >= 5 else ""


def render_card(p):
    tags = "".join(f'<span class="tag">{_esc(t)}</span>' for t in p.get("tags", []))
    tags += "".join(f'<span class="tag cat">{_esc(c)}</span>' for c in p["categories"][:3])
    out = [
        '<div class="card"><div class="top">',
        f'<div class="score {_score_class(p["score"])}">{p["score"]}</div><div>',
        f'<a class="title" href="{p["url"]}">{_esc(p["title"])}</a>',
        f'<div class="authors">{_esc(p["authors"])}</div>',
        f'<div class="tags">{tags}</div></div></div>',
    ]
    analysis = p.get("analysis")
    if analysis:
        out.append(f'<div class="tldr">💡 {_md(analysis.get("tldr"))}</div><table>')
        for key, label in ANALYSIS_LABELS:
            if analysis.get(key):
                out.append(f'<tr><td class="k">{label}</td><td>{_bullets(analysis[key])}</td></tr>')
        out.append("</table>")
    else:
        out.append(f'<div class="reason">{_md(p.get("reason"))}</div>')
    links = [f'<a href="{p["url"]}">abs</a>', f'<a href="{p["pdf"]}">pdf</a>', f'<a href="{p["html"]}">html</a>']
    links += [f'<a href="{_esc(u)}">🔗 {_esc(u.split("://", 1)[-1])}</a>' for u in p.get("code_links", [])]
    out.append(f'<div class="links">{"".join(links)}</div>')
    out.append(f'<details><summary>Abstract</summary>{_esc(p["abstract"])}</details></div>')
    return "\n".join(out)


def render_html(analyzed, briefs, extras, stats, date_str):
    """analyzed: full briefings; briefs: one-line cards; extras: title-only list beyond the caps."""
    body = []
    if not (analyzed or briefs):
        body.append('<div class="card">오늘은 관심 기준을 넘은 논문이 없습니다.</div>')
    if analyzed:
        body.append(f"<h2>상세 분석 ({len(analyzed)})</h2>")
        body += [render_card(p) for p in analyzed]
    if briefs:
        body.append(f"<h2>함께 볼 만한 논문 ({len(briefs)})</h2>")
        body += [render_card(p) for p in briefs]
    if extras:
        rows = "".join(
            f'<li><b>{p["score"]}</b> · <a href="{p["url"]}">{_esc(p["title"])}</a><br>{_md(p.get("reason"))}</li>'
            for p in extras
        )
        body.append(f'<details class="card"><summary>그 외 관련 논문 {len(extras)}편 (제목만)</summary><ul>{rows}</ul></details>')
    shown = len(analyzed) + len(briefs) + len(extras)
    skipped = f" · 이전에 본 논문 {stats['seen']}편 제외" if stats.get("seen") else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SciML arXiv Digest {date_str}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>SciML arXiv Digest · {date_str}</h1>
<div class="meta">전체 {stats['fetched']}편{skipped} → 키워드 필터 {stats['prefiltered']}편 → LLM 스크리닝 후 {shown}편 선정 · {_esc(stats['model'])}</div>
{''.join(body)}
</div></body></html>"""


# ---------------------------------------------------------------------------
# 6. History of already-screened papers (so the same paper is never sent twice)
# ---------------------------------------------------------------------------
def load_history(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_history(path, history, keep_days=365):
    cutoff = (datetime.date.today() - datetime.timedelta(days=keep_days)).isoformat()
    history = {k: v for k, v in history.items() if v.get("date", "") >= cutoff}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=0, sort_keys=True)


# ---------------------------------------------------------------------------
# 7. Email
# ---------------------------------------------------------------------------
def send_email(html_body, subject):
    """Send via SMTP (MAIL_USERNAME / MAIL_PASSWORD, e.g. a Gmail app password). Skips if unset."""
    username = os.environ.get("MAIL_USERNAME")
    password = os.environ.get("MAIL_PASSWORD")
    to_email = os.environ.get("TO_EMAIL") or username
    if not (username and password and to_email):
        print("Email not configured (MAIL_USERNAME / MAIL_PASSWORD / TO_EMAIL). Skipping email.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("FROM_EMAIL") or username
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(server, port, context=ssl.create_default_context()) as smtp:
        smtp.login(username, password)
        smtp.sendmail(msg["From"], [a.strip() for a in to_email.split(",")], msg.as_string())
    print(f"Email sent to {to_email}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Windows consoles default to cp949/cp1252, which cannot print many paper titles.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    parser = argparse.ArgumentParser(description="SciML arXiv digest (AI for CFD / neural PDE solvers)")
    parser.add_argument("--config", default=os.path.join(ROOT_DIR, "config_sciml.yaml"))
    parser.add_argument("--no-llm", action="store_true", help="only fetch + keyword prefilter (no API calls)")
    parser.add_argument("--no-email", action="store_true", help="do not send email")
    parser.add_argument("--limit", type=int, default=None, help="max number of papers to screen (for testing)")
    parser.add_argument("--no-history", action="store_true",
                        help="ignore and do not update the list of already-screened papers (for testing)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    date_str = datetime.date.today().isoformat()
    papers = fetch_papers(
        config["categories"],
        include_cross_lists=config.get("include_cross_lists", True),
        include_replacements=config.get("include_replacements", False),
    )
    print(f"Fetched {len(papers)} papers")
    if not papers:
        print("No new announcements today (arXiv does not announce on weekends/holidays). Exiting.")
        return

    history_path = os.path.join(ROOT_DIR, config.get("history_file", "history/seen_papers.json"))
    history = {} if args.no_history else load_history(history_path)
    fetched = len(papers)
    papers = [p for p in papers if p["id"] not in history]
    if fetched > len(papers):
        print(f"Skipped {fetched - len(papers)} papers already screened before (use --no-history to include them)")

    candidates = keyword_prefilter(papers, config.get("prefilter"))
    print(f"{len(candidates)} papers passed the keyword prefilter")
    if args.limit:
        candidates = candidates[:args.limit]

    if args.no_llm:
        for p in candidates:
            print(f"- [{p['id']}] {p['title']}  ({', '.join(p['categories'][:3])})")
        return

    llm_cfg = config.get("llm", {}) or {}
    provider = detect_provider(llm_cfg.get("provider", "auto"))
    screen_model = llm_cfg.get("screening_model") or DEFAULT_MODELS[provider][0]
    analysis_model = llm_cfg.get("analysis_model") or DEFAULT_MODELS[provider][1]
    print(f"Using provider={provider}, screening={screen_model}, analysis={analysis_model}")

    interest = config["interest"]
    language = config.get("language", "Korean")
    effort = llm_cfg.get("reasoning_effort")
    screen_client = LLMClient(provider, screen_model, reasoning_effort=effort)
    analysis_client = LLMClient(provider, analysis_model, reasoning_effort=effort)
    screen_papers(candidates, screen_client, interest, language, batch_size=llm_cfg.get("batch_size", 10))

    selected = sorted(
        (p for p in candidates if p["score"] >= config.get("threshold", 5)),
        key=lambda p: p["score"], reverse=True,
    )
    to_analyze = [p for p in selected if p["score"] >= config.get("analysis_threshold", 7)]
    to_analyze = to_analyze[:config.get("max_analyzed", 5)]
    rest = [p for p in selected if p not in to_analyze]
    briefs, extras = rest[:config.get("max_listed", 5)], rest[config.get("max_listed", 5):]
    print(f"{len(selected)} papers selected: {len(to_analyze)} deep analysis, {len(briefs)} one-line, {len(extras)} title-only")

    if to_analyze:
        max_chars = llm_cfg.get("max_fulltext_chars", 40000)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda p: analyze_paper(p, analysis_client, interest, language, max_chars), to_analyze))

    stats = {"fetched": fetched, "seen": fetched - len(papers), "prefiltered": len(candidates),
             "model": f"{screen_model} / {analysis_model}"}
    report = render_html(to_analyze, briefs, extras, stats, date_str)

    out_path = os.path.join(DIGEST_DIR, f"sciml_digest_{date_str}.html")
    for path in (out_path, os.path.join(ROOT_DIR, "digest.html")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
    with open(os.path.join(DATA_DIR, f"sciml_{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)
    print(f"Digest written to {out_path}")
    for name, c in (("screening", screen_client), ("analysis", analysis_client)):
        print(f"Token usage [{name}: {c.model}] input={c.usage_in:,} output={c.usage_out:,}")

    if not args.no_email and selected:
        send_email(report, f"SciML arXiv Digest {date_str} — {len(selected)}편")

    # Record screened papers only after the digest is out, so a failed run can simply be retried.
    if not args.no_history:
        for p in candidates:
            history[p["id"]] = {"date": date_str, "score": p.get("score", 0)}
        save_history(history_path, history)
        print(f"History updated: {len(history)} papers in {history_path}")


if __name__ == "__main__":
    main()
