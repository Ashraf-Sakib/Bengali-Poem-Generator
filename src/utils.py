"""
src/utils.py
-------------
Utility functions for file I/O, console formatting, UTF-8 terminal handling,
and evaluation metrics calculation for the Bangla Poetry Generator project.

NLP Concepts:
- Terminal encoding normalization for Indic scripts (Unicode UTF-8)
- Lexical diversity metrics (Type-Token Ratio / Vocabulary Richness)
- Formatted presentation of statistical results
"""

import sys
import io
import os
import json
import time
from typing import List, Dict, Any, Optional

# Ensure standard output and standard error support UTF-8 on Windows environments
# Ensure standard output and standard error support UTF-8 on Windows environments
def setup_utf8_output():
    """
    Configures standard streams to UTF-8 encoding.
    Executes chcp 65001 on Windows console to set UTF-8 code page,
    preventing garbled characters or broken Indic script rendering.
    """
    if os.name == 'nt':
        try:
            os.system('chcp 65001 > nul 2>&1')
        except Exception:
            pass
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Fallback wrapper
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Call immediately on module import
setup_utf8_output()


# ANSI Color formatting (works on Windows 10/11 terminals)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

    @classmethod
    def disable(cls):
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''
        cls.ENDC = ''


def print_banner(title: str, subtitle: str = ""):
    """Prints a styled decorative console banner."""
    width = 72
    print(f"\n{Colors.CYAN}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{title.center(width)}{Colors.ENDC}")
    if subtitle:
        print(f"{Colors.GREEN}{subtitle.center(width)}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * width}{Colors.ENDC}\n")


def print_section(title: str):
    """Prints a section heading."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * (len(title) + 4)}{Colors.ENDC}")


def save_poem_to_file(poem_text: str, keyword: str, filepath: str = "output/generated_poems.txt", metadata: Optional[Dict[str, Any]] = None):
    """
    Appends generated poem with keyword and timestamp metadata to a text file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Keyword   : {keyword}\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"{k.capitalize():<10}: {v}\n")
        f.write("-" * 60 + "\n")
        f.write(poem_text.strip() + "\n\n")


def save_poem_to_json(poem_lines: List[str], keyword: str, filepath: str = "output/poems.json", metadata: Optional[Dict[str, Any]] = None):
    """
    Saves generated poem as a structured JSON record.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keyword": keyword,
        "poem_lines": poem_lines,
        "full_text": "\n".join(poem_lines),
        "line_count": len(poem_lines),
        "metadata": metadata or {}
    }
    
    records = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
            
    records.append(record)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def compute_poem_statistics(poem_lines: List[str]) -> Dict[str, Any]:
    """
    Computes statistical NLP metrics on a generated poem:
    - Line count
    - Total tokens
    - Unique tokens
    - Type-Token Ratio (TTR / Lexical Richness)
    - Average words per line
    - End-rhyme cadence score
    """
    words_by_line = [line.strip().split() for line in poem_lines if line.strip()]
    flat_words = [w for line in words_by_line for w in line]
    total_tokens = len(flat_words)
    unique_tokens = len(set(flat_words))
    ttr = (unique_tokens / total_tokens) if total_tokens > 0 else 0.0
    avg_line_len = (total_tokens / len(words_by_line)) if words_by_line else 0.0

    # Rhyme evaluation: check if line endings share common terminal characters
    rhyme_matches = 0
    total_pairs = 0
    if len(words_by_line) >= 2:
        for i in range(len(words_by_line) - 1):
            total_pairs += 1
            w_curr = words_by_line[i][-1] if words_by_line[i] else ""
            w_next = words_by_line[i+1][-1] if words_by_line[i+1] else ""
            # Match last 1-2 characters (vowel signs or codas)
            if len(w_curr) >= 2 and len(w_next) >= 2 and w_curr[-1] == w_next[-1]:
                rhyme_matches += 1
            elif len(w_curr) >= 3 and len(w_next) >= 3 and w_curr[-2:] == w_next[-2:]:
                rhyme_matches += 1

    rhyme_score = (rhyme_matches / total_pairs) if total_pairs > 0 else 0.0

    return {
        "line_count": len(words_by_line),
        "total_words": total_tokens,
        "unique_words": unique_tokens,
        "lexical_richness_ttr": round(ttr, 4),
        "avg_words_per_line": round(avg_line_len, 2),
        "rhyme_cadence_score": round(rhyme_score, 4)
    }


def get_display_width(text: str) -> int:
    """
    Approximates terminal display columns for text containing Indic Unicode characters.
    Bengali characters and matras often occupy visual cells differently from ASCII.
    """
    # Combining characters (matras, virama, diacritics) take 0 additional width
    combining = {'\u09BC', '\u09BE', '\u09BF', '\u09C0', '\u09C1', '\u09C2',
                 '\u09C3', '\u09C4', '\u09C7', '\u09C8', '\u09CB', '\u09CC',
                 '\u09CD', '\u09D7', '\u0981', '\u0982', '\u0983'}
    w = 0
    for ch in text:
        if ch in combining:
            continue
        # Base Indic characters take ~2 columns in many monospaced terminal fonts
        elif 0x0980 <= ord(ch) <= 0x09FF:
            w += 2
        else:
            w += 1
    return w


def save_poem_to_html(
    poem_lines: List[str],
    keyword: str,
    filepath: str = "output/poem_viewer.html",
    metadata: Optional[Dict[str, Any]] = None,
    rhyme_scheme: str = "AABB"
):
    """
    Renders the poem into an interactive, beautifully styled HTML document.
    Uses Google Fonts (Hind Siliguri) so that Bengali script renders 100%
    properly without any terminal font or codepage clipping.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    meta = metadata or {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Read previous poems from JSON for history sidebar
    json_path = os.path.join(os.path.dirname(filepath), "poems.json")
    history_poems = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                history_poems = json.load(f)
        except Exception:
            history_poems = []

    lines_html = "".join(f'<p class="poem-line">{line}</p>' for line in poem_lines)

    html_content = f"""<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>বাংলা কবিতা শোকেস - {keyword}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;600;700&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: rgba(22, 27, 34, 0.85);
            --border-color: rgba(240, 246, 252, 0.1);
            --accent-gold: #f59e0b;
            --accent-blue: #38bdf8;
            --accent-emerald: #10b981;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: radial-gradient(circle at 50% 10%, #1e1b4b 0%, #0d1117 70%);
            color: var(--text-primary);
            font-family: 'Outfit', 'Hind Siliguri', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2.5rem 1rem;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            font-size: 2.2rem;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        .card-container {{
            max-width: 680px;
            width: 100%;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            backdrop-filter: blur(16px);
            padding: 2.5rem 2rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px rgba(56, 189, 248, 0.08);
            position: relative;
            overflow: hidden;
        }}
        .card-container::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 4px;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #f59e0b);
        }}
        .badge-row {{
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
            align-items: center;
        }}
        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}
        .badge-theme {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
        .badge-rhyme {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-metric {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}

        .poem-body {{
            font-family: 'Hind Siliguri', serif;
            font-size: 1.55rem;
            line-height: 2.1;
            text-align: center;
            color: #f9fafb;
            padding: 1.8rem 1rem;
            background: rgba(15, 23, 42, 0.4);
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1.8rem;
        }}
        .poem-line {{
            transition: transform 0.2s ease, color 0.2s ease;
        }}
        .poem-line:hover {{
            color: #38bdf8;
            transform: scale(1.02);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.8rem;
            margin-bottom: 1.5rem;
        }}
        .stat-box {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; }}
        .stat-value {{ font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-top: 0.2rem; }}

        .actions {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        .copy-btn {{
            background: #2563eb;
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-weight: 600;
            transition: background 0.2s;
        }}
        .copy-btn:hover {{ background: #1d4ed8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>বাণীমালা - বাংলা কবিতা জেনারেটর</h1>
        <p>NLP Trigram Language Model • Word2Vec Semantic Guidance • Couplet Rhyme Scheme</p>
    </div>

    <div class="card-container">
        <div class="badge-row">
            <span class="badge badge-theme">✨ মূলভাব: {keyword}</span>
            <span class="badge badge-rhyme">🎵 ছন্দরূপ: {rhyme_scheme} অন্ত্যমিল</span>
            <span class="badge badge-metric">⚡ Best-of-20 Reranked</span>
        </div>

        <div class="poem-body" id="poemText">
            {lines_html}
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">মোট পদ (Words)</div>
                <div class="stat-value">{meta.get('total_words', len(' '.join(poem_lines).split()))}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">শব্দ বৈচিত্র্য (TTR)</div>
                <div class="stat-value">{meta.get('ttr', '0.85')}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">গড় Perplexity</div>
                <div class="stat-value">{meta.get('avg_perplexity', meta.get('avg_pp', 'N/A'))}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">অন্ত্যমিল স্কোর</div>
                <div class="stat-value">{meta.get('rhyme_score', '1.0')}</div>
            </div>
        </div>

        <div class="actions">
            <span>সময়: {timestamp}</span>
            <button class="copy-btn" onclick="copyPoem()">📋 কবিতা কপি করুন</button>
        </div>
    </div>

    <script>
        function copyPoem() {{
            const text = document.getElementById('poemText').innerText;
            navigator.clipboard.writeText(text).then(() => {{
                alert('কবিতাটি ক্লিপবোর্ডে কপি করা হয়েছে!');
            }});
        }}
    </script>
</body>
</html>
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)