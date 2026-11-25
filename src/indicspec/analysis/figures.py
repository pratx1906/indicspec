"""README / report figures, regenerated from results/tables (``make figures``).

Every figure is written twice, ``docs/assets/<name>-light.svg`` and ``-dark.svg``, so GitHub can
pick the theme with ``<picture>``. Colours, mark specs and ink roles follow the reference palette of
the data-viz method (categorical slots 1-3 validated in both modes; aqua is below 3:1 on the light
surface, so every chart that uses it is direct-labelled and has a table view in the README).

Static images cannot hover; the README places the numbers in a table next to each chart.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "docs" / "assets"
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]


LIGHT = Theme(
    "light",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    ("#2a78d6", "#eb6834", "#1baf7a"),
)
DARK = Theme(
    "dark",
    "#1a1a19",
    "#f0efec",
    "#c3c2b7",
    "#898781",
    "#2c2c2a",
    "#383835",
    ("#3987e5", "#d95926", "#199e70"),
)


def _csv(name: str) -> list[dict[str, str]]:
    with open(REPO / "results/tables" / name, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def _setup(theme: Theme, w: float, h: float):  # -> (fig, ax)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "path",  # text as outlines: identical rendering everywhere
            "font.family": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "svg.hashsalt": "indicspec",  # deterministic ids -> byte-stable SVGs
        }
    )
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.axis)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=theme.muted, length=0, labelsize=9)
    ax.xaxis.grid(True, color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax


def _hbar(ax, y: float, x0: float, x1: float, height: float, color: str, radius_px: float = 4):
    """Horizontal bar as ONE path: square at the baseline (x0), 4px rounded data end (x1)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    rx = radius_px * (xlim[1] - xlim[0]) / bbox.width
    ry = radius_px * abs(ylim[1] - ylim[0]) / bbox.height
    rx = min(rx, max(x1 - x0, 0.0) / 2)
    ry = min(ry, height / 2)
    ya, yb = y - height / 2, y + height / 2
    verts = [
        (x0, ya),
        (x1 - rx, ya),
        (x1, ya),
        (x1, ya + ry),
        (x1, yb - ry),
        (x1, yb),
        (x1 - rx, yb),
        (x0, yb),
        (x0, ya),
    ]
    codes = [
        MPath.MOVETO,
        MPath.LINETO,
        MPath.CURVE3,
        MPath.CURVE3,
        MPath.LINETO,
        MPath.CURVE3,
        MPath.CURVE3,
        MPath.LINETO,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, linewidth=0))


def _fmt(value: str, nd: int = 2) -> str:
    """Round half-up from the stored decimal string (matches README and claim ledger)."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(value).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def _title(fig, theme: Theme, title: str, subtitle: str) -> None:
    fig.text(
        0.012,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=theme.ink,
    )
    fig.text(0.012, 0.895, subtitle, ha="left", va="top", fontsize=9.5, color=theme.ink2)


def _save(fig, name: str, theme: Theme) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}-{theme.name}.svg"
    fig.savefig(out, format="svg", facecolor=theme.surface, metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def token_tax(theme: Theme) -> Path:
    rows = {
        r["lang"]: r
        for r in _csv("token_tax.csv")
        if r["tokenizer"] == "qwen3" and r["task"] == "translation"
    }
    order = ["en", "hi", "ta", "kn", "te", "ml"]
    fig, ax = _setup(theme, 8.0, 3.6)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.74, bottom=0.12)
    ax.set_xlim(0, 8.2)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks(range(len(order)), [LANG_NAMES[x] for x in order])
    ax.tick_params(axis="y", colors=theme.ink2, labelsize=10)
    ax.set_xticks([0, 2, 4, 6, 8], ["0×", "2×", "4×", "6×", "8×"])
    for i, lang in enumerate(order):
        v = float(rows[lang]["parity_vs_en_ratio_of_sums"])
        lo, hi = float(rows[lang]["parity_ci95_lo"]), float(rows[lang]["parity_ci95_hi"])
        _hbar(ax, i, 0, v, 0.42, theme.series[0])
        if lang != "en":
            ax.plot([lo, hi], [i, i], color=theme.ink2, linewidth=1, solid_capstyle="butt")
        ax.text(
            max(v, hi) + 0.12,
            i,
            f"{_fmt(rows[lang]['parity_vs_en_ratio_of_sums'])}×",
            va="center",
            ha="left",
            fontsize=9.5,
            color=theme.ink,
        )
    _title(
        fig,
        theme,
        "The token tax: tokens needed for the same content",
        "Qwen3 tokenizer, FLORES devtest, 220 parallel items per language; "
        "ratio vs English with 95% bootstrap CI",
    )
    return _save(fig, "token-tax", theme)


def fragmentation(theme: Theme) -> Path:
    rows = {r["lang"]: r for r in _csv("fragmentation_ref.csv") if r["task"] == "ALL"}
    order = ["en", "hi", "ta", "kn", "ml", "te"]
    fig, ax = _setup(theme, 8.0, 3.8)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.70, bottom=0.12)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks(range(len(order)), [LANG_NAMES[x] for x in order])
    ax.tick_params(axis="y", colors=theme.ink2, labelsize=10)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    labels = ["Character fragments", "Whole characters", "Words, ASCII, punctuation"]
    gap = 0.35  # ~2px surface gap at this width
    for i, lang in enumerate(order):
        r = rows[lang]
        frag = 100 * (
            float(r["share_BYTE_FRAGMENT_LEAD"])
            + float(r["share_BYTE_FRAGMENT_CONT"])
            + float(r["share_BYTE_INVALID"])
        )
        char = 100 * (float(r["share_CHAR"]) + float(r["share_MULTI_CHAR"]))
        other = 100 - frag - char
        x = 0.0
        for k, val in enumerate((frag, char, other)):
            if val <= 0.05:
                continue
            w = max(val - gap, 0.0)
            from matplotlib.patches import Rectangle

            ax.add_patch(Rectangle((x, i - 0.21), w, 0.42, linewidth=0, facecolor=theme.series[k]))
            if w >= 9:
                ax.text(
                    x + 1.2, i, f"{val:.0f}%", va="center", ha="left", fontsize=9, color="#ffffff"
                )
            x += val
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, edgecolor="none") for c in theme.series]
    leg = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.115, 0.83),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.6,
    )
    for t in leg.get_texts():
        t.set_color(theme.ink2)
    _title(
        fig,
        theme,
        "Most Dravidian tokens are pieces of a character",
        "Share of Qwen3 tokens by class, all benchmark reference text",
    )
    return _save(fig, "fragmentation", theme)


def ceiling(theme: Theme) -> Path:
    rows = [r for r in _csv("oracle_tau_bound.csv") if r["k"] == "7"]
    a3s = {r["lang"]: r["tau_oracle"] for r in rows if r["drafter"] == "A3"}
    a3 = {k: float(v) for k, v in a3s.items()}
    a4 = {r["lang"]: float(r["tau_oracle"]) for r in rows if r["drafter"] == "A4"}
    full = {r["lang"]: float(r["full_vocab_tau_oracle"]) for r in rows if r["drafter"] == "A3"}
    order = ["en", "hi", "ta", "te", "kn", "ml"]
    fig, ax = _setup(theme, 8.0, 5.2)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.78, bottom=0.09)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(len(order) - 0.45, -0.6)
    ax.set_yticks(range(len(order)), [LANG_NAMES[x] for x in order])
    ax.tick_params(axis="y", colors=theme.ink2, labelsize=10)
    ax.set_xticks([0, 1, 2, 4, 6, 8], ["0", "1", "2", "4", "6", "8"])
    ax.axvline(1, color=theme.axis, linewidth=1)
    h = 0.24
    for i, lang in enumerate(order):
        for k, (vals, off) in enumerate(((full, -h - 0.02), (a4, 0.0), (a3, h + 0.02))):
            v = vals[lang]
            _hbar(ax, i + off, 0, v, h, theme.series[k])
            if k == 2 or (k == 0 and i == 0):
                label = _fmt(a3s[lang]) if k == 2 else f"{v:.2f}"
                ax.text(
                    v + 0.1, i + off, label, va="center", ha="left", fontsize=9, color=theme.ink
                )
    from matplotlib.patches import Patch

    labels = ["Any full-vocabulary drafter", "Hindi/Gujarati EAGLE-3 head", "English EAGLE-3 head"]
    handles = [Patch(facecolor=c, edgecolor="none") for c in theme.series]
    leg = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.115, 0.865),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.6,
    )
    for t in leg.get_texts():
        t.set_color(theme.ink2)
    _title(
        fig,
        theme,
        "The coverage ceiling: best possible acceptance with 7 draft tokens",
        "Tokens per target step if the drafter were always right whenever the token is in "
        "its vocabulary. 1.0 = no speed-up",
    )
    return _save(fig, "ceiling", theme)


def _noise(run: str) -> list[dict[str, str]]:
    return _csv(f"{run}_noise.csv")


def _g(rows: list[dict[str, str]], arm: str, lang: str, conc: int, key: str) -> float | None:
    for r in rows:
        if r["arm"] == arm and r["language"] == lang and r["concurrency"] == str(conc):
            return float(r[key]) if r.get(key) else None
    return None


def pilot_speedup(theme: Theme) -> Path:
    """Pilot (exploratory): speed-up vs no speculation, one user, three drafters."""
    p4, p8 = _noise("pilot_vllm_4b"), _noise("pilot_vllm_8b")
    series = [
        ("n-gram copier (4B)", lambda lang: _g(p4, "A1", lang, 1, "speedup")),
        ("DFlash, full vocabulary (4B)", lambda lang: _g(p4, "A5", lang, 1, "speedup")),
        ("English EAGLE-3 head (8B)", lambda lang: _g(p8, "A3", lang, 1, "speedup")),
    ]
    order = ["en", "hi", "ta", "te", "kn", "ml"]
    fig, ax = _setup(theme, 8.0, 5.2)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.78, bottom=0.09)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(len(order) - 0.45, -0.6)
    ax.set_yticks(range(len(order)), [LANG_NAMES[x] for x in order])
    ax.tick_params(axis="y", colors=theme.ink2, labelsize=10)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3], ["0", "0.5×", "1×", "1.5×", "2×", "2.5×", "3×"])
    ax.axvline(1, color=theme.axis, linewidth=1)
    h = 0.24
    for i, lang in enumerate(order):
        for k, (_, fn) in enumerate(series):
            v = fn(lang)
            if v is None:
                continue
            y = i + (k - 1) * (h + 0.02)
            _hbar(ax, y, 0, v, h, theme.series[k])
            ax.text(v + 0.04, y, f"{v:.2f}×", va="center", ha="left", fontsize=8.5, color=theme.ink)
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, edgecolor="none") for c in theme.series]
    leg = fig.legend(
        handles,
        [n for n, _ in series],
        loc="upper left",
        bbox_to_anchor=(0.115, 0.865),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.2,
    )
    for t in leg.get_texts():
        t.set_color(theme.ink2)
    _title(
        fig,
        theme,
        "Pilot: speed-up vs no speculation, one user",
        "Exploratory · pilot split, 12 prompts × 3 repeats per language · vLLM 0.34 · 1× = no gain",
    )
    return _save(fig, "pilot-speedup", theme)


ORDINAL = {  # sequential blue ramp, ordinal-safe steps (palette reference)
    "light": ("#86b6ef", "#2a78d6", "#104281"),
    "dark": ("#184f95", "#3987e5", "#9ec5f4"),
}


def pilot_load(theme: Theme) -> Path:
    """Pilot (exploratory): DFlash speed-up at 1, 4, 16 concurrent users per language."""
    p4 = _noise("pilot_vllm_4b")
    order = ["en", "hi", "ta", "te", "kn", "ml"]
    conc = [1, 4, 16]
    cols = ORDINAL[theme.name]
    fig, ax = _setup(theme, 8.0, 5.2)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.78, bottom=0.09)
    ax.set_xlim(0, 2.8)
    ax.set_ylim(len(order) - 0.45, -0.6)
    ax.set_yticks(range(len(order)), [LANG_NAMES[x] for x in order])
    ax.tick_params(axis="y", colors=theme.ink2, labelsize=10)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5], ["0", "0.5×", "1×", "1.5×", "2×", "2.5×"])
    ax.axvline(1, color=theme.axis, linewidth=1)
    h = 0.24
    for i, lang in enumerate(order):
        for k, c in enumerate(conc):
            v = _g(p4, "A5", lang, c, "speedup")
            if v is None:
                continue
            y = i + (k - 1) * (h + 0.02)
            _hbar(ax, y, 0, v, h, cols[k])
            ax.text(v + 0.04, y, f"{v:.2f}×", va="center", ha="left", fontsize=8.5, color=theme.ink)
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, edgecolor="none") for c in cols]
    leg = fig.legend(
        handles,
        ["1 user", "4 users", "16 users"],
        loc="upper left",
        bbox_to_anchor=(0.115, 0.865),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.6,
    )
    for t in leg.get_texts():
        t.set_color(theme.ink2)
    _title(
        fig,
        theme,
        "Pilot: DFlash under load, by language",
        "Exploratory · Qwen3-4B · vs no speculation at the same load · below 1× = slower",
    )
    return _save(fig, "pilot-load", theme)


# --------------------------------------------------------------------------- banner

BANNER_TOKENS = [  # Qwen3 tokenization of తెలుగు (measured; see README "Findings")
    ("త", "E0 B0 A4", 1),
    ("ె", "E0 B1", 0),
    ("", "86", 0),
    ("ల", "E0 B0 B2", 1),
    ("ు", "E0 B1", 0),
    ("", "81", 0),
    ("గ", "E0 B0 97", 1),
    ("ు", "E0 B1", 0),
    ("", "81", 0),
]


def banner(theme: Theme) -> Path:
    """Hand-built SVG: 'Telugu' = 2 tokens vs 'తెలుగు' = 9 tokens (real Qwen3 output)."""
    frag, whole, eng = theme.series[0], theme.series[1], theme.series[2]
    w, h = 1200, 300
    font = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="IndicSpec: the word Telugu is 2 tokens '
        f'in English and 9 tokens in Telugu script for Qwen3">',
        f'<rect width="{w}" height="{h}" rx="14" fill="{theme.surface}"/>',
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="14" fill="none" '
        f'stroke="{theme.grid}"/>',
        f'<text x="48" y="78" font-family="{font}" font-size="44" font-weight="700" '
        f'fill="{theme.ink}">IndicSpec</text>',
        f'<text x="48" y="112" font-family="{font}" font-size="18" fill="{theme.ink2}">'
        "Speculative decoding for Indic LLM serving · vLLM · SGLang · one 24 GB laptop GPU"
        "</text>",
    ]
    # English row
    y_en, y_te = 170, 238
    parts.append(
        f'<text x="48" y="{y_en + 6}" font-family="{font}" font-size="16" '
        f'fill="{theme.ink2}">“Telugu”</text>'
    )
    x = 200
    for piece in ("Tel", "ugu"):
        bw = 64
        parts.append(
            f'<rect x="{x}" y="{y_en - 18}" width="{bw}" height="34" rx="4" fill="{eng}"/>'
        )
        parts.append(
            f'<text x="{x + bw / 2}" y="{y_en + 6}" text-anchor="middle" '
            f'font-family="{mono}" font-size="16" fill="#ffffff">{piece}</text>'
        )
        x += bw + 4
    parts.append(
        f'<text x="{x + 14}" y="{y_en + 6}" font-family="{font}" font-size="16" '
        f'fill="{theme.ink}">2 tokens</text>'
    )
    # Telugu row
    parts.append(
        f'<text x="48" y="{y_te + 6}" font-family="{font}" font-size="20" '
        f'fill="{theme.ink2}">తెలుగు</text>'
    )
    x = 200
    for _glyph, hexes, is_whole in BANNER_TOKENS:
        bw = 84 if " " in hexes and len(hexes) > 5 else (62 if " " in hexes else 40)
        fill = whole if is_whole else frag
        parts.append(
            f'<rect x="{x}" y="{y_te - 18}" width="{bw}" height="34" rx="4" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{x + bw / 2}" y="{y_te + 5}" text-anchor="middle" '
            f'font-family="{mono}" font-size="12.5" fill="#ffffff">{hexes}</text>'
        )
        x += bw + 4
    parts.append(
        f'<text x="{x + 14}" y="{y_te + 6}" font-family="{font}" font-size="16" '
        f'fill="{theme.ink}">9 tokens</text>'
    )
    # legend
    lx = 200
    for label, col in (
        ("whole character", whole),
        ("fragment of a character", frag),
        ("English subword", eng),
    ):
        parts.append(f'<rect x="{lx}" y="272" width="12" height="12" rx="2" fill="{col}"/>')
        parts.append(
            f'<text x="{lx + 18}" y="283" font-family="{font}" font-size="13" '
            f'fill="{theme.ink2}">{label}</text>'
        )
        lx += 30 + 8 * len(label)
    parts.append(
        f'<text x="{w - 48}" y="283" text-anchor="end" font-family="{font}" '
        f'font-size="13" fill="{theme.muted}">Qwen3 tokenizer · UTF-8 bytes shown in hex'
        "</text>"
    )
    parts.append("</svg>")
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"banner-{theme.name}.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def main() -> int:
    for theme in (LIGHT, DARK):
        for fn in (banner, token_tax, fragmentation, ceiling, pilot_speedup, pilot_load):
            print(fn(theme).relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
