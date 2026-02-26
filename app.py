# app.py
# Streamlit Cloud–ready: Screaming Frog "All Inlinks" Analyzer (chunk-safe)
# Focus Page Priority Report (focus pages are DESTINATION targets only)
# Changes:
# - Removed: "🧠 Optional: template CTA dominance filter (Top Anchor %)" (UI + logic + stats)
# - Added: Source page path exclusion filter (e.g. exclude sources containing /page/)

import re
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Set, Optional, Tuple, List
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st


# =========================
# Page / App Setup
# =========================
st.set_page_config(page_title="All Inlinks Analyzer (Chunk-safe)", layout="wide")
st.title("🔗 All Inlinks Analyzer (Chunk-safe for big files)")
st.caption(
    "Upload Screaming Frog All Inlinks (.csv, .csv.gz, or .zip), apply filters, and get Top Destinations + downloads."
)

uploaded_file = st.file_uploader("Upload All Inlinks (.csv, .csv.gz, or .zip)", type=["csv", "gz", "zip"])


# =========================
# Config / Data Structures
# =========================
@dataclass
class AppConfig:
    chunksize: int
    sample_rows: int
    max_distinct_anchors_per_destination: int

    # Output settings
    write_filtered_csv: bool
    max_filtered_rows: int  # 0 = unlimited

    # Row-level filters
    remove_self: bool
    exclude_params: bool

    # External destination filtering
    exclude_external_destinations: bool
    allowed_destination_domains: Set[str]
    keep_relative_destinations_as_internal: bool

    type_keep: List[str]
    follow_keep: List[str]
    status_keep: List[str]
    link_position_keep: List[str]

    rx_link_path_excl: Optional[re.Pattern]
    rx_anchor_excl: Optional[re.Pattern]

    # NEW: Source page path exclusion (row-level)
    rx_source_path_excl: Optional[re.Pattern]


@dataclass
class Aggregates:
    total_inlinks: Dict[str, int]
    unique_sources: Dict[str, Set[str]]

    anchor_total: Dict[str, int]                 # dest -> total non-empty anchor occurrences
    anchor_counts: Dict[str, Dict[str, int]]     # dest -> {anchor_text -> count} (capped)

    filtered_out_path: Optional[str]
    filtered_rows_written: int


@dataclass
class FilterStats:
    total_rows_read: int
    total_rows_kept: int

    removed_self: int
    removed_params: int
    removed_external_destination: int
    removed_type: int
    removed_follow: int
    removed_status: int
    removed_link_position: int
    removed_link_path: int
    removed_anchor: int

    # NEW
    removed_source_path: int


# =========================
# Helpers: normalization & patterns
# =========================
def normalize_url_for_compare(u: str) -> str:
    """Normalize for Source==Destination compare: lowercase scheme/host, trim trailing slash, drop fragment."""
    if u is None or pd.isna(u):
        return ""
    u = str(u).strip()
    if not u:
        return ""
    parts = urlsplit(u)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or ""
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))  # drop fragment


def compile_contains_patterns(lines: str) -> Optional[re.Pattern]:
    """One token per line, case-insensitive 'contains' match."""
    pats = []
    for line in (lines or "").splitlines():
        line = line.strip()
        if line:
            pats.append(re.escape(line))
    if not pats:
        return None
    return re.compile("|".join(pats), flags=re.IGNORECASE)


def uniq_values(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    vals = df[col].astype("string").fillna("").unique().tolist()
    return sorted(vals, key=lambda x: (x == "" or x is None, str(x).lower()))


def normalize_host(h: str) -> str:
    h = (h or "").strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h


def is_internal_destination(dest: str, allowed_domains: Set[str], keep_relative: bool = True) -> bool:
    """
    True if:
    - Destination is relative (no scheme/netloc) and keep_relative=True
    - OR destination host equals / is a subdomain of an allowed domain
    """
    if dest is None or pd.isna(dest):
        return False
    dest = str(dest).strip()
    if not dest:
        return False

    parts = urlsplit(dest)

    # Relative URLs
    if parts.scheme == "" and parts.netloc == "":
        return keep_relative

    host = normalize_host(parts.netloc)
    if not host:
        return False

    for ad in allowed_domains:
        ad = normalize_host(ad)
        if not ad:
            continue
        if host == ad or host.endswith("." + ad):
            return True

    return False


def source_path_only(src: str) -> str:
    """
    Returns just the path portion of the Source.
    - If absolute URL: returns parts.path
    - If relative/path-only: returns the string path as-is (via urlsplit)
    """
    if src is None or pd.isna(src):
        return ""
    s = str(src).strip()
    if not s:
        return ""
    parts = urlsplit(s)
    return parts.path or ""


def focus_key(url: str) -> str:
    """
    Create a join key that matches focus URLs to Destination URLs robustly.
    - If url is absolute: host+path (no query/fragment), lowercased host, trimmed trailing slash in path.
    - If url is relative/path-only: path only (leading slash enforced), trimmed trailing slash.
    """
    if url is None or pd.isna(url):
        return ""
    s = str(url).strip()
    if not s:
        return ""

    parts = urlsplit(s)

    # Relative (no scheme/netloc): treat as path
    if parts.scheme == "" and parts.netloc == "":
        p = parts.path if parts.path else s
        p = p.strip()
        if not p.startswith("/"):
            p = "/" + p
        if p != "/" and p.endswith("/"):
            p = p.rstrip("/")
        return p

    host = normalize_host(parts.netloc)
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return f"{host}{path}"


def destination_join_key(destination_url: str) -> str:
    """
    Join key for Destination URLs:
    - If absolute: host+path
    - If relative: path
    """
    return focus_key(destination_url)


# =========================
# IO: materialize upload + sample + chunk iterator
# =========================
def materialize_to_path(uploaded) -> str:
    name = (uploaded.name or "").lower()

    if name.endswith(".zip"):
        uploaded.seek(0)
        with zipfile.ZipFile(uploaded) as z:
            members = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ValueError("No .csv found inside the uploaded .zip.")
            csv_name = members[0]
            with z.open(csv_name) as f_in:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                with open(tmp.name, "wb") as f_out:
                    f_out.write(f_in.read())
                return tmp.name

    suffix = name[name.rfind("."):] if "." in name else ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f_out:
        f_out.write(uploaded.getbuffer())
    return tmp.name


def read_sample(path: str, nrows: int) -> pd.DataFrame:
    kwargs = dict(nrows=int(nrows), low_memory=False, dtype="string", compression="infer")
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "cp1252"
        return pd.read_csv(path, **kwargs)


def chunk_iterator(path: str, chunksize: int):
    kwargs = dict(chunksize=int(chunksize), low_memory=False, dtype="string", compression="infer")
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "cp1252"
        return pd.read_csv(path, **kwargs)


# =========================
# Filtering: apply row-level filters with per-step stats
# =========================
def apply_row_filters_with_stats(chunk: pd.DataFrame, cfg: AppConfig, stats: FilterStats) -> pd.DataFrame:
    if "Source" not in chunk.columns or "Destination" not in chunk.columns:
        return chunk.iloc[0:0].copy()

    chunk["Source"] = chunk["Source"].astype("string").fillna("")
    chunk["Destination"] = chunk["Destination"].astype("string").fillna("")

    # 1) self-ref
    if cfg.remove_self:
        before = len(chunk)
        src_norm = chunk["Source"].map(normalize_url_for_compare)
        dst_norm = chunk["Destination"].map(normalize_url_for_compare)
        chunk = chunk[src_norm != dst_norm]
        stats.removed_self += (before - len(chunk))

    # 2) query params (Destination)
    if cfg.exclude_params:
        before = len(chunk)
        chunk = chunk[~chunk["Destination"].str.contains(r"\?", regex=True, na=False)]
        stats.removed_params += (before - len(chunk))

    # 3) external destination filter
    if cfg.exclude_external_destinations and cfg.allowed_destination_domains:
        before = len(chunk)
        dests = chunk["Destination"].astype("string").fillna("")
        mask = dests.map(
            lambda d: is_internal_destination(
                d,
                cfg.allowed_destination_domains,
                cfg.keep_relative_destinations_as_internal
            )
        )
        chunk = chunk[mask]
        stats.removed_external_destination += (before - len(chunk))

    # 4) NEW: exclude by Source PATH patterns (e.g. /page/)
    if cfg.rx_source_path_excl is not None:
        before = len(chunk)
        src_paths = chunk["Source"].map(source_path_only).astype("string").fillna("")
        chunk = chunk[~src_paths.str.contains(cfg.rx_source_path_excl, na=False)]
        stats.removed_source_path += (before - len(chunk))

    # 5) include-only filters
    if cfg.type_keep and "Type" in chunk.columns:
        before = len(chunk)
        chunk = chunk[chunk["Type"].astype("string").fillna("").isin(cfg.type_keep)]
        stats.removed_type += (before - len(chunk))

    if cfg.follow_keep and "Follow" in chunk.columns:
        before = len(chunk)
        chunk = chunk[chunk["Follow"].astype("string").fillna("").isin(cfg.follow_keep)]
        stats.removed_follow += (before - len(chunk))

    if cfg.status_keep and "Status Code" in chunk.columns:
        before = len(chunk)
        chunk = chunk[chunk["Status Code"].astype("string").fillna("").isin(cfg.status_keep)]
        stats.removed_status += (before - len(chunk))

    if cfg.link_position_keep and "Link Position" in chunk.columns:
        before = len(chunk)
        chunk = chunk[chunk["Link Position"].astype("string").fillna("").isin(cfg.link_position_keep)]
        stats.removed_link_position += (before - len(chunk))

    # 6) link path exclusion
    if cfg.rx_link_path_excl is not None and "Link Path" in chunk.columns:
        before = len(chunk)
        lp = chunk["Link Path"].astype("string").fillna("")
        chunk = chunk[~lp.str.contains(cfg.rx_link_path_excl, na=False)]
        stats.removed_link_path += (before - len(chunk))

    # 7) anchor exclusion
    if cfg.rx_anchor_excl is not None and "Anchor" in chunk.columns:
        before = len(chunk)
        anc = chunk["Anchor"].astype("string").fillna("")
        chunk = chunk[~anc.str.contains(cfg.rx_anchor_excl, na=False)]
        stats.removed_anchor += (before - len(chunk))

    return chunk


# =========================
# Aggregation
# =========================
def update_aggregates_from_chunk(chunk: pd.DataFrame, cfg: AppConfig, agg: Aggregates) -> None:
    dests = chunk["Destination"].astype("string").fillna("").tolist()
    srcs = chunk["Source"].astype("string").fillna("").tolist()

    anchors = None
    if "Anchor" in chunk.columns:
        anchors = chunk["Anchor"].astype("string").fillna("").tolist()

    for i, d in enumerate(dests):
        if not d:
            continue
        s = srcs[i] if i < len(srcs) else ""

        agg.total_inlinks[d] = agg.total_inlinks.get(d, 0) + 1
        agg.unique_sources.setdefault(d, set()).add(s)

        if anchors is None:
            continue

        a = str(anchors[i]).strip() if i < len(anchors) else ""
        if not a:
            continue

        agg.anchor_total[d] = agg.anchor_total.get(d, 0) + 1
        dest_map = agg.anchor_counts.get(d)
        if dest_map is None:
            dest_map = {}
            agg.anchor_counts[d] = dest_map

        if a in dest_map:
            dest_map[a] += 1
        else:
            if len(dest_map) < int(cfg.max_distinct_anchors_per_destination):
                dest_map[a] = 1


# =========================
# Output table building
# =========================
def compute_top_anchor_fields(destinations: List[str], agg: Aggregates) -> Tuple[List[str], List[float]]:
    top_anchor = []
    top_anchor_pct = []

    for d in destinations:
        total_anchor_occ = agg.anchor_total.get(d, 0)
        if total_anchor_occ == 0:
            top_anchor.append("")
            top_anchor_pct.append(0.0)
            continue

        amap = agg.anchor_counts.get(d, {})
        if not amap:
            top_anchor.append("")
            top_anchor_pct.append(0.0)
            continue

        best_a, best_c = max(amap.items(), key=lambda kv: kv[1])
        pct = (best_c / total_anchor_occ) * 100.0
        top_anchor.append(best_a)
        top_anchor_pct.append(round(pct, 2))

    return top_anchor, top_anchor_pct


# =========================
# Optional: stream-write filtered CSV
# =========================
def maybe_write_filtered_chunk(chunk: pd.DataFrame, cfg: AppConfig, agg: Aggregates, wrote_header: bool) -> bool:
    if not cfg.write_filtered_csv or agg.filtered_out_path is None:
        return wrote_header

    if cfg.max_filtered_rows != 0 and agg.filtered_rows_written >= int(cfg.max_filtered_rows):
        return wrote_header

    remaining = None if cfg.max_filtered_rows == 0 else int(cfg.max_filtered_rows) - agg.filtered_rows_written
    to_write = chunk if remaining is None else chunk.head(max(0, remaining))

    mode = "w" if not wrote_header else "a"
    to_write.to_csv(agg.filtered_out_path, mode=mode, header=not wrote_header, index=False)

    agg.filtered_rows_written += len(to_write)
    return True


# =========================
# UI: Build config
# =========================
def build_config(sample_df: pd.DataFrame) -> AppConfig:
    st.sidebar.header("⚙️ Performance")
    chunksize = st.sidebar.number_input("Chunk size (rows)", 50_000, 1_000_000, 200_000, 50_000)
    sample_rows = st.sidebar.number_input("Sample rows (detect filter values)", 5_000, 200_000, 50_000, 5_000)

    max_distinct = st.sidebar.number_input(
        "Max distinct anchors tracked per destination",
        min_value=25,
        max_value=2000,
        value=300,
        step=25,
    )

    st.sidebar.header("⬇️ Output files")
    write_filtered_csv = st.sidebar.checkbox("Create downloadable filtered CSV (can be large)", value=False)
    max_filtered_rows = st.sidebar.number_input(
        "Max filtered rows to write (0 = no limit)",
        min_value=0,
        max_value=50_000_000,
        value=0,
        step=100_000,
    )

    type_options = uniq_values(sample_df, "Type")
    follow_options = uniq_values(sample_df, "Follow")
    status_options = uniq_values(sample_df, "Status Code")
    linkpos_options = uniq_values(sample_df, "Link Position")

    st.subheader("🎛️ Filters")
    preset = st.button("✨ Preset: Content + Follow + 200 + Hyperlink")

    c1, c2, c3 = st.columns(3)
    with c1:
        remove_self = st.checkbox("Remove self-referring (Source == Destination)", value=True)
    with c2:
        exclude_params = st.checkbox("Exclude Destination with '?' (query params)", value=True)
    with c3:
        st.caption("Tip: leave include-only filters empty to keep all values.")

    st.markdown("### 🌐 Destination domain filter")
    exclude_external_destinations = st.checkbox(
        "Exclude external Destination links (keep only allowed domains)",
        value=True,
        help="Keeps Destination URLs that match allowed domain(s). Removes other domains.",
    )
    allowed_domains_text = st.text_input(
        "Allowed destination domain(s) (comma-separated)",
        value="",
        help="Example: so.energy, blog.so.energy",
    )
    keep_relative_destinations = st.checkbox(
        "Treat relative Destination URLs as internal",
        value=True,
        help="Keeps destinations like /blog/post even if they have no domain.",
    )
    allowed_domains = set(d.strip() for d in (allowed_domains_text or "").split(",") if d.strip())

    if exclude_external_destinations and not allowed_domains:
        st.warning("External destination exclusion is ON, but no allowed domains were provided. The filter will be skipped.")

    default_type = ["Hyperlink"] if "Hyperlink" in type_options else []
    default_follow = ["Follow"] if "Follow" in follow_options else []
    default_status = ["200"] if "200" in status_options else (["200 OK"] if "200 OK" in status_options else [])
    default_linkpos = ["Content"] if "Content" in linkpos_options else []

    if preset:
        type_keep = default_type
        follow_keep = default_follow
        status_keep = default_status
        link_position_keep = default_linkpos
    else:
        type_keep = st.multiselect("Keep Type values (empty = keep all)", type_options, default=default_type)
        follow_keep = st.multiselect("Keep Follow values (empty = keep all)", follow_options, default=default_follow)
        status_keep = st.multiselect("Keep Status Code values (empty = keep all)", status_options, default=default_status)
        link_position_keep = st.multiselect("Keep Link Position values (empty = keep all)", linkpos_options, default=default_linkpos)

    # NEW: Exclude by SOURCE page path patterns
    st.markdown("### 🚫 Exclude by Source page path (e.g. pagination)")
    enable_source_path_exclusions = st.checkbox("Enable Source path exclusions", value=True)

    default_source_path_excl = "\n".join(["/page/"])
    source_path_excl_text = st.text_area(
        "Source path patterns to exclude (one per line, 'contains' match)",
        value=default_source_path_excl,
        height=80,
        disabled=not enable_source_path_exclusions,
        help="Matches against Source PATH only. Example: /page/ excludes sources like /blog/page/2/.",
    )
    rx_source_path_excl = compile_contains_patterns(source_path_excl_text) if enable_source_path_exclusions else None

    st.markdown("### Exclude by Link Path patterns (breadcrumbs/nav etc.)")
    exclude_by_link_path = st.checkbox("Enable Link Path exclusions", value=True)
    default_lp = "\n".join(["breadcrumb", "/ol/li", "aria-label=\"breadcrumb\"", "aria-label='breadcrumb'"])
    lp_text = st.text_area("Link Path patterns to exclude (one per line)", value=default_lp, height=100, disabled=not exclude_by_link_path)
    rx_link_path_excl = compile_contains_patterns(lp_text) if exclude_by_link_path else None

    st.markdown("### 🚫 Exclude by Anchor text (template CTAs etc.)")
    enable_anchor_exclusions = st.checkbox("Enable Anchor exclusions", value=True)
    default_anchor_excl = "\n".join(["discover more", "read more", "learn more", "find out more", "view more", "see more"])
    anchor_excl_text = st.text_area(
        "Anchor phrases to exclude (one per line, 'contains' match)",
        value=default_anchor_excl,
        height=120,
        disabled=not enable_anchor_exclusions,
    )
    rx_anchor_excl = compile_contains_patterns(anchor_excl_text) if enable_anchor_exclusions else None

    return AppConfig(
        chunksize=int(chunksize),
        sample_rows=int(sample_rows),
        max_distinct_anchors_per_destination=int(max_distinct),

        write_filtered_csv=write_filtered_csv,
        max_filtered_rows=int(max_filtered_rows),

        remove_self=remove_self,
        exclude_params=exclude_params,

        exclude_external_destinations=exclude_external_destinations,
        allowed_destination_domains=allowed_domains,
        keep_relative_destinations_as_internal=keep_relative_destinations,

        type_keep=type_keep,
        follow_keep=follow_keep,
        status_keep=status_keep,
        link_position_keep=link_position_keep,

        rx_link_path_excl=rx_link_path_excl,
        rx_anchor_excl=rx_anchor_excl,

        rx_source_path_excl=rx_source_path_excl,
    )


# =========================
# Focus Pages (UI + parsing)
# =========================
def load_focus_pages_csv(uploaded) -> Optional[pd.DataFrame]:
    """Reads focus pages CSV with encoding fallback."""
    if uploaded is None:
        return None
    kwargs = dict(low_memory=False, dtype="string")
    try:
        df = pd.read_csv(uploaded, **kwargs)
    except UnicodeDecodeError:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="cp1252", **kwargs)
    return df


def build_focus_report(dest_summary: pd.DataFrame, focus_df: pd.DataFrame, url_col: str, metric_col: str) -> pd.DataFrame:
    """
    dest_summary: Destination summary table (post-filters)
    focus_df: user focus pages table
    Returns focus report merged + priority scores.
    """
    f = focus_df.copy()

    # Build focus keys
    f["__focus_url"] = f[url_col].astype("string").fillna("")
    f["__focus_key"] = f["__focus_url"].map(focus_key)

    # Metric numeric (coerce)
    metric_raw = f[metric_col].astype("string").fillna("")
    metric_num = pd.to_numeric(metric_raw.str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0.0)
    f["__metric"] = metric_num

    # Destination keys
    d = dest_summary.copy()
    d["__dest_key"] = d["Destination"].astype("string").fillna("").map(destination_join_key)

    merged = d.merge(
        f[["__focus_key", "__focus_url", "__metric"]],
        left_on="__dest_key",
        right_on="__focus_key",
        how="inner",
    )

    merged.rename(columns={"__metric": "Focus Metric"}, inplace=True)

    # Deficit score (simple, robust)
    merged["Deficit Score"] = 1.0 / (merged["Unique_Source_Pages"].astype(float) + 1.0)

    # Value score normalized within focus set
    max_metric = float(merged["Focus Metric"].max()) if len(merged) else 0.0
    if max_metric > 0:
        merged["Value Score"] = merged["Focus Metric"].astype(float) / max_metric
    else:
        merged["Value Score"] = 0.0

    # Priority
    merged["Priority Score"] = merged["Deficit Score"] * merged["Value Score"]

    out_cols = [
        "Destination",
        "Focus Metric",
        "Value Score",
        "Unique_Source_Pages",
        "Total_Inlinks",
        "Deficit Score",
        "Priority Score",
        "Top Anchor",
        "Top Anchor %",
    ]
    out = merged[out_cols].copy()

    out = out.sort_values(["Priority Score", "Focus Metric"], ascending=False).reset_index(drop=True)
    return out


# =========================
# Main flow
# =========================
if uploaded_file is None:
    st.info("Upload an All Inlinks file to begin.")
    st.stop()

try:
    path = materialize_to_path(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

sample_df_initial = read_sample(path, 50_000)
missing = {"Source", "Destination"} - set(sample_df_initial.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

cfg = build_config(sample_df_initial)

st.divider()
st.subheader("🎯 Focus Page Priority Report (optional)")
focus_upload = st.file_uploader(
    "Upload Focus Pages CSV (URL + Metric). Focus pages are treated as DESTINATION targets only.",
    type=["csv"],
    key="focus_csv",
)

focus_df = load_focus_pages_csv(focus_upload)

focus_url_col = None
focus_metric_col = None

if focus_df is not None and not focus_df.empty:
    st.success(f"Focus CSV loaded: {focus_df.shape[0]:,} rows × {focus_df.shape[1]:,} cols")
    with st.expander("Preview focus CSV"):
        st.dataframe(focus_df.head(50), use_container_width=True)

    cols = list(focus_df.columns)
    c1, c2 = st.columns(2)
    with c1:
        focus_url_col = st.selectbox("Select URL column", options=cols, index=0)
    with c2:
        metric_default_index = 1 if len(cols) > 1 else 0
        focus_metric_col = st.selectbox("Select Metric column (numeric)", options=cols, index=metric_default_index)

    st.caption(
        "Matching works if your focus URLs are full URLs (https://...) OR path-only (/solar/...). "
        "Metric can be sessions, revenue, conversions, or a manual priority score."
    )
else:
    st.info("No focus CSV uploaded — you’ll still get Top Destinations + downloads.")

run = st.button("🚀 Run analysis", type="primary")

if run:
    # Init aggregates
    filtered_out_path = None
    if cfg.write_filtered_csv:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        filtered_out_path = tmpf.name
        tmpf.close()

    agg = Aggregates(
        total_inlinks={},
        unique_sources={},
        anchor_total={},
        anchor_counts={},
        filtered_out_path=filtered_out_path,
        filtered_rows_written=0,
    )

    stats = FilterStats(
        total_rows_read=0,
        total_rows_kept=0,
        removed_self=0,
        removed_params=0,
        removed_external_destination=0,
        removed_type=0,
        removed_follow=0,
        removed_status=0,
        removed_link_position=0,
        removed_link_path=0,
        removed_anchor=0,
        removed_source_path=0,
    )

    progress = st.progress(0, text="Reading file in chunks...")
    wrote_header = False
    chunks_seen = 0

    for chunk in chunk_iterator(path, cfg.chunksize):
        chunks_seen += 1
        stats.total_rows_read += len(chunk)

        filtered = apply_row_filters_with_stats(chunk, cfg, stats)
        stats.total_rows_kept += len(filtered)

        if filtered.empty:
            if chunks_seen % 5 == 0:
                progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {stats.total_rows_read:,} rows...")
            continue

        wrote_header = maybe_write_filtered_chunk(filtered, cfg, agg, wrote_header)
        update_aggregates_from_chunk(filtered, cfg, agg)

        if chunks_seen % 5 == 0:
            progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {stats.total_rows_read:,} rows...")

    progress.progress(1.0, text=f"Done. Processed {stats.total_rows_read:,} rows.")

    if stats.total_rows_kept == 0:
        st.warning("No rows matched your filters. Try clearing include-only filters or disabling Anchor exclusions.")
        st.stop()

    # Build destination summary
    destinations = list(agg.total_inlinks.keys())
    top_anchor, top_anchor_pct = compute_top_anchor_fields(destinations, agg)

    out = pd.DataFrame({
        "Destination": destinations,
        "Total_Inlinks": [agg.total_inlinks[d] for d in destinations],
        "Unique_Source_Pages": [len(agg.unique_sources.get(d, set())) for d in destinations],
        "Top Anchor": top_anchor,
        "Top Anchor %": top_anchor_pct,
    })

    out = out.sort_values(["Total_Inlinks", "Unique_Source_Pages"], ascending=False).reset_index(drop=True)

    # =========================
    # Filter Impact Summary
    # =========================
    st.subheader("📊 Filter impact summary")

    total_removed_tracked = (
        stats.removed_self
        + stats.removed_params
        + stats.removed_external_destination
        + stats.removed_source_path
        + stats.removed_type
        + stats.removed_follow
        + stats.removed_status
        + stats.removed_link_position
        + stats.removed_link_path
        + stats.removed_anchor
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows read", f"{stats.total_rows_read:,}")
    with c2:
        st.metric("Rows kept", f"{stats.total_rows_kept:,}")
    with c3:
        st.metric("Rows removed (tracked)", f"{total_removed_tracked:,}")
    with c4:
        st.metric("Read - kept", f"{(stats.total_rows_read - stats.total_rows_kept):,}")

    breakdown = pd.DataFrame([
        {"Filter": "Remove self-referring", "Rows removed": stats.removed_self},
        {"Filter": "Exclude query params", "Rows removed": stats.removed_params},
        {"Filter": "Exclude external destinations", "Rows removed": stats.removed_external_destination},
        {"Filter": "Source path exclusions", "Rows removed": stats.removed_source_path},
        {"Filter": "Type include-only", "Rows removed": stats.removed_type},
        {"Filter": "Follow include-only", "Rows removed": stats.removed_follow},
        {"Filter": "Status Code include-only", "Rows removed": stats.removed_status},
        {"Filter": "Link Position include-only", "Rows removed": stats.removed_link_position},
        {"Filter": "Link Path exclusions", "Rows removed": stats.removed_link_path},
        {"Filter": "Anchor exclusions", "Rows removed": stats.removed_anchor},
    ])
    st.dataframe(breakdown, use_container_width=True)

    # =========================
    # Focus Page Priority Report
    # =========================
    if focus_df is not None and focus_url_col and focus_metric_col:
        st.subheader("🎯 Focus Page Priority Report")

        try:
            focus_report = build_focus_report(out, focus_df, focus_url_col, focus_metric_col)
        except Exception as e:
            st.error(f"Failed to build Focus Page report: {e}")
            focus_report = pd.DataFrame()

        if focus_report.empty:
            st.warning(
                "No focus pages matched the Destination URLs after filtering.\n\n"
                "Common causes:\n"
                "- Focus URLs are paths but Destination URLs are different paths (e.g. missing trailing segment)\n"
                "- Focus list includes URLs filtered out (e.g. non-200, excluded by anchor/link path filters)\n"
                "- Domain mismatch (if using external destination filtering)\n"
            )
        else:
            st.caption("Priority Score = Deficit Score × Value Score (Value Score is normalized within focus set).")
            rows_to_show = st.number_input("Focus report rows to show", min_value=10, max_value=2000, value=100, step=10)
            st.dataframe(focus_report.head(int(rows_to_show)), use_container_width=True)

            st.download_button(
                "Download Focus Page Priority Report CSV",
                data=focus_report.to_csv(index=False).encode("utf-8"),
                file_name="focus_page_priority_report.csv",
                mime="text/csv",
            )

    # =========================
    # Top Destinations
    # =========================
    st.subheader("🏆 Top Destination URLs")
    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
    out_top = out.head(int(top_n))
    st.dataframe(out_top, use_container_width=True)

    # =========================
    # Downloads
    # =========================
    st.subheader("⬇️ Downloads")

    st.download_button(
        "Download top destinations CSV",
        data=out_top.to_csv(index=False).encode("utf-8"),
        file_name="top_destinations.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download ALL destinations CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="all_destinations_summary.csv",
        mime="text/csv",
    )

    if agg.filtered_out_path is not None:
        with open(agg.filtered_out_path, "rb") as f:
            st.download_button(
                "Download FILTERED links CSV (all columns, filtered rows)",
                data=f.read(),
                file_name="filtered_all_inlinks.csv",
                mime="text/csv",
            )
        if cfg.max_filtered_rows and agg.filtered_rows_written >= int(cfg.max_filtered_rows):
            st.info(f"Filtered CSV was capped at **{agg.filtered_rows_written:,} rows** (max rows setting).")

    with st.expander("Notes"):
        st.markdown(
            """
- **Focus pages are destination targets only**. The report shows which of those targets have the biggest internal linking deficit.
- Matching works for focus URLs that are **full URLs** or **path-only**. Destination URLs can also be full or path-only.
- Source path exclusion matches against **Source PATH only** (e.g. `/page/`), useful for excluding paginated listing pages from being counted as linking sources.
"""
        )





