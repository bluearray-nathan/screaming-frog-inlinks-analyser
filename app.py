# app.py
# Streamlit Cloud–ready: Screaming Frog "All Inlinks" Analyzer (chunk-safe)
# + Focus Page Priority Report (focus pages are DESTINATION targets only)
# + Internal Linking Recommendations (MVP):
#     - Upload embeddings (URL + embeddings string)
#     - Upload source metrics (URL + metric)
#     - Recommend "link from X -> link to Y" for priority targets
#     - Enforce similarity threshold, optional same folder, cap recs per source
#     - Avoid recommending links that already exist (from All Inlinks)

import re
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Set, Optional, Tuple, List
from urllib.parse import urlsplit, urlunsplit

import numpy as np
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

    # Exclude rows by SOURCE path (row-level)
    rx_source_path_excl: Optional[re.Pattern]

    # Destination-level dominance filter (module amplification)
    enable_anchor_dominance_filter: bool
    dominance_threshold: int
    dominance_min_unique_sources: int


@dataclass
class Aggregates:
    total_inlinks: Dict[str, int]                      # Destination raw string -> count
    unique_sources: Dict[str, Set[str]]                # Destination raw -> set(Source raw)

    anchor_total: Dict[str, int]                       # Destination raw -> total non-empty anchor occurrences
    anchor_counts: Dict[str, Dict[str, int]]           # Destination raw -> {anchor_text -> count} (capped)

    # For recommendation engine: canonical keys + existing links
    existing_link_pairs: Set[Tuple[str, str]]          # (source_key, dest_key)
    key_to_url: Dict[str, str]                         # canonical key -> best-seen absolute/relative URL

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
    removed_source_path: int

    # Destination-level dominance stats
    destinations_before: int
    destinations_removed_by_dominance: int


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
    """Return path portion of Source."""
    if src is None or pd.isna(src):
        return ""
    s = str(src).strip()
    if not s:
        return ""
    parts = urlsplit(s)
    return parts.path or ""


def canonical_url_key(url: str) -> str:
    """
    Canonical key for matching across:
    - All Inlinks Source/Destination
    - Embeddings URLs
    - Metrics URLs
    - Focus URLs

    Rules:
    - Absolute URL => host(lower, no www) + normalized path (trim trailing slash) ; ignore query/fragment
    - Relative/path-only => normalized path (leading /, trim trailing slash) ; ignore query/fragment
    """
    if url is None or pd.isna(url):
        return ""
    s = str(url).strip()
    if not s:
        return ""

    parts = urlsplit(s)

    # Relative/path-only
    if parts.scheme == "" and parts.netloc == "":
        p = parts.path if parts.path else s
        p = p.strip()
        # strip any query/fragment if user pasted them in a "relative" string
        if "?" in p:
            p = p.split("?", 1)[0]
        if "#" in p:
            p = p.split("#", 1)[0]
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


def url_path_for_folder(url: str) -> str:
    """Get a normalized path from URL (absolute or relative)."""
    if url is None or pd.isna(url):
        return ""
    s = str(url).strip()
    if not s:
        return ""
    parts = urlsplit(s)
    p = parts.path or ""
    if not p.startswith("/"):
        p = "/" + p
    if p != "/" and p.endswith("/"):
        p = p.rstrip("/")
    return p


def folder_key(url: str, depth: int = 1) -> str:
    """
    Folder key based on first N path segments.
    depth=1 => '/blog'
    depth=2 => '/blog/category'
    """
    p = url_path_for_folder(url)
    if not p or p == "/":
        return "/"
    segs = [seg for seg in p.split("/") if seg]
    if not segs:
        return "/"
    depth = max(1, int(depth))
    take = segs[:depth]
    return "/" + "/".join(take)


def focus_key(url: str) -> str:
    """Backward-compatible alias (used by focus report)."""
    return canonical_url_key(url)


def destination_join_key(destination_url: str) -> str:
    """Backward-compatible alias (used by focus report)."""
    return canonical_url_key(destination_url)


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
                cfg.keep_relative_destinations_as_internal,
            )
        )
        chunk = chunk[mask]
        stats.removed_external_destination += (before - len(chunk))

    # 4) exclude by Source PATH patterns (e.g. /page/)
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

        # Destination summary aggregation (raw URLs)
        agg.total_inlinks[d] = agg.total_inlinks.get(d, 0) + 1
        agg.unique_sources.setdefault(d, set()).add(s)

        # For recommender: store canonical key link pair
        sk = canonical_url_key(s)
        dk = canonical_url_key(d)
        if sk and dk:
            agg.existing_link_pairs.add((sk, dk))

        # key_to_url mapping (best effort)
        if sk and sk not in agg.key_to_url and s:
            agg.key_to_url[sk] = s
        if dk and dk not in agg.key_to_url and d:
            agg.key_to_url[dk] = d

        # Anchor distribution per destination (raw destination string)
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


def apply_destination_level_filters(out: pd.DataFrame, cfg: AppConfig, stats: FilterStats) -> pd.DataFrame:
    stats.destinations_before = len(out)

    if not cfg.enable_anchor_dominance_filter:
        stats.destinations_removed_by_dominance = 0
        return out

    thr = float(cfg.dominance_threshold)
    min_u = int(cfg.dominance_min_unique_sources)

    filtered = out[~((out["Top Anchor %"] >= thr) & (out["Unique_Source_Pages"] >= min_u))]
    stats.destinations_removed_by_dominance = len(out) - len(filtered)
    return filtered


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
        help="Example: example.com, blog.example.com",
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

    # Source path exclusion (row-level)
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
    lp_text = st.text_area(
        "Link Path patterns to exclude (one per line)",
        value=default_lp,
        height=100,
        disabled=not exclude_by_link_path,
    )
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

    # Dominance filter (destination-level) – recommended for repeated “related/trending blocks”
    st.markdown("### 🧠 Anchor dominance filter (module amplification)")
    enable_anchor_dominance_filter = st.checkbox(
        "Enable dominance filter (Destination-level)",
        value=True,
        help="Excludes destinations where the Top Anchor dominates AND the destination has enough unique source pages. "
             "Useful for removing 'related/trending block' inflation where anchors are repeated titles.",
    )
    dominance_threshold = st.slider(
        "Exclude destinations when Top Anchor % ≥",
        0,
        100,
        90,
        1,
        disabled=not enable_anchor_dominance_filter,
        help="Recommended 85–95 for related/trending blocks using repeated title anchors.",
    )
    dominance_min_unique_sources = st.number_input(
        "…and Unique Source Pages ≥",
        min_value=0,
        max_value=1_000_000,
        value=20,
        step=5,
        disabled=not enable_anchor_dominance_filter,
        help="Prevents tiny samples (e.g. 3 links) from being excluded just because they share an anchor.",
    )

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

        enable_anchor_dominance_filter=enable_anchor_dominance_filter,
        dominance_threshold=int(dominance_threshold),
        dominance_min_unique_sources=int(dominance_min_unique_sources),
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
# Embeddings + Metrics (MVP recommender)
# =========================
def load_csv_with_fallback(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    kwargs = dict(low_memory=False, dtype="string")
    try:
        df = pd.read_csv(uploaded, **kwargs)
    except UnicodeDecodeError:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="cp1252", **kwargs)
    return df


@st.cache_data(show_spinner=False)
def parse_embeddings_df(df: pd.DataFrame, url_col: str, emb_col: str) -> Tuple[pd.DataFrame, int, int]:
    """
    Returns (emb_df, dim, bad_rows)
    emb_df columns: ['__url', '__key', '__emb'] where __emb is np.ndarray float32.
    """
    tmp = df[[url_col, emb_col]].copy()
    tmp["__url"] = tmp[url_col].astype("string").fillna("").str.strip()
    tmp["__key"] = tmp["__url"].map(canonical_url_key)

    bad = 0
    vecs = []
    keys = []
    urls = []

    dim = None
    for u, k, emb in zip(tmp["__url"].tolist(), tmp["__key"].tolist(), tmp[emb_col].astype("string").fillna("").tolist()):
        if not u or not k:
            bad += 1
            continue
        s = str(emb).strip()
        if not s:
            bad += 1
            continue
        try:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        except Exception:
            bad += 1
            continue
        if arr.size == 0:
            bad += 1
            continue
        if dim is None:
            dim = int(arr.size)
        elif int(arr.size) != int(dim):
            bad += 1
            continue
        vecs.append(arr)
        keys.append(k)
        urls.append(u)

    if dim is None or len(vecs) == 0:
        return pd.DataFrame(columns=["__url", "__key", "__emb"]), 0, bad

    out = pd.DataFrame({"__url": urls, "__key": keys})
    out["__emb"] = vecs
    # drop duplicates by key (keep first)
    out = out.drop_duplicates(subset="__key", keep="first").reset_index(drop=True)
    return out, dim, bad


@st.cache_resource(show_spinner=False)
def build_normalized_matrix(keys: Tuple[str, ...], vecs: Tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Build an L2-normalized matrix (N x D) float32.
    Cached by (keys, vecs) identity; caller should pass stable tuples.
    """
    X = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return X


def topk_cosine_sim(X_norm: np.ndarray, idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, sims) for top-k most similar to row idx (excluding itself).
    """
    v = X_norm[idx]
    sims = X_norm @ v  # cosine similarity because normalized
    sims[idx] = -1.0   # exclude self
    if k >= len(sims):
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, kth=k)[:k]
        order = part[np.argsort(-sims[part])]
    return order, sims[order]


def normalize_metric(series: pd.Series) -> pd.Series:
    """
    Normalize a metric to 0..1 with log1p scaling + max division (robust-ish).
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    s = np.log1p(s)
    mx = float(s.max()) if len(s) else 0.0
    if mx <= 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return s / mx


def best_url_for_key(key: str, fallback: str, agg: Aggregates) -> str:
    """
    Prefer URL captured from inlinks processing, else fallback (embeddings/metrics).
    """
    if key in agg.key_to_url and agg.key_to_url[key]:
        return agg.key_to_url[key]
    return fallback


def build_recommendations(
    focus_report: pd.DataFrame,
    agg: Aggregates,
    emb_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    metrics_url_col: str,
    metrics_value_col: str,
    similarity_threshold: float,
    per_target_k: int,
    max_recs_per_target: int,
    max_recs_per_source: int,
    enforce_same_folder: bool,
    folder_depth: int,
) -> pd.DataFrame:
    """
    Returns a dataframe of recommendations: Source -> Target, scored & capped.
    Requires focus_report with Destination + Priority Score.
    """

    # --- metrics map ---
    m = metrics_df[[metrics_url_col, metrics_value_col]].copy()
    m["__url"] = m[metrics_url_col].astype("string").fillna("").str.strip()
    m["__key"] = m["__url"].map(canonical_url_key)
    m["__metric_raw"] = m[metrics_value_col].astype("string").fillna("")
    m["__metric"] = pd.to_numeric(m["__metric_raw"].str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0.0)
    # keep best (max metric) per key
    m = m.sort_values("__metric", ascending=False).drop_duplicates(subset="__key", keep="first")
    m["__metric_norm"] = normalize_metric(m["__metric"])
    metric_map = dict(zip(m["__key"], m["__metric"]))
    metric_norm_map = dict(zip(m["__key"], m["__metric_norm"]))
    metric_url_map = dict(zip(m["__key"], m["__url"]))

    # --- embeddings index ---
    keys = emb_df["__key"].tolist()
    urls = emb_df["__url"].tolist()
    vecs = emb_df["__emb"].tolist()

    key_to_idx = {k: i for i, k in enumerate(keys)}
    idx_to_key = {i: k for i, k in enumerate(keys)}
    idx_to_url = {i: u for i, u in enumerate(urls)}

    X_norm = build_normalized_matrix(tuple(keys), tuple(vecs))

    # --- focus targets list ---
    fr = focus_report.copy()
    fr["__target_url"] = fr["Destination"].astype("string").fillna("").str.strip()
    fr["__target_key"] = fr["__target_url"].map(canonical_url_key)
    fr["__priority"] = fr["Priority Score"].astype(float)

    # keep only targets with embeddings
    fr = fr[fr["__target_key"].isin(key_to_idx)].copy()
    if fr.empty:
        return pd.DataFrame()

    # deterministic order: priority desc
    fr = fr.sort_values("__priority", ascending=False).reset_index(drop=True)

    rec_rows = []
    used_per_source = {}  # source_key -> count

    # We’ll build and later cap per-target, but also enforce per-source online.
    for _, row in fr.iterrows():
        t_url = row["__target_url"]
        t_key = row["__target_key"]
        t_priority = float(row["__priority"])

        t_idx = key_to_idx.get(t_key)
        if t_idx is None:
            continue

        t_folder = folder_key(t_url, depth=folder_depth)

        cand_idx, cand_sims = topk_cosine_sim(X_norm, t_idx, k=max(per_target_k, max_recs_per_target * 3))

        per_target_added = 0
        for si, sim in zip(cand_idx.tolist(), cand_sims.tolist()):
            if sim < float(similarity_threshold):
                break  # because sorted desc
            s_key = idx_to_key[si]
            s_url_raw = idx_to_url[si]

            # optional same folder
            if enforce_same_folder:
                s_folder = folder_key(s_url_raw, depth=folder_depth)
                if s_folder != t_folder:
                    continue
            else:
                s_folder = folder_key(s_url_raw, depth=folder_depth)

            # avoid recommending if already linked
            if (s_key, t_key) in agg.existing_link_pairs:
                continue

            # cap per source
            c = used_per_source.get(s_key, 0)
            if c >= int(max_recs_per_source):
                continue

            # metric for source (optional but recommended; if missing, treat as 0)
            s_metric = float(metric_map.get(s_key, 0.0))
            s_metric_norm = float(metric_norm_map.get(s_key, 0.0))

            score = (t_priority * s_metric_norm * float(sim))

            rec_rows.append({
                "Source URL": best_url_for_key(s_key, metric_url_map.get(s_key, s_url_raw), agg),
                "Target URL": best_url_for_key(t_key, t_url, agg),
                "Similarity": round(float(sim), 4),
                "Source Metric": s_metric,
                "Source Metric Score": round(s_metric_norm, 4),
                "Target Priority Score": round(t_priority, 6),
                "Recommendation Score": round(score, 6),
                "Source Folder": s_folder,
                "Target Folder": t_folder,
            })

            used_per_source[s_key] = c + 1
            per_target_added += 1
            if per_target_added >= int(max_recs_per_target):
                break

    if not rec_rows:
        return pd.DataFrame()

    out = pd.DataFrame(rec_rows)
    out = out.sort_values(["Recommendation Score", "Similarity"], ascending=False).reset_index(drop=True)
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
        "Matching works if your focus URLs are full URLs (https://...) OR path-only (/blog/... ). "
        "Metric can be sessions, revenue, conversions, or a manual priority score."
    )
else:
    st.info("No focus CSV uploaded — you’ll still get Top Destinations + downloads.")

st.divider()
st.subheader("🧩 Internal Linking Recommendations (MVP) — optional uploads")
emb_upload = st.file_uploader("Upload Embeddings CSV (URL + embeddings)", type=["csv"], key="emb_csv")
met_upload = st.file_uploader("Upload Source Metrics CSV (URL + metric)", type=["csv"], key="met_csv")

emb_df_raw = load_csv_with_fallback(emb_upload)
met_df_raw = load_csv_with_fallback(met_upload)

emb_url_col = emb_emb_col = None
met_url_col = met_val_col = None

if emb_df_raw is not None and not emb_df_raw.empty:
    st.success(f"Embeddings CSV loaded: {emb_df_raw.shape[0]:,} rows × {emb_df_raw.shape[1]:,} cols")
    with st.expander("Preview embeddings CSV"):
        st.dataframe(emb_df_raw.head(20), use_container_width=True)

    ecols = list(emb_df_raw.columns)
    c1, c2 = st.columns(2)
    with c1:
        emb_url_col = st.selectbox("Embeddings: URL column", options=ecols, index=0, key="emb_url_col")
    with c2:
        emb_emb_col = st.selectbox("Embeddings: Embedding column", options=ecols, index=1 if len(ecols) > 1 else 0, key="emb_emb_col")
else:
    st.info("No embeddings CSV uploaded (recommendations will be unavailable).")

if met_df_raw is not None and not met_df_raw.empty:
    st.success(f"Source metrics CSV loaded: {met_df_raw.shape[0]:,} rows × {met_df_raw.shape[1]:,} cols")
    with st.expander("Preview source metrics CSV"):
        st.dataframe(met_df_raw.head(20), use_container_width=True)

    mcols = list(met_df_raw.columns)
    c1, c2 = st.columns(2)
    with c1:
        met_url_col = st.selectbox("Metrics: URL column", options=mcols, index=0, key="met_url_col")
    with c2:
        met_val_col = st.selectbox("Metrics: Metric column (numeric)", options=mcols, index=1 if len(mcols) > 1 else 0, key="met_val_col")
else:
    st.info("No source metrics CSV uploaded (recommendations will still run, but sources without metrics score as 0).")

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
        existing_link_pairs=set(),
        key_to_url={},
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
        destinations_before=0,
        destinations_removed_by_dominance=0,
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

    # Apply dominance filter
    out = apply_destination_level_filters(out, cfg, stats)

    # Sort final
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

    st.caption(
        f"Destinations before dominance filter: {stats.destinations_before:,} • "
        f"Removed by dominance filter: {stats.destinations_removed_by_dominance:,}"
    )

    # =========================
    # Focus Page Priority Report
    # =========================
    focus_report = None
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
                "- Focus URLs are paths but Destination URLs are different paths\n"
                "- Focus list includes URLs filtered out\n"
                "- Domain mismatch (if using destination domain filtering)\n"
                "- Dominance filter removed focus destinations (try disabling it)\n"
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
    # Internal Linking Recommendations (MVP)
    # =========================
    st.subheader("🧩 Internal Linking Recommendations (MVP)")

    if focus_report is None or focus_report is False or not isinstance(focus_report, pd.DataFrame) or focus_report.empty:
        st.info("Upload a Focus Pages CSV and run analysis to generate a priority set. Recommendations are based on that priority set.")
    elif emb_df_raw is None or emb_url_col is None or emb_emb_col is None:
        st.info("Upload an embeddings CSV (URL + embeddings) to enable recommendations.")
    else:
        # Parse embeddings
        emb_parsed, dim, bad_rows = parse_embeddings_df(emb_df_raw, emb_url_col, emb_emb_col)
        if emb_parsed.empty or dim == 0:
            st.error("Embeddings parsing produced no usable vectors. Check your selected columns.")
        else:
            st.caption(f"Parsed embeddings: {len(emb_parsed):,} URLs • dim={dim:,} • skipped rows={bad_rows:,}")

            # Config UI
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.78, 0.01)
            with c2:
                top_focus_n = st.number_input("Top focus targets to process", min_value=10, max_value=5000, value=100, step=10)
            with c3:
                per_target_k = st.number_input("Candidate sources to consider per target (K)", min_value=10, max_value=5000, value=200, step=10)
            with c4:
                max_recs_per_target = st.number_input("Max recommendations per target", min_value=1, max_value=200, value=20, step=1)

            c1, c2, c3 = st.columns(3)
            with c1:
                max_recs_per_source = st.number_input("Cap recommendations per source", min_value=1, max_value=200, value=5, step=1)
            with c2:
                enforce_same_folder = st.checkbox("Only recommend within same folder", value=True)
            with c3:
                folder_depth = st.selectbox("Folder depth", options=[1, 2, 3], index=0, disabled=not enforce_same_folder)

            use_metrics = (met_df_raw is not None and met_url_col is not None and met_val_col is not None)
            if not use_metrics:
                st.warning("No source metrics selected. Sources without metrics will score 0 (Recommendation Score will be 0).")

            # Compute recommendations
            fr_top = focus_report.head(int(top_focus_n)).copy()

            try:
                recs = build_recommendations(
                    focus_report=fr_top,
                    agg=agg,
                    emb_df=emb_parsed,
                    metrics_df=(met_df_raw if use_metrics else pd.DataFrame({met_url_col or "url": [], met_val_col or "metric": []})),
                    metrics_url_col=(met_url_col or "url"),
                    metrics_value_col=(met_val_col or "metric"),
                    similarity_threshold=float(similarity_threshold),
                    per_target_k=int(per_target_k),
                    max_recs_per_target=int(max_recs_per_target),
                    max_recs_per_source=int(max_recs_per_source),
                    enforce_same_folder=bool(enforce_same_folder),
                    folder_depth=int(folder_depth),
                )
            except Exception as e:
                st.error(f"Failed to build recommendations: {e}")
                recs = pd.DataFrame()

            if recs.empty:
                st.warning(
                    "No recommendations produced. Common causes:\n"
                    "- Similarity threshold too high\n"
                    "- Focus targets missing embeddings\n"
                    "- Most candidates already link to targets (dedup)\n"
                    "- Folder constraint too strict\n"
                    "- Metrics missing (scores may all be 0; still should show rows though)\n"
                )
            else:
                st.caption("Recommendations are 'link from Source URL → link to Target URL'. No anchor text is proposed in this MVP.")
                rows_show = st.number_input("Recommendation rows to show", min_value=10, max_value=5000, value=200, step=10)
                st.dataframe(recs.head(int(rows_show)), use_container_width=True)

                st.download_button(
                    "Download recommendations CSV",
                    data=recs.to_csv(index=False).encode("utf-8"),
                    file_name="internal_link_recommendations.csv",
                    mime="text/csv",
                )

                # Source-centric view (helpful for implementation)
                st.markdown("#### Source-centric view (group by Source URL)")
                src_view = (
                    recs.sort_values(["Source URL", "Recommendation Score"], ascending=[True, False])
                    .groupby("Source URL", as_index=False)
                    .head(int(max_recs_per_source))
                    .reset_index(drop=True)
                )
                st.dataframe(src_view.head(int(rows_show)), use_container_width=True)

                st.download_button(
                    "Download source-centric recommendations CSV",
                    data=src_view.to_csv(index=False).encode("utf-8"),
                    file_name="internal_link_recommendations_by_source.csv",
                    mime="text/csv",
                )

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
- **Dominance filter** removes DESTINATIONS from the destination summary when:
  - **Top Anchor % ≥ threshold**, and
  - **Unique_Source_Pages ≥ minimum**.
  This reduces inflation from repeated “related/trending blocks” where anchors are repeated (often titles).

- **Recommendations (MVP)**:
  - Require **Focus Page Priority Report** + **Embeddings CSV**.
  - Optional **Source metrics CSV** boosts valuable source pages.
  - Enforces **similarity threshold**, optional **same folder**, caps **recs per source**, and avoids **already-existing links**.
  - Outputs only: **link from page X → link to page Y** (no anchor text yet).
"""
        )





