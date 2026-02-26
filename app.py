import re
import zipfile
import tempfile
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st

st.set_page_config(page_title="All Inlinks Analyzer (Chunk-safe)", layout="wide")
st.title("🔗 All Inlinks Analyzer (Chunk-safe for big files)")

uploaded_file = st.file_uploader("Upload All Inlinks (.csv, .csv.gz, or .zip)", type=["csv", "gz", "zip"])

st.sidebar.header("⚙️ Performance")
chunksize = st.sidebar.number_input(
    "Chunk size (rows)",
    min_value=50_000,
    max_value=1_000_000,
    value=200_000,
    step=50_000,
)

sample_rows = st.sidebar.number_input(
    "Sample rows (to detect real filter values)",
    min_value=5_000,
    max_value=200_000,
    value=50_000,
    step=5_000,
)

# Anchor counting can get memory-heavy on huge/dense sites
max_distinct_anchors_per_destination = st.sidebar.number_input(
    "Max distinct anchors tracked per destination",
    min_value=25,
    max_value=2000,
    value=300,
    step=25,
    help=(
        "Caps how many distinct anchor texts are stored per destination to keep memory stable. "
        "Higher = more accurate top-anchor detection, more memory."
    ),
)

st.sidebar.header("⬇️ Output files")
write_filtered_csv = st.sidebar.checkbox(
    "Create downloadable filtered CSV (can be large)",
    value=False,
    help=(
        "Writes the filtered rows to a CSV while processing. "
        "This can take longer and the resulting file may still be very large."
    ),
)

max_filtered_rows = st.sidebar.number_input(
    "Max filtered rows to write (0 = no limit)",
    min_value=0,
    max_value=50_000_000,
    value=0,
    step=100_000,
    help="Set a cap to prevent generating a massive output file.",
)

# -------------------------
# Helpers
# -------------------------
def normalize_url_for_compare(u: str) -> str:
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

def compile_contains_patterns(lines: str) -> re.Pattern | None:
    """
    Each non-empty line is treated as a 'contains' token.
    We escape tokens and OR them into a single case-insensitive regex.
    """
    pats = []
    for line in (lines or "").splitlines():
        line = line.strip()
        if line:
            pats.append(re.escape(line))
    if not pats:
        return None
    return re.compile("|".join(pats), flags=re.IGNORECASE)

def materialize_to_path(uploaded) -> str:
    name = (uploaded.name or "").lower()

    if name.endswith(".zip"):
        uploaded.seek(0)
        with zipfile.ZipFile(uploaded) as z:
            members = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ValueError("No .csv found inside the .zip.")
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
    kwargs = dict(
        nrows=int(nrows),
        low_memory=False,
        dtype="string",
        compression="infer",
    )
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "cp1252"
        return pd.read_csv(path, **kwargs)

def uniq(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    vals = df[col].astype("string").fillna("").unique().tolist()
    return sorted(vals, key=lambda x: (x == "" or x is None, str(x).lower()))

def norm_anchor(a: str) -> str:
    # normalize whitespace + lowercase for comparisons
    a = "" if a is None else str(a)
    a = a.strip()
    a = re.sub(r"\s+", " ", a)
    return a.lower()

# -------------------------
# Load input
# -------------------------
if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    path = materialize_to_path(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

sample_df = read_sample(path, int(sample_rows))

required = {"Source", "Destination"}
missing = required - set(sample_df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

# Detect real filter values from sample
type_options = uniq(sample_df, "Type")
follow_options = uniq(sample_df, "Follow")
status_options = uniq(sample_df, "Status Code")
linkpos_options = uniq(sample_df, "Link Position")

# -------------------------
# Filters UI
# -------------------------
st.subheader("🎛️ Filters")

preset = st.button("✨ Preset: Content + Follow + 200 + Hyperlink")

c1, c2, c3 = st.columns(3)
with c1:
    remove_self = st.checkbox("Remove self-referring (Source == Destination)", value=True)
with c2:
    exclude_params = st.checkbox("Exclude Destination with '?' (query params)", value=True)
with c3:
    st.caption("Tip: leave include-only filters empty to keep all values.")

default_type = ["Hyperlink"] if ("Hyperlink" in type_options) else []
default_follow = ["Follow"] if ("Follow" in follow_options) else []
default_status = ["200"] if ("200" in status_options) else (["200 OK"] if ("200 OK" in status_options) else [])
default_linkpos = ["Content"] if ("Content" in linkpos_options) else []

if preset:
    type_keep = default_type
    follow_keep = default_follow
    status_keep = default_status
    link_position_keep = default_linkpos
else:
    type_keep = st.multiselect("Keep Type values (empty = keep all)", options=type_options, default=default_type)
    follow_keep = st.multiselect("Keep Follow values (empty = keep all)", options=follow_options, default=default_follow)
    status_keep = st.multiselect("Keep Status Code values (empty = keep all)", options=status_options, default=default_status)
    link_position_keep = st.multiselect("Keep Link Position values (empty = keep all)", options=linkpos_options, default=default_linkpos)

st.markdown("### Exclude by Link Path patterns (breadcrumbs/nav etc.)")
exclude_by_link_path = st.checkbox("Enable Link Path exclusions", value=True)
default_lp = "\n".join(["breadcrumb", "/ol/li", "aria-label=\"breadcrumb\"", "aria-label='breadcrumb'"])
lp_text = st.text_area("Link Path patterns to exclude (one per line)", value=default_lp, height=100, disabled=not exclude_by_link_path)
rx_lp = compile_contains_patterns(lp_text) if exclude_by_link_path else None

st.markdown("### 🚫 Exclude by Anchor text (template CTAs etc.)")
enable_anchor_exclusions = st.checkbox(
    "Enable Anchor exclusions",
    value=True,
    help="Excludes rows where Anchor contains any of the phrases below (case-insensitive)."
)

default_anchor_excl = "\n".join([
    "discover more",
    "read more",
    "learn more",
    "find out more",
    "view more",
    "see more",
])
anchor_excl_text = st.text_area(
    "Anchor phrases to exclude (one per line, 'contains' match)",
    value=default_anchor_excl,
    height=120,
    disabled=not enable_anchor_exclusions
)
rx_anchor_excl = compile_contains_patterns(anchor_excl_text) if enable_anchor_exclusions else None

st.markdown("### 🧠 Optional: template CTA dominance filter (Top Anchor %)")
enable_anchor_dominance_filter = st.checkbox(
    "Enable Top Anchor % filter",
    value=False,
    help=(
        "After analysis, exclude destinations whose Top Anchor accounts for >= threshold% "
        "of all anchor occurrences to that destination."
    ),
)

dominance_threshold = st.slider(
    "Exclude destinations with Top Anchor % ≥",
    min_value=0,
    max_value=100,
    value=70,
    step=1,
    disabled=not enable_anchor_dominance_filter,
)

only_apply_dominance_when_top_anchor_is_cta = st.checkbox(
    "Only apply Top Anchor % filter when Top Anchor matches CTA list above",
    value=True,
    disabled=not enable_anchor_dominance_filter,
    help="Recommended: avoids excluding genuinely popular pages with diverse contextual anchors.",
)

# -------------------------
# Run analysis
# -------------------------
run = st.button("🚀 Run analysis", type="primary")

if run:
    # Core aggregates
    total_inlinks: dict[str, int] = {}
    unique_sources: dict[str, set[str]] = {}

    # Anchor aggregates (for Top Anchor + %)
    anchor_total: dict[str, int] = {}                 # dest -> total non-empty anchor occurrences
    anchor_counts: dict[str, dict[str, int]] = {}     # dest -> {anchor_text -> count} (capped)
    anchor_other: dict[str, int] = {}                 # dest -> count of anchors not stored due to cap

    # Optional filtered CSV output
    filtered_out_path = None
    wrote_header = False
    filtered_rows_written = 0

    if write_filtered_csv:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        filtered_out_path = tmpf.name
        tmpf.close()

    # Prepare anchor exclusion normalized CTA set (for dominance "only when CTA")
    cta_set = set()
    for line in (anchor_excl_text or "").splitlines():
        line = norm_anchor(line)
        if line:
            cta_set.add(line)

    progress = st.progress(0, text="Reading file in chunks...")

    read_kwargs = dict(
        chunksize=int(chunksize),
        low_memory=False,
        dtype="string",
        compression="infer",
    )

    try:
        iterator = pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "cp1252"
        iterator = pd.read_csv(path, **read_kwargs)

    rows_seen = 0
    chunks_seen = 0

    for chunk in iterator:
        chunks_seen += 1
        rows_seen += len(chunk)

        if "Source" not in chunk.columns or "Destination" not in chunk.columns:
            continue

        # normalize required cols
        chunk["Source"] = chunk["Source"].astype("string").fillna("")
        chunk["Destination"] = chunk["Destination"].astype("string").fillna("")

        # self-links
        if remove_self:
            src_norm = chunk["Source"].map(normalize_url_for_compare)
            dst_norm = chunk["Destination"].map(normalize_url_for_compare)
            chunk = chunk[src_norm != dst_norm]

        # params
        if exclude_params:
            chunk = chunk[~chunk["Destination"].str.contains(r"\?", regex=True, na=False)]

        # include-only filters (only apply if selection is non-empty and column exists)
        if type_keep and "Type" in chunk.columns:
            chunk = chunk[chunk["Type"].astype("string").fillna("").isin(type_keep)]

        if follow_keep and "Follow" in chunk.columns:
            chunk = chunk[chunk["Follow"].astype("string").fillna("").isin(follow_keep)]

        if status_keep and "Status Code" in chunk.columns:
            chunk = chunk[chunk["Status Code"].astype("string").fillna("").isin(status_keep)]

        if link_position_keep and "Link Position" in chunk.columns:
            chunk = chunk[chunk["Link Position"].astype("string").fillna("").isin(link_position_keep)]

        # Link Path exclusions
        if rx_lp is not None and "Link Path" in chunk.columns:
            lps = chunk["Link Path"].astype("string").fillna("")
            chunk = chunk[~lps.str.contains(rx_lp, na=False)]

        # Anchor exclusions (row-level)
        if rx_anchor_excl is not None and "Anchor" in chunk.columns:
            anc = chunk["Anchor"].astype("string").fillna("")
            # exclude rows whose Anchor contains excluded phrase(s)
            chunk = chunk[~anc.str.contains(rx_anchor_excl, na=False)]

        if chunk.empty:
            if chunks_seen % 5 == 0:
                progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {rows_seen:,} rows...")
            continue

        # Stream-write filtered rows (optional)
        if filtered_out_path is not None:
            if max_filtered_rows == 0 or filtered_rows_written < int(max_filtered_rows):
                remaining = None if max_filtered_rows == 0 else int(max_filtered_rows) - filtered_rows_written
                chunk_to_write = chunk if remaining is None else chunk.head(max(0, remaining))

                mode = "w" if not wrote_header else "a"
                chunk_to_write.to_csv(
                    filtered_out_path,
                    mode=mode,
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
                filtered_rows_written += len(chunk_to_write)

        # Anchor series for stats (after row-level anchor exclusions)
        anchor_series = chunk["Anchor"].astype("string").fillna("") if "Anchor" in chunk.columns else None

        dests = chunk["Destination"].tolist()
        srcs = chunk["Source"].tolist()
        anchors = anchor_series.tolist() if anchor_series is not None else ["" for _ in dests]

        for d, s, a in zip(dests, srcs, anchors):
            if not d:
                continue

            total_inlinks[d] = total_inlinks.get(d, 0) + 1
            unique_sources.setdefault(d, set()).add(s)

            a_str = str(a).strip()
            if a_str:
                anchor_total[d] = anchor_total.get(d, 0) + 1

                dest_map = anchor_counts.get(d)
                if dest_map is None:
                    dest_map = {}
                    anchor_counts[d] = dest_map

                if a_str in dest_map:
                    dest_map[a_str] += 1
                else:
                    if len(dest_map) < int(max_distinct_anchors_per_destination):
                        dest_map[a_str] = 1
                    else:
                        anchor_other[d] = anchor_other.get(d, 0) + 1

        if chunks_seen % 5 == 0:
            progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {rows_seen:,} rows...")

    progress.progress(1.0, text=f"Done. Processed {rows_seen:,} rows.")

    if not total_inlinks:
        st.warning(
            "No rows matched your filters.\n\n"
            "Try clearing one or more include-only filters, or disabling Anchor exclusions, then run again."
        )
        st.stop()

    # Build base output with top anchor + %
    destinations = list(total_inlinks.keys())
    top_anchor = []
    top_anchor_pct = []

    for d in destinations:
        total_anchor_occ = anchor_total.get(d, 0)
        if total_anchor_occ == 0:
            top_anchor.append("")
            top_anchor_pct.append(0.0)
            continue

        amap = anchor_counts.get(d, {})
        if not amap:
            top_anchor.append("")
            top_anchor_pct.append(0.0)
            continue

        best_a, best_c = max(amap.items(), key=lambda kv: kv[1])
        pct = (best_c / total_anchor_occ) * 100.0
        top_anchor.append(best_a)
        top_anchor_pct.append(round(pct, 2))

    out = pd.DataFrame({
        "Destination": destinations,
        "Total_Inlinks": [total_inlinks[d] for d in destinations],
        "Unique_Source_Pages": [len(unique_sources.get(d, set())) for d in destinations],
        "Top Anchor": top_anchor,
        "Top Anchor %": top_anchor_pct,
    })

    # Apply optional dominance filter (destination-level, after stats computed)
    if enable_anchor_dominance_filter:
        before = len(out)
        if only_apply_dominance_when_top_anchor_is_cta:
            # apply only where top anchor (normalized) matches one of the CTA phrases
            norm_top = out["Top Anchor"].map(norm_anchor)
            is_cta = norm_top.isin(cta_set)
            out = out[~(is_cta & (out["Top Anchor %"] >= float(dominance_threshold)))]
        else:
            out = out[out["Top Anchor %"] < float(dominance_threshold)]

        st.info(f"Top Anchor % filter removed **{before - len(out):,}** destinations.")

    # Sort output
    out = out.sort_values(["Total_Inlinks", "Unique_Source_Pages"], ascending=False).reset_index(drop=True)

    st.subheader("🏆 Top Destination URLs")
    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
    out_top = out.head(int(top_n))
    st.dataframe(out_top, use_container_width=True)

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

    if filtered_out_path is not None:
        with open(filtered_out_path, "rb") as f:
            st.download_button(
                "Download FILTERED links CSV (all columns, filtered rows)",
                data=f.read(),
                file_name="filtered_all_inlinks.csv",
                mime="text/csv",
            )
        if max_filtered_rows and filtered_rows_written >= int(max_filtered_rows):
            st.info(f"Filtered CSV was capped at **{filtered_rows_written:,} rows** (max rows setting).")

    with st.expander("How the CTA/template filters work"):
        st.markdown(
            """
**Anchor exclusions (row-level)**  
Removes individual links where the **Anchor** contains any excluded phrase (case-insensitive).  
This is great for template card CTAs like “Read more” / “Discover more”.

**Top Anchor % filter (destination-level)**  
After computing top anchors for each destination, you can remove destinations where one anchor dominates  
(e.g. Top Anchor % ≥ 70%).  

Recommended setting: **Only apply when Top Anchor matches your CTA list**, so you don’t accidentally hide a genuinely popular page whose top anchor is meaningful.
"""
        )





