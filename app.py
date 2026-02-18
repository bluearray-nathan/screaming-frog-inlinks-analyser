import re
import zipfile
import tempfile
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st

st.set_page_config(page_title="All Inlinks Analyzer (Chunked)", layout="wide")
st.title("🔗 All Inlinks Analyzer (Chunk-safe for big files)")
st.write(
    "This version processes large All Inlinks exports in **chunks** to avoid Streamlit Cloud memory crashes. "
    "It outputs **Top Destination URLs** based on the filters you select."
)

uploaded_file = st.file_uploader("Upload All Inlinks (.csv, .csv.gz, or .zip)", type=["csv", "gz", "zip"])

st.sidebar.header("⚙️ Performance")
chunksize = st.sidebar.number_input(
    "Chunk size (rows)",
    min_value=50_000,
    max_value=1_000_000,
    value=200_000,
    step=50_000,
    help="Bigger chunks are faster but use more RAM."
)

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
    pats = []
    for line in (lines or "").splitlines():
        line = line.strip()
        if line:
            pats.append(re.escape(line))
    if not pats:
        return None
    return re.compile("|".join(pats), flags=re.IGNORECASE)

def materialize_to_csv_path(uploaded) -> str:
    """
    Writes uploaded csv/gz/zip to a temp file.
    If zip: extracts first CSV inside and writes it out as a temp CSV.
    Returns a filesystem path we can stream-read in chunks.
    """
    name = (uploaded.name or "").lower()

    # If it's a zip, extract first CSV
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

    # Otherwise write raw bytes to temp (could be .csv or .gz)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=name[name.rfind("."):])
    with open(tmp.name, "wb") as f_out:
        f_out.write(uploaded.getbuffer())
    return tmp.name

if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    path = materialize_to_csv_path(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# -------------------------
# Filters (kept simple + high value)
# -------------------------
st.subheader("🎛️ Filters")

c1, c2, c3 = st.columns(3)
with c1:
    remove_self = st.checkbox("Remove self-referring (Source == Destination)", value=True)
with c2:
    exclude_params = st.checkbox("Exclude Destination with '?' (query params)", value=False)
with c3:
    only_hyperlinks = st.checkbox("Type = Hyperlink only (if column exists)", value=True)

st.markdown("### Include-only filters (applied if columns exist)")
link_position_keep = st.multiselect(
    "Keep Link Position values (leave empty to keep all)",
    options=["Content", "Navigation", "Header", "Footer", "Sidebar", "Body", "Main", "Other"],
    default=["Content"],
    help="If your export uses different labels, leave this empty or adjust options later."
)

follow_keep = st.multiselect(
    "Keep Follow values (leave empty to keep all)",
    options=["Follow", "Nofollow", "NoFollow", "nofollow"],
    default=["Follow"]
)

status_code_keep = st.multiselect(
    "Keep Status Code values (leave empty to keep all)",
    options=["200", "301", "302", "404", "410", "500"],
    default=["200"]
)

st.markdown("### Exclude by Link Path patterns (breadcrumbs/nav etc.)")
exclude_by_link_path = st.checkbox("Enable Link Path exclusions", value=False)
default_lp = "\n".join(["breadcrumb", "/ol/li", "aria-label=\"breadcrumb\"", "aria-label='breadcrumb'"])
lp_text = st.text_area("Patterns to exclude (one per line)", value=default_lp, height=100, disabled=not exclude_by_link_path)
rx_lp = compile_contains_patterns(lp_text) if exclude_by_link_path else None

# -------------------------
# Chunk processing aggregation
# -------------------------
st.subheader("🏆 Top Destination URLs")

run = st.button("🚀 Run analysis", type="primary")

if run:
    total_inlinks = {}          # Destination -> count rows
    unique_sources = {}         # Destination -> set(Source)  (OK for medium; can be heavy for huge)
    # If this set becomes too large on your biggest jobs, we can switch this to an approximate method later.

    progress = st.progress(0, text="Reading file in chunks...")

    read_kwargs = dict(
        chunksize=int(chunksize),
        low_memory=False,
        dtype="string",              # reduce dtype inference issues
        compression="infer",
    )

    # Encoding fallback
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

        # Ensure required cols are string
        chunk["Source"] = chunk["Source"].astype("string").fillna("")
        chunk["Destination"] = chunk["Destination"].astype("string").fillna("")

        # Filter: remove self
        if remove_self:
            src_norm = chunk["Source"].map(normalize_url_for_compare)
            dst_norm = chunk["Destination"].map(normalize_url_for_compare)
            chunk = chunk[src_norm != dst_norm]

        # Filter: exclude params
        if exclude_params:
            chunk = chunk[~chunk["Destination"].str.contains(r"\?", regex=True, na=False)]

        # Filter: Type=Hyperlink only
        if only_hyperlinks and "Type" in chunk.columns:
            chunk["Type"] = chunk["Type"].astype("string").fillna("")
            chunk = chunk[chunk["Type"].str.lower() == "hyperlink"]

        # Include-only: Link Position
        if link_position_keep and "Link Position" in chunk.columns:
            lp = chunk["Link Position"].astype("string").fillna("")
            chunk = chunk[lp.isin(link_position_keep)]

        # Include-only: Follow
        if follow_keep and "Follow" in chunk.columns:
            fl = chunk["Follow"].astype("string").fillna("")
            chunk = chunk[fl.isin(follow_keep)]

        # Include-only: Status Code
        if status_code_keep and "Status Code" in chunk.columns:
            sc = chunk["Status Code"].astype("string").fillna("")
            chunk = chunk[sc.isin(status_code_keep)]

        # Exclude Link Path patterns
        if rx_lp is not None and "Link Path" in chunk.columns:
            lps = chunk["Link Path"].astype("string").fillna("")
            chunk = chunk[~lps.str.contains(rx_lp, na=False)]

        if chunk.empty:
            if chunks_seen % 5 == 0:
                progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {rows_seen:,} rows...")
            continue

        # Aggregate totals + uniques
        dests = chunk["Destination"].tolist()
        srcs = chunk["Source"].tolist()

        for d, s in zip(dests, srcs):
            if not d:
                continue
            total_inlinks[d] = total_inlinks.get(d, 0) + 1
            # unique sources per destination
            if d not in unique_sources:
                unique_sources[d] = set()
            unique_sources[d].add(s)

        if chunks_seen % 5 == 0:
            progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {rows_seen:,} rows...")

    progress.progress(1.0, text=f"Done. Processed {rows_seen:,} rows.")

    # Build output
    out = pd.DataFrame({
        "Destination": list(total_inlinks.keys()),
        "Total_Inlinks": [total_inlinks[d] for d in total_inlinks.keys()],
        "Unique_Source_Pages": [len(unique_sources.get(d, set())) for d in total_inlinks.keys()],
    }).sort_values(["Total_Inlinks", "Unique_Source_Pages"], ascending=False)

    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
    out_top = out.head(int(top_n))

    st.dataframe(out_top, use_container_width=True)

    st.download_button(
        "Download top destinations CSV",
        data=out_top.to_csv(index=False).encode("utf-8"),
        file_name="top_destinations.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download full destination summary CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="destinations_summary.csv",
        mime="text/csv",
    )

with st.expander("Why the old version crashed"):
    st.markdown(
        """
Your old version loaded the full CSV into pandas, then filtering/groupby created extra copies in memory.
With 300–500MB CSVs, that often spikes into multiple GB and Streamlit Cloud containers crash.

This version avoids that by:
- reading in chunks
- filtering per chunk
- only storing aggregated results
"""
    )





