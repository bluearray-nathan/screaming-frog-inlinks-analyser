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

def materialize_to_path(uploaded) -> str:
    """
    Writes uploaded csv/gz/zip to a temp file.
    If zip: extracts first CSV inside -> temp CSV path.
    Otherwise writes raw bytes -> temp file (keeps extension).
    """
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

if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    path = materialize_to_path(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# --- sample for real options ---
sample_df = read_sample(path, int(sample_rows))

required = {"Source", "Destination"}
if not required.issubset(sample_df.columns):
    st.error(f"Missing required columns: {', '.join(sorted(required - set(sample_df.columns)))}")
    st.stop()

def uniq(col: str):
    if col not in sample_df.columns:
        return []
    vals = sample_df[col].astype("string").fillna("").unique().tolist()
    # nice sort: blanks last
    vals = sorted(vals, key=lambda x: (x == "" or x is None, str(x).lower()))
    return vals

type_options = uniq("Type")
follow_options = uniq("Follow")
status_options = uniq("Status Code")
linkpos_options = uniq("Link Position")

st.subheader("🎛️ Filters")

# Quick preset button
preset = st.button("✨ Preset: Content + Follow + 200 + Hyperlink")

c1, c2, c3 = st.columns(3)
with c1:
    remove_self = st.checkbox("Remove self-referring (Source == Destination)", value=True)
with c2:
    exclude_params = st.checkbox("Exclude Destination with '?' (query params)", value=True)
with c3:
    st.caption("Tip: leave include-only filters empty to keep all values.")

# Include-only filters (dynamic options from sample)
# Empty selection means "keep all"
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
lp_text = st.text_area("Patterns to exclude (one per line)", value=default_lp, height=100, disabled=not exclude_by_link_path)
rx_lp = compile_contains_patterns(lp_text) if exclude_by_link_path else None

run = st.button("🚀 Run analysis", type="primary")

if run:
    total_inlinks = {}      # Destination -> count
    unique_sources = {}     # Destination -> set(Source)  (OK for many cases; if this becomes too big, we can swap later)

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

        chunk["Source"] = chunk["Source"].astype("string").fillna("")
        chunk["Destination"] = chunk["Destination"].astype("string").fillna("")

        before_chunk = len(chunk)

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

        # link path exclusions
        if rx_lp is not None and "Link Path" in chunk.columns:
            lps = chunk["Link Path"].astype("string").fillna("")
            chunk = chunk[~lps.str.contains(rx_lp, na=False)]

        # aggregate
        if not chunk.empty:
            for d, s in zip(chunk["Destination"].tolist(), chunk["Source"].tolist()):
                if not d:
                    continue
                total_inlinks[d] = total_inlinks.get(d, 0) + 1
                unique_sources.setdefault(d, set()).add(s)

        # progress
        if chunks_seen % 5 == 0:
            progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {rows_seen:,} rows...")

    progress.progress(1.0, text=f"Done. Processed {rows_seen:,} rows.")

    if not total_inlinks:
        st.warning(
            "No rows matched your filters. Most likely cause: your include-only selections don’t match the file values.\n\n"
            "Try clearing the include-only filters (set them to empty) and run again."
        )
        st.stop()

    out = pd.DataFrame({
        "Destination": list(total_inlinks.keys()),
        "Total_Inlinks": [total_inlinks[d] for d in total_inlinks.keys()],
        "Unique_Source_Pages": [len(unique_sources.get(d, set())) for d in total_inlinks.keys()],
    }).sort_values(["Total_Inlinks", "Unique_Source_Pages"], ascending=False)

    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
    out_top = out.head(int(top_n))

    st.subheader("🏆 Top Destination URLs")
    st.dataframe(out_top, use_container_width=True)

    st.download_button(
        "Download top destinations CSV",
        data=out_top.to_csv(index=False).encode("utf-8"),
        file_name="top_destinations.csv",
        mime="text/csv",
    )





