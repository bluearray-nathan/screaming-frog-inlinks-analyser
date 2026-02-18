# app.py
# Streamlit Cloud–ready: Screaming Frog "All Inlinks" CSV filter + Top Destinations
#
# Updates:
# - Users can upload: .csv, .csv.gz, or .zip (Mac Finder "Compress" output)
# - If .zip: app extracts the first .csv inside and reads it
# - Deduplication happens AFTER all filters (including column filters)

import re
import zipfile
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st

st.set_page_config(page_title="All Inlinks Internal Link Analyzer", layout="wide")

st.title("🔗 Screaming Frog All Inlinks Analyzer")
st.write(
    "Upload a **Screaming Frog → All Inlinks** file (`.csv`, `.csv.gz`, or `.zip`), apply filters, "
    "then view the **most internally linked-to pages**."
)

# ---------------------------
# Upload + loading
# ---------------------------
uploaded_file = st.file_uploader("Upload your All Inlinks file", type=["csv", "gz", "zip"])

st.sidebar.header("⚙️ Settings")
force_string_dtypes = st.sidebar.checkbox(
    "Load all columns as text (recommended for large files)",
    value=True,
    help="Prevents dtype inference issues and usually reduces memory surprises."
)
show_raw_preview = st.sidebar.checkbox("Show raw preview table", value=False)

@st.cache_data(show_spinner=False)
def load_csv(uploaded, force_str: bool) -> pd.DataFrame:
    """
    Load CSV safely for Streamlit Cloud.
    Supports:
      - .csv
      - .csv.gz  (pandas compression='infer')
      - .zip     (extracts first .csv found inside)

    Tries UTF-8 then CP1252 fallback.
    """
    read_kwargs = dict(low_memory=False, compression="infer")
    if force_str:
        read_kwargs["dtype"] = "string"

    filename = (uploaded.name or "").lower()

    # --- ZIP handling (Mac Finder "Compress" creates .zip) ---
    if filename.endswith(".zip"):
        # Streamlit UploadedFile is file-like; zipfile can open it directly
        uploaded.seek(0)
        with zipfile.ZipFile(uploaded) as z:
            # Pick the first CSV-like file inside
            members = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ValueError("No .csv file found inside the uploaded .zip.")
            csv_name = members[0]

            with z.open(csv_name) as f:
                # f is a file-like object (bytes). pandas can read it directly.
                try:
                    return pd.read_csv(f, **read_kwargs)
                except UnicodeDecodeError:
                    # reopen stream for second attempt (since first read consumed it)
                    with z.open(csv_name) as f2:
                        return pd.read_csv(f2, encoding="cp1252", **read_kwargs)

    # --- Normal CSV / GZ handling ---
    uploaded.seek(0)
    try:
        return pd.read_csv(uploaded, **read_kwargs)
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="cp1252", **read_kwargs)

def normalize_url_for_compare(u: str) -> str:
    """
    Light URL normalization for comparing Source vs Destination:
    - lowercases scheme + host
    - removes fragment (#...)
    - trims trailing slash in path (except root '/')
    Keeps query params.
    """
    if u is None or pd.isna(u):
        return ""
    u = str(u).strip()
    if not u:
        return ""

    parts = urlsplit(u)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or ""

    # trim trailing slash (but keep "/" root)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # drop fragment
    return urlunsplit((scheme, netloc, path, parts.query, ""))

def compile_contains_patterns(lines: str) -> re.Pattern | None:
    """Each non-empty line is treated as a substring match (escaped), OR'd together, case-insensitive."""
    pats = []
    for line in (lines or "").splitlines():
        line = line.strip()
        if not line:
            continue
        pats.append(re.escape(line))
    if not pats:
        return None
    return re.compile("|".join(pats), flags=re.IGNORECASE)

def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")

FILTER_COLUMNS = [
    "Type",
    "Follow",
    "Status Code",
    "Status",
    "Link Position",
    "Link Origin",
    "Target",
    "Rel",
    "Path Type",
]

# ---------------------------
# App main
# ---------------------------
if uploaded_file is None:
    st.info("Upload an All Inlinks file to begin.")
    st.stop()

try:
    df = load_csv(uploaded_file, force_string_dtypes)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.success(f"✅ File loaded: **{df.shape[0]:,} rows** × **{df.shape[1]:,} columns**")

if show_raw_preview:
    with st.expander("Preview raw data"):
        st.dataframe(df.head(50), use_container_width=True)

missing_core = [c for c in ["Source", "Destination"] if c not in df.columns]
if missing_core:
    st.error(f"Missing expected column(s): {', '.join(missing_core)}")
    st.stop()

filtered_df = df.copy()

# ---------------------------
# Quick actions
# ---------------------------
st.subheader("⚡ Quick actions")

qa1, qa2, qa3 = st.columns([1, 1, 2])

with qa1:
    remove_self = st.checkbox(
        "Remove self-referring links",
        value=False,
        help="Removes rows where Source and Destination are the same URL (after removing fragments & trailing slashes)."
    )
with qa2:
    dedupe_pairs = st.checkbox(
        "Deduplicate Source→Destination pairs (after all filters)",
        value=False,
        help="Applies deduplication after all filters (including column filters) have been applied."
    )
with qa3:
    exclude_params = st.checkbox(
        "Exclude URLs with query parameters",
        value=False,
        help="Removes rows where Destination contains '?'. Helpful for faceted/sort/filter URLs."
    )

# Link Path pattern exclusion
st.subheader("🧭 Exclude breadcrumb / structural navigation via Link Path patterns")

exclude_breadcrumbs = st.checkbox(
    "Enable Link Path pattern exclusions (breadcrumbs, etc.)",
    value=False,
    help="Matches substrings against Link Path (and optionally Link Origin)."
)

default_patterns = "\n".join([
    "breadcrumb",
    "/ol/li",
    "aria-label=\"breadcrumb\"",
    "aria-label='breadcrumb'",
])

patterns_text = st.text_area(
    "Patterns to EXCLUDE (one per line, matched case-insensitively)",
    value=default_patterns,
    height=120,
    disabled=not exclude_breadcrumbs
)

apply_patterns_to_origin = st.checkbox(
    "Also apply patterns to Link Origin (if present)",
    value=False,
    disabled=not exclude_breadcrumbs
)

show_excluded_preview = st.checkbox(
    "Show a preview of excluded rows",
    value=False,
    disabled=not exclude_breadcrumbs
)

# ---------------------------
# Apply non-column filters
# ---------------------------
if remove_self:
    before = len(filtered_df)
    src_norm = filtered_df["Source"].map(normalize_url_for_compare)
    dst_norm = filtered_df["Destination"].map(normalize_url_for_compare)
    filtered_df = filtered_df[src_norm != dst_norm].copy()
    st.info(f"Removed **{before - len(filtered_df):,}** self-referring rows.")

if exclude_params:
    before = len(filtered_df)
    dst = safe_str_series(filtered_df["Destination"])
    filtered_df = filtered_df[~dst.str.contains(r"\?", regex=True)].copy()
    st.info(f"Removed **{before - len(filtered_df):,}** rows with query parameters in Destination.")

if exclude_breadcrumbs:
    if "Link Path" not in filtered_df.columns:
        st.warning("No **Link Path** column found, so Link Path exclusions can’t be applied.")
    else:
        rx = compile_contains_patterns(patterns_text)
        if rx is None:
            st.info("No patterns provided, so nothing will be excluded by Link Path.")
        else:
            lp = safe_str_series(filtered_df["Link Path"])
            mask = lp.str.contains(rx, na=False)

            if apply_patterns_to_origin and "Link Origin" in filtered_df.columns:
                lo = safe_str_series(filtered_df["Link Origin"])
                mask = mask | lo.str.contains(rx, na=False)

            excluded_count = int(mask.sum())
            if show_excluded_preview and excluded_count > 0:
                with st.expander(f"Preview excluded rows ({excluded_count:,} matched)"):
                    st.dataframe(filtered_df[mask].head(100), use_container_width=True)

            filtered_df = filtered_df[~mask].copy()
            st.info(f"Excluded **{excluded_count:,}** rows matching Link Path/Origin patterns.")

# ---------------------------
# Column include-filters
# ---------------------------
st.subheader("🎛️ Column filters (include-only)")

for col in FILTER_COLUMNS:
    if col in filtered_df.columns:
        ser = filtered_df[col].astype("string")
        vals = ser.fillna("").unique().tolist()

        display_vals = []
        for v in vals:
            v = "" if v is None else str(v)
            display_vals.append("(blank)" if v.strip() == "" else v)

        pairs = sorted(
            set((dv, "" if dv == "(blank)" else dv) for dv in display_vals),
            key=lambda x: x[0].lower()
        )
        options_display = [p[0] for p in pairs]
        display_to_real = {p[0]: p[1] for p in pairs}

        selected_display = st.multiselect(
            f"Include values for **{col}**",
            options=options_display,
            default=options_display
        )

        if len(selected_display) == 0:
            filtered_df = filtered_df.iloc[0:0].copy()
        else:
            selected_real = set(display_to_real[d] for d in selected_display)
            filtered_df = filtered_df[
                filtered_df[col].astype("string").fillna("").isin(selected_real)
            ].copy()

# ✅ Dedupe AFTER all filters (including column filters)
if dedupe_pairs and not filtered_df.empty:
    before = len(filtered_df)
    filtered_df = filtered_df.drop_duplicates(subset=["Source", "Destination"]).copy()
    st.info(f"Removed **{before - len(filtered_df):,}** duplicate Source→Destination rows (after all filters).")

st.success(f"✅ Final dataset: **{filtered_df.shape[0]:,} rows**")

# ---------------------------
# Output: Most linked-to Destination URLs
# ---------------------------
st.subheader("🏆 Most linked-to Destination URLs")

if filtered_df.empty:
    st.warning("No rows left after filtering.")
else:
    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)

    top_links = (
        filtered_df.groupby("Destination", dropna=False)
        .agg(
            Total_Inlinks=("Destination", "count"),
            Unique_Source_Pages=("Source", "nunique"),
        )
        .sort_values(["Total_Inlinks", "Unique_Source_Pages"], ascending=False)
        .head(int(top_n))
        .reset_index()
    )

    st.dataframe(top_links, use_container_width=True)

# ---------------------------
# Downloads
# ---------------------------
st.subheader("⬇️ Downloads")

c1, c2 = st.columns(2)

with c1:
    st.download_button(
        label="Download filtered links CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_all_inlinks.csv",
        mime="text/csv",
        disabled=filtered_df.empty,
    )

with c2:
    if not filtered_df.empty:
        st.download_button(
            label="Download top destinations CSV",
            data=top_links.to_csv(index=False).encode("utf-8"),
            file_name="top_destinations.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            label="Download top destinations CSV",
            data="Destination,Total_Inlinks,Unique_Source_Pages\n".encode("utf-8"),
            file_name="top_destinations.csv",
            mime="text/csv",
            disabled=True,
        )

with st.expander("How users create a .zip on Mac (no Terminal)"):
    st.markdown(
        """
1. In Finder, right-click the CSV  
2. Choose **Compress "filename.csv"**  
3. Upload the resulting **.zip** file here
"""
    )




