import streamlit as st
import pandas as pd
from urllib.parse import urlsplit, urlunsplit

st.set_page_config(page_title="All Inlinks Internal Link Analyzer", layout="wide")

st.title("🔗 Screaming Frog All Inlinks Analyzer")
st.write(
    """
Upload your **Screaming Frog → All Inlinks CSV**, filter down to only the links you care about,
and see the **most internally linked-to pages**.
"""
)

uploaded_file = st.file_uploader("Upload your All Inlinks CSV file", type=["csv"])

FILTER_COLUMNS = [
    "Type",
    "Follow",
    "Status Code",
    "Link Position",
    "Link Origin",
    "Target",
    "Rel",
    "Path Type",
]

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """Load CSV safely for Streamlit Cloud."""
    try:
        return pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="cp1252", low_memory=False)

def normalize_url_for_compare(u: str) -> str:
    """
    Light URL normalization for comparing Source vs Destination:
    - lowercases scheme + host
    - removes fragment (#...)
    - removes trailing slash on path (except root)
    Keeps query params by default (because ?x=y can be a real different URL).
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

    # drop fragment
    fragment = ""

    # trim trailing slash (but keep "/" root)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((scheme, netloc, path, parts.query, fragment))

if uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully: {df.shape[0]:,} rows")

    with st.expander("Preview raw data"):
        st.dataframe(df.head(30), use_container_width=True)

    # --- Quick action buttons / toggles ---
    st.subheader("⚡ Quick actions")

    colA, colB = st.columns([1, 2])
    with colA:
        remove_self = st.checkbox(
            "Remove self-referring links (Source == Destination)",
            value=False
        )
    with colB:
        st.caption(
            "Self-referring links are rows where Source and Destination are the same URL "
            "(after removing fragments and trailing slashes for comparison)."
        )

    # --- Filtering UI ---
    st.subheader("🎛️ Filter Links")

    filtered_df = df.copy()

    # Apply quick action: remove self-referring links
    if remove_self:
        if "Source" in filtered_df.columns and "Destination" in filtered_df.columns:
            src_norm = filtered_df["Source"].map(normalize_url_for_compare)
            dst_norm = filtered_df["Destination"].map(normalize_url_for_compare)
            before = len(filtered_df)
            filtered_df = filtered_df[src_norm != dst_norm].copy()
            st.info(f"Removed {before - len(filtered_df):,} self-referring rows.")
        else:
            st.warning("Couldn't remove self-referring links because 'Source' and/or 'Destination' columns are missing.")

    # Value include filters (multiselect)
    for col in FILTER_COLUMNS:
        if col in filtered_df.columns:
            unique_values = (
                filtered_df[col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            unique_values.sort()

            selected = st.multiselect(
                f"Include only these values for **{col}**:",
                options=unique_values,
                default=unique_values
            )

            # If user deselects everything, show none (empty filter result)
            if len(selected) == 0:
                filtered_df = filtered_df.iloc[0:0]
            else:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    st.info(f"Filtered dataset: **{filtered_df.shape[0]:,} rows**")

    # --- Output: Most Linked-To URLs ---
    st.subheader("🏆 Most Linked-To Destination URLs")

    if "Destination" not in filtered_df.columns:
        st.error("⚠️ No 'Destination' column found in this CSV.")
    else:
        source_col = "Source" if "Source" in filtered_df.columns else None

        if source_col:
            top_links = (
                filtered_df.groupby("Destination")
                .agg(
                    Total_Inlinks=("Destination", "count"),
                    Unique_Source_Pages=(source_col, "nunique")
                )
                .sort_values("Total_Inlinks", ascending=False)
                .head(50)
                .reset_index()
            )
        else:
            top_links = (
                filtered_df.groupby("Destination")
                .size()
                .sort_values(ascending=False)
                .head(50)
                .reset_index(name="Total_Inlinks")
            )

        st.dataframe(top_links, use_container_width=True)

    # --- Download filtered CSV ---
    st.subheader("⬇️ Download Filtered Output")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Links CSV",
        data=csv_data,
        file_name="filtered_all_inlinks.csv",
        mime="text/csv",
    )

