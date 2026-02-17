import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="All Inlinks Internal Link Analyzer",
    layout="wide"
)

st.title("🔗 Screaming Frog All Inlinks Analyzer")
st.write(
    """
Upload your **Screaming Frog → All Inlinks CSV**, filter down to only the links you care about,
and see the **most internally linked-to pages**.
"""
)

# --- Upload CSV ---
uploaded_file = st.file_uploader(
    "Upload your All Inlinks CSV file",
    type=["csv"]
)

# Columns we want to offer filtering for (only if present)
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
def load_csv(file):
    """Load CSV safely for Streamlit Cloud."""
    try:
        return pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="cp1252", low_memory=False)

if uploaded_file:

    # --- Load Data ---
    df = load_csv(uploaded_file)

    st.success(f"✅ File uploaded successfully: {df.shape[0]:,} rows")

    # --- Preview ---
    with st.expander("Preview raw data"):
        st.dataframe(df.head(30), use_container_width=True)

    # --- Filtering UI ---
    st.subheader("🎛️ Filter Links")

    filtered_df = df.copy()

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

            if selected:
                filtered_df = filtered_df[
                    filtered_df[col].astype(str).isin(selected)
                ]

    st.info(f"Filtered dataset: **{filtered_df.shape[0]:,} rows**")

    # --- Output: Most Linked-To URLs ---
    st.subheader("🏆 Most Linked-To Destination URLs")

    if "Destination" not in filtered_df.columns:
        st.error("⚠️ No 'Destination' column found in this CSV.")
    else:
        top_links = (
            filtered_df.groupby("Destination")
            .agg(
                Total_Inlinks=("Source", "count"),
                Unique_Source_Pages=("Source", "nunique")
            )
            .sort_values("Total_Inlinks", ascending=False)
            .head(50)
            .reset_index()
        )

        st.dataframe(top_links, use_container_width=True)

    # --- Download filtered CSV ---
    st.subheader("⬇️ Download Filtered Output")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Links CSV",
        data=csv_data,
        file_name="filtered_all_inlinks.csv",
        mime="text/csv"
    )
