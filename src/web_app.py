import streamlit as st
from PIL import Image
import io
import tempfile
import os
import pandas as pd

from predict_api import predict_from_path, get_overlay_from_path


st.set_page_config(page_title="CSIRO Biomass Predictor", layout="wide")


def load_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), image_bytes


def save_temp_image(image_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        temp_file.write(image_bytes)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()


@st.cache_data(show_spinner=False)
def load_metadata_df():
    metadata_path = os.path.join("data", "final_metadata_clean_FIXED.csv")
    if os.path.exists(metadata_path):
        try:
            return pd.read_csv(metadata_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def lookup_metadata(sample_id, metadata_df):
    if metadata_df.empty:
        return None
    row = metadata_df[metadata_df["image_path"].str.contains(sample_id, na=False)]
    if row.empty:
        return None
    row = row.iloc[0]
    metadata_fields = ["State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm", "year", "month", "day_of_year", "season"]
    return {field: row[field] for field in metadata_fields if field in row}


def preview_metadata(metadata):
    if metadata is None:
        st.info("No preview metadata found for this image.")
        return
    st.markdown("**Image metadata preview**")
    st.table(pd.DataFrame(list(metadata.items()), columns=["Field", "Value"]))


def display_predictions(preds):
    st.subheader("Biomass Predictions")
    df = pd.DataFrame({"Target": list(preds.keys()), "Value (g)": list(preds.values())})
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Target"))


def display_metadata(metadata):
    st.subheader("Auto Metadata")
    st.table(pd.DataFrame(list(metadata.items()), columns=["Field", "Value"]))


def main():
    st.title("CSIRO Biomass Predictor")
    st.markdown(
        "A small interactive dashboard for running biomass predictions and visualizing segmentation overlays."
    )

    metadata_df = load_metadata_df()

    with st.sidebar:
        st.header("Input")
        input_mode = st.radio("Choose input type", ["Upload Image", "Local Path"])

        uploaded = None
        local_path = ""

        if input_mode == "Upload Image":
            uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        else:
            local_path = st.text_input("Path to image", "data/test/sample.jpg")
            if local_path:
                sample_id = os.path.splitext(os.path.basename(local_path))[0]
                metadata_preview = lookup_metadata(sample_id, metadata_df)
                with st.expander("Preview metadata for local path", expanded=False):
                    preview_metadata(metadata_preview)

        st.header("Overlay")
        show_overlay = st.checkbox("Show mask overlay", value=True)

        st.markdown("---")
        run_prediction = st.button("Run prediction")

    input_error = None
    image_path = None
    image = None

    if run_prediction:
        if input_mode == "Upload Image":
            if uploaded is None:
                input_error = "Please upload an image file."
            else:
                image, image_bytes = load_uploaded_image(uploaded)
                image_path = save_temp_image(image_bytes)
        elif input_mode == "Local Path":
            if not local_path:
                input_error = "Please enter a valid local image path."
            else:
                image_path = local_path
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as exc:
                    input_error = f"Unable to open local image: {exc}"

        if input_error:
            st.sidebar.error(input_error)
            return

        with st.spinner("Running prediction..."):
            try:
                metadata, preds = predict_from_path(image_path)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            overlay_image = None
            if show_overlay:
                try:
                    overlay_image = get_overlay_from_path(image_path, alpha=0.5)
                except Exception:
                    overlay_image = None

        image_tab, overlay_tab, details_tab = st.tabs(["Input Image", "Mask Overlay", "Prediction Details"])

        with image_tab:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)

        with overlay_tab:
            if overlay_image is not None:
                st.subheader("Mask Overlay")
                st.image(overlay_image, use_column_width=True)
            else:
                st.info("Overlay unavailable. Make sure the SAM model is loaded and the image path is valid.")

        with details_tab:
            display_metadata(metadata)
            display_predictions(preds)

        if input_mode == "Upload Image" and image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
