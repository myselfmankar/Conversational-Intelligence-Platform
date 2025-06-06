# whatsapp-chat-analyzer/utils/file_handler.py
import streamlit as st
import zipfile 
import io      

def handle_uploaded_file(uploaded_file):
    """
    Handles the uploaded file. For now, assumes .txt.
    """
    if uploaded_file is None:
        return None
        
    if uploaded_file.name.endswith(".txt"):
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Error decoding .txt file. Please ensure it's UTF-8 encoded.")
            return None
        except Exception as e:
            st.error(f"Error reading .txt file: {e}")
            return None

    elif uploaded_file.name.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), 'r') as zip_ref:
                # Assuming the first .txt file in the zip is the chat
                txt_files = [f for f in zip_ref.namelist() if f.lower().endswith('.txt')]
                if not txt_files:
                    st.error("No .txt file found in the uploaded .zip archive.")
                    return None
                # Read the first .txt file found
                with zip_ref.open(txt_files[0]) as txt_file:
                    return txt_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error processing .zip file: {e}")
            return None
    else:
        st.error("Unsupported file type. Please upload a WhatsApp exported .txt file.")
        return None