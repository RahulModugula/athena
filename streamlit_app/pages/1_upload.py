import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Upload Documents", page_icon="📄", layout="wide")
st.title("📄 Upload Documents")

st.markdown("Upload PDFs, text files, or Markdown documents to index them for search.")

with st.form("upload_form"):
    uploaded_file = st.file_uploader(
        "Select a document",
        type=["pdf", "txt", "md"],
        help="Maximum file size: 50MB",
    )
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=["recursive", "fixed", "semantic"],
        index=0,
        help="recursive: best general purpose | fixed: simple overlap | semantic: embedding-based",
    )
    submit = st.form_submit_button("Upload & Ingest", type="primary")

if submit and uploaded_file:
    with st.spinner(f"Ingesting {uploaded_file.name} with {chunking_strategy} chunking..."):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{API_URL}/api/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"chunking_strategy": chunking_strategy},
                )
            if resp.status_code == 200:
                doc = resp.json()
                st.success(f"✅ Ingested **{doc['filename']}** — {doc['chunk_count']} chunks created")
            elif resp.status_code == 409:
                st.warning("Document already exists in the index.")
            else:
                st.error(f"Upload failed: {resp.json().get('detail', 'unknown error')}")
        except Exception as e:
            st.error(f"Connection error: {e}")

st.divider()
st.subheader("Indexed Documents")

try:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{API_URL}/api/documents")
    docs = resp.json()
    if docs:
        for doc in docs:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            col1.write(f"📄 **{doc['filename']}**")
            col2.write(doc["mime_type"].split("/")[-1].upper())
            col3.write(f"{doc.get('chunk_count', '?')} chunks")
            if col4.button("🗑️", key=f"del_{doc['id']}"):
                with httpx.Client(timeout=5.0) as client:
                    client.delete(f"{API_URL}/api/documents/{doc['id']}")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")
except Exception as e:
    st.error(f"Could not fetch documents: {e}")
