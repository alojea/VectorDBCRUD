import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "local_documents_ui"

# Create collection if not exists
if collection_name not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )

st.title("📄 Qdrant Document Uploader, Search, Delete & Modify")

# -------------------------------
# Upload file
# -------------------------------
uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("File Content", content, height=200)

    if st.button("Upload to Qdrant"):
        vector = np.random.rand(4)  # Replace with real embeddings for semantic search
        point_id = np.random.randint(1000, 9999)

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"content": content}
                )
            ]
        )

        st.success(f"✅ Document uploaded with ID {point_id}")

# -------------------------------
# Search
# -------------------------------
st.subheader("🔎 Search Documents")

query_text = st.text_input("Enter search text")

if st.button("Search"):
    query_vector = np.random.rand(4)  # Replace with real embedding
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
    )

    if len(search_result) == 0:
        st.warning("No matching documents found.")
    else:
        for point in search_result:
            with st.expander(f"Document ID: {point.id} | Score: {point.score:.2f}"):
                st.write(point.payload["content"])

# -------------------------------
# Show and manage all documents
# -------------------------------
st.subheader("📄 All Stored Documents")

scroll_result, _ = client.scroll(collection_name=collection_name, limit=50)

for point in scroll_result:
    with st.expander(f"Document ID: {point.id}"):
        st.write(point.payload["content"])

        col1, col2 = st.columns(2)

        # Delete button
        if col1.button(f"Delete {point.id}"):
            client.delete(collection_name=collection_name, points_selector=models.PointIdsList(points=[point.id]))
            st.success(f"🗑️ Deleted document ID {point.id}")
            st.experimental_rerun()  # Refresh UI

        # Modify button
        if col2.button(f"Modify {point.id}"):
            new_content = st.text_area(f"New content for ID {point.id}", point.payload["content"], key=f"modify_{point.id}")
            if st.button(f"Save {point.id}"):
                # Generate new vector (demo)
                new_vector = np.random.rand(4)
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=point.id,
                            vector=new_vector,
                            payload={"content": new_content}
                        )
                    ]
                )
                st.success(f"✅ Updated document ID {point.id}")
                st.experimental_rerun()  # Refresh UI

