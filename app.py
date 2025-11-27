import streamlit as st
from retrieval import get_answer

st.set_page_config(page_title="Mini RAG Chat", layout="wide")
st.title("Mini RAG — Contract Q&A")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about the indexed documents:")

if st.button("Ask") or (query and st.session_state.get("auto_submit", False)):
    if not query.strip():
        st.warning("Enter a question")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                out = get_answer(query)
                st.session_state.history.append(out)
            except Exception as e:
                st.error(str(e))

# show conversation history
for entry in reversed(st.session_state.history):
    st.markdown("---")
    st.markdown(f"**Q:** {entry['question']}")
    st.markdown(f"**A:** {entry['answer']}")
    st.markdown("**Retrieved chunks (top 5)**")
    for i, c in enumerate(entry["retrieved_chunks"], start=1):
        md = c.get("metadata", {})
        st.markdown(f"- **Chunk {i}** — {md.get('contract_id')} | {md.get('chunk_id')} | section: {md.get('section_title')}")
        # show preview of chunk
        with st.expander("Preview chunk text"):
            st.write(c["text"][:2000])

