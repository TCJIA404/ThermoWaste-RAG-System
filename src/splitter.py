# src/splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(pages, chunk_size=800, chunk_overlap=150):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page in pages:

        texts = splitter.split_text(page["text"])

        for text in texts:
            chunks.append({
                "text": text,
                "page": page["page"],
                "source": page["source"]
            })

    return chunks