from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
##split the text documents into chunks so that we can further creat embeddings

def clean_page_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()

        # keep empty lines but compress later
        if not s:
            cleaned.append("")
            continue

        # plain page numbers: "3"
        if re.fullmatch(r"\d{1,3}", s):
            continue

        # "Page 12", "Page 12 of 123", "p. 3/10"
        if re.fullmatch(r"(page|p\.)\s*\d+(\s*(/|of)\s*\d+)?",
                        s, flags=re.IGNORECASE):
            continue

        # tiny non-text junk like "---", "•", "1/3"
        if len(s) <= 4 and not re.search(r"[A-Za-z]", s):
            continue

        cleaned.append(line)

    # collapse many blank lines
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_docs(documents,chunk_size,chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = [
            "\n```",      
            "\n#include", 
            "\ndef ",       
            "\nclass ",     
            "\n## ",
            "\n### ",
            "\n\n",
            "\n- ",
            "\n* ",
            ". ",      
            "\n",
            " ",
            ""
        ]
    )

    splitted_text = text_splitter.split_documents(documents)
    return splitted_text


