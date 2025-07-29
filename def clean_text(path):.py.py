def clean_text(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Merge lines that shouldn't break
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)  # replaces single \n with space
    return cleaned


path = "/Users/clara-lea/Documents/GitHub/pharma-rag/input-datasets/books/21_CFR_Part_211.txt"
clean-text