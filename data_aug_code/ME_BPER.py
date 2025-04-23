import fitz

B_PER = []
with fitz.open("../data_aug_sources/Ordbog_over_muslimske_fornavne_i_DK.pdf") as doc:
    for page in doc: # Iterating through all pages
        blocks = page.get_text("dict")["blocks"] # Extracting text on each page
        for block in blocks: 
            for line in block["lines"]:  # Iterating through the text lines
                for span in line["spans"]:  # Iterating through the text spans
                    if span["flags"] & 16:  # 16 targets bold text
                        name = span["text"].strip()
                        if name:
                            B_PER.append(name)

ME_BPER = [name.replace("*", "") for name in B_PER]
ME_BPER = list(set(ME_BPER))