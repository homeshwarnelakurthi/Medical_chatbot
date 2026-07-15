# Data folder

Place the medical PDF(s) you want the chatbot to answer from in this folder,
for example `Medical_book.pdf` (The Gale Encyclopedia of Medicine).

Then build the Pinecone index once:

```bash
python store_index.py
```

PDFs are intentionally not committed to the repository (see `.gitignore`).
