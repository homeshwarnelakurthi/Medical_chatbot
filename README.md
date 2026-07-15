# Medical Chatbot — LLMs + LangChain + Pinecone + Flask + AWS

An end-to-end medical question-answering chatbot built with **retrieval-augmented
generation (RAG)**. Medical PDFs are chunked and embedded with
sentence-transformers, stored in a **Pinecone** vector index, and served through a
**Flask** chat UI where an LLM answers questions grounded in the retrieved context.

## How it works

```
data/*.pdf ──> load & split ──> embeddings (all-MiniLM-L6-v2) ──> Pinecone index
                                                                       │
user question ──> Flask app ──> retriever (top-k chunks) ──> LLM ──> answer
```

| Component  | Technology                                     |
| ---------- | ---------------------------------------------- |
| LLM        | OpenAI `gpt-4o-mini` (via `langchain-openai`)  |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384d)|
| Vector DB  | Pinecone (serverless)                          |
| Framework  | LangChain 0.3                                  |
| Web app    | Flask                                          |
| Deployment | Docker + GitHub Actions + AWS (ECR + EC2)      |

## Project structure

```
├── app.py                  # Flask app serving the chat UI + RAG chain
├── store_index.py          # One-time script: embed PDFs and build the Pinecone index
├── src/
│   ├── helper.py           # PDF loading, chunking, embedding helpers
│   └── prompt.py           # System prompt for the RAG chain
├── templates/chat.html     # Chat interface
├── static/style.css        # Chat interface styles
├── research/trials.ipynb   # Step-by-step notebook of the whole pipeline
├── data/                   # Put your medical PDF(s) here (not committed)
├── Dockerfile
└── .github/workflows/      # CI + AWS deployment pipelines
```

## How to run locally

### STEP 00 — Clone the repository

```bash
git clone https://github.com/homeshwarnelakurthi/Medical_chatbot.git
cd Medical_chatbot
```

### STEP 01 — Create and activate a conda environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### STEP 02 — Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 03 — Add your API keys

Copy `.env.example` to `.env` and fill in your keys:

```ini
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
```

### STEP 04 — Add data and build the vector index

Place your medical PDF(s) (e.g. `Medical_book.pdf`) in the `data/` folder, then:

```bash
python store_index.py
```

This creates the `medical-chatbot` Pinecone index and upserts all chunk
embeddings (only needs to be run once, or whenever the data changes).

### STEP 05 — Run the app

```bash
python app.py
```

Open http://localhost:8080 and start chatting.

## Run with Docker

```bash
docker build -t medical-chatbot .
docker run -p 8080:8080 --env-file .env medical-chatbot
```

## AWS deployment (CI/CD)

The [aws-deploy workflow](.github/workflows/aws-deploy.yaml) builds the Docker
image, pushes it to Amazon ECR, and runs it on an EC2 instance registered as a
GitHub self-hosted runner.

1. **IAM user** with `AmazonEC2ContainerRegistryFullAccess` and
   `AmazonEC2FullAccess` policies.
2. **ECR repository** (e.g. `medical-chatbot`) — note the repository name.
3. **EC2 instance** (Ubuntu) with Docker installed:
   ```bash
   sudo apt-get update -y && sudo apt-get upgrade -y
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```
4. **Self-hosted runner**: repo → Settings → Actions → Runners → New self-hosted
   runner, and run the setup commands on the EC2 instance.
5. **Repository secrets** (Settings → Secrets and variables → Actions):
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_DEFAULT_REGION` (e.g. `us-east-1`)
   - `ECR_REPO` (e.g. `medical-chatbot`)
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
6. Trigger the **Deploy to AWS** workflow from the Actions tab (once everything
   is configured you can switch its trigger to run on every push to `main`).

## Disclaimer

This chatbot is for educational purposes only and is **not** a substitute for
professional medical advice, diagnosis, or treatment.

## License

Licensed under the [Apache License 2.0](LICENSE).
