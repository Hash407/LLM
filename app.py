import sys
import os
import faiss
import pickle
import numpy as np
import ollama
import random
import nltk

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QLabel, QFileDialog, QLineEdit
)

# ---------------------------
# NLTK SETUP
# ---------------------------
NLTK_PATH = "./nltk_data"
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.append(NLTK_PATH)

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet", download_dir=NLTK_PATH)
    nltk.download("omw-1.4", download_dir=NLTK_PATH)

from nltk.corpus import wordnet


# ---------------------------
# EMBEDDING
# ---------------------------
def get_embedding(text):
    res = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return np.array(res["embedding"], dtype="float32")


# ---------------------------
# MAIN APP
# ---------------------------
class HealthRAGApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Physical Health RAG System")
        self.resize(700, 600)

        layout = QVBoxLayout()

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask about physical health...")

        # Buttons
        btn_load_data = QPushButton("Load data.txt")
        btn_build_index = QPushButton("Build FAISS Index")
        btn_upgrade = QPushButton("Upgrade Index")
        btn_query = QPushButton("Ask Question")
        btn_test = QPushButton("Run Test Agent")

        # Connect
        btn_load_data.clicked.connect(self.load_data)
        btn_build_index.clicked.connect(self.build_index)
        btn_upgrade.clicked.connect(self.upgrade_index)
        btn_query.clicked.connect(self.ask_question)
        btn_test.clicked.connect(self.run_tests)

        # Layout
        layout.addWidget(QLabel("Physical Health RAG Dashboard"))
        layout.addWidget(self.query_input)
        layout.addWidget(btn_query)
        layout.addWidget(btn_load_data)
        layout.addWidget(btn_build_index)
        layout.addWidget(btn_upgrade)
        layout.addWidget(btn_test)
        layout.addWidget(self.log)

        self.setLayout(layout)

        self.chunks = []

    # ---------------------------
    def log_msg(self, msg):
        self.log.append(msg)
        print(msg)

    # ---------------------------
    def load_data(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select data.txt")

        if not file:
            return

        with open(file, "r", encoding="utf-8") as f:
            self.chunks = [l.strip() for l in f if l.strip()]

        self.log_msg(f"✅ Loaded {len(self.chunks)} chunks")

    # ---------------------------
    def build_index(self):
        if not self.chunks:
            self.log_msg("❌ Load data first!")
            return

        embeddings = []

        for i, chunk in enumerate(self.chunks):
            embeddings.append(get_embedding(chunk))

            if i % 50 == 0:
                self.log_msg(f"Embedding {i}")

        embeddings = np.array(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, "health.index")

        with open("chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        self.log_msg("✅ Index built successfully!")

    # ---------------------------
    def upgrade_index(self):
        if not os.path.exists("chunks.pkl"):
            self.log_msg("❌ Build base index first!")
            return

        file, _ = QFileDialog.getOpenFileName(self, "Select knowledgebase.txt")
        if not file:
            return

        with open("chunks.pkl", "rb") as f:
            old_chunks = pickle.load(f)

        with open(file, "r", encoding="utf-8") as f:
            new_chunks = [l.strip() for l in f if l.strip()]

        all_chunks = old_chunks + new_chunks

        self.log_msg(f"Upgrading with {len(new_chunks)} new entries")

        embeddings = []

        for i, chunk in enumerate(all_chunks):
            embeddings.append(get_embedding(chunk))

            if i % 50 == 0:
                self.log_msg(f"Embedding {i}")

        embeddings = np.array(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, "health_v2.index")

        with open("chunks_v2.pkl", "wb") as f:
            pickle.dump(all_chunks, f)

        self.log_msg("✅ Upgraded index created!")

    # ---------------------------
    def ask_question(self):
        query = self.query_input.text()

        if not query.strip():
            self.log_msg("❌ Empty query!")
            return

        if not os.path.exists("health.index"):
            self.log_msg("❌ Build index first!")
            return

        index = faiss.read_index("health.index")

        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        q_vec = get_embedding(query)

        D, I = index.search(np.array([q_vec]), k=3)

        results = [chunks[i] for i in I[0]]

        # Convert distance → similarity
        similarities = [1 / (1 + d) for d in D[0]]

        context = "\n".join(results)

        response = ollama.chat(
            model="llama3",
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:{query}"
            }]
        )

        self.log_msg(f"\nQ: {query}")

        for i in range(len(results)):
            self.log_msg(f"\nResult {i+1}:")
            self.log_msg(f"Text: {results[i]}")
            self.log_msg(f"Distance: {D[0][i]:.4f}")
            self.log_msg(f"Similarity: {similarities[i]:.4f}")

        self.log_msg(f"\nAnswer:\n{response['message']['content']}\n")

    # ---------------------------
    def run_tests(self):
        if not os.path.exists("health.index"):
            self.log_msg("❌ Build index first!")
            return

        index = faiss.read_index("health.index")

        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        base_questions = [
            "benefits of exercise",
            "importance of sleep",
            "healthy diet",
            "protein for muscles"
        ]

        def get_synonyms(word):
            syns = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    syns.append(l.name())
            return syns

        failures = 0

        for q in base_questions:
            test_cases = [
                q,
                q + " random xyz",
                " ".join(q.split()[::-1]),
                ""
            ]

            # synonym version
            words = q.split()
            new_q = []
            for w in words:
                syn = get_synonyms(w)
                new_q.append(random.choice(syn) if syn else w)
            test_cases.append(" ".join(new_q))

            for test in test_cases:

                if not test.strip():
                    self.log_msg("⚠ Empty query")
                    failures += 1
                    continue

                vec = get_embedding(test)
                D, I = index.search(np.array([vec]), k=1)

                if D[0][0] > 1.0:
                    self.log_msg(f"❌ Fail: {test} | {D[0][0]:.4f}")
                    failures += 1
                else:
                    self.log_msg(f"✅ Pass: {test} | {D[0][0]:.4f}")

        self.log_msg(f"\nTotal Failures: {failures}")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HealthRAGApp()
    window.show()
    sys.exit(app.exec())