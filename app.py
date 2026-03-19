import sys
import os
import faiss
import pickle
import numpy as np
import ollama
import random
import nltk
import urllib.request

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    from PySide6.QtCore import QUrl
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QFileDialog, QLineEdit, QScrollArea, QFrame
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
        self.resize(1300, 650)

        # Apply a glossy, transparent, gradient theme (similar to the CSS)
        self.setStyleSheet("""
            HealthRAGApp {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff7eb3, stop:0.5 #4facfe, stop:1 #00f2fe);
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }
            QFrame#dashboard {
                background-color: rgba(255, 255, 255, 40);
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 16px;
            }
            QFrame#tip_card {
                background-color: rgba(0, 0, 0, 50);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 12px;
                margin-bottom: 15px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QWidget#tips_content {
                background: transparent;
            }
            QTextEdit, QLineEdit {
                background-color: rgba(0, 0, 0, 80);
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
                padding: 10px;
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 50);
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
                padding: 10px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 90);
                color: #000;
            }
            QLabel {
                color: #ffffff;
            }
            QLabel#header {
                font-size: 22px;
                font-weight: bold;
                background: transparent;
            }
            /* Custom Scrollbar */
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 25);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 100);
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 150);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

        main_layout = QHBoxLayout()

        # ================= LEFT COLUMN: DASHBOARD =================
        self.dashboard_frame = QFrame()
        self.dashboard_frame.setObjectName("dashboard")
        self.dashboard_frame.setFixedWidth(380)
        dashboard_layout = QVBoxLayout(self.dashboard_frame)

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

        # Dashboard Layout
        header_lbl = QLabel("Physical Health RAG Dashboard")
        header_lbl.setObjectName("header")
        dashboard_layout.addWidget(header_lbl)
        dashboard_layout.addWidget(self.query_input)
        dashboard_layout.addWidget(btn_query)
        dashboard_layout.addWidget(btn_load_data)
        dashboard_layout.addWidget(btn_build_index)
        dashboard_layout.addWidget(btn_upgrade)
        dashboard_layout.addWidget(btn_test)
        dashboard_layout.addWidget(self.log)

        # ================= CENTER COLUMN: VIDEO =================
        self.video_frame = QFrame()
        self.video_frame.setObjectName("dashboard")
        video_layout = QVBoxLayout(self.video_frame)
        
        video_header = QLabel("Physical Health Video")
        video_header.setObjectName("header")
        video_header.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_header)

        if HAS_MULTIMEDIA:
            self.video_widget = QVideoWidget()
            video_layout.addWidget(self.video_widget)

            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.audio_output.setVolume(1.0)
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)
            
            # Loop video infinitely (-1 means infinite loop in PySide6)
            self.media_player.setLoops(-1)

            # Download a local video if it doesn't exist
            self.video_path = os.path.join(os.getcwd(), "local_health_video.mp4")
            if not os.path.exists(self.video_path) or os.path.getsize(self.video_path) == 0:
                print("Downloading local health video...")
                try:
                    # Reliable MP4 video placeholder
                    video_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
                    req = urllib.request.Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(self.video_path, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"Error downloading video: {e}")

            if os.path.exists(self.video_path):
                self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
                self.media_player.play()
        else:
            error_lbl = QLabel("Video player requires QtMultimedia.\n\nEnsure PySide6 is fully installed.")
            error_lbl.setAlignment(Qt.AlignCenter)
            video_layout.addWidget(error_lbl)

        # ================= RIGHT COLUMN: HEALTH TIPS =================
        self.tips_area = QScrollArea()
        self.tips_area.setWidgetResizable(True)
        self.tips_area.setFixedWidth(340)
        
        self.tips_content = QWidget()
        self.tips_content.setObjectName("tips_content")
        tips_layout = QVBoxLayout(self.tips_content)
        
        tips_header = QLabel("Health Tips")
        tips_header.setObjectName("header")
        tips_header.setAlignment(Qt.AlignCenter)
        tips_layout.addWidget(tips_header)

        # Add Tip Cards
        tips_layout.addWidget(self.create_tip_card("Stay Active", "Aim for at least 30 minutes of moderate physical activity daily to improve heart health.", "https://images.unsplash.com/photo-1517836357463-d25dfeac3438?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"))
        tips_layout.addWidget(self.create_tip_card("Eat a Balanced Diet", "Focus on whole foods, lean proteins, and plenty of vegetables to sustain energy.", "https://images.unsplash.com/photo-1490645935967-10de6ba17061?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"))
        tips_layout.addWidget(self.create_tip_card("Prioritize Rest", "Adults need 7-9 hours of sleep. It supports muscle recovery and cognitive function.", "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"))
        tips_layout.addWidget(self.create_tip_card("Stay Hydrated", "Drink plenty of water to regulate body temperature and keep joints lubricated.", "https://images.unsplash.com/photo-1527004013197-251fddd89c06?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"))
        
        tips_layout.addStretch()
        self.tips_area.setWidget(self.tips_content)

        # Add columns to main layout
        main_layout.addWidget(self.tips_area)
        main_layout.addWidget(self.dashboard_frame)
        main_layout.addWidget(self.video_frame)
        self.setLayout(main_layout)

        self.chunks = []

    # ---------------------------
    def create_tip_card(self, title, text, img_url):
        card = QFrame()
        card.setObjectName("tip_card")
        card_layout = QVBoxLayout(card)

        img_label = QLabel()
        try:
            req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req).read()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            img_label.setPixmap(pixmap.scaled(280, 150, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            img_label.setFixedSize(280, 150)
        except Exception:
            img_label.setText("Image unavailable")

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 5px;")
        
        text_lbl = QLabel(text)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("font-size: 13px; color: rgba(255, 255, 255, 200);")

        card_layout.addWidget(img_label)
        card_layout.addWidget(title_lbl)
        card_layout.addWidget(text_lbl)
        return card

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

        try:
            for i, chunk in enumerate(self.chunks):
                embeddings.append(get_embedding(chunk))
    
                if i % 50 == 0:
                    self.log_msg(f"Embedding {i}")
        except Exception as e:
            self.log_msg("❌ Failed to connect to Ollama. Please ensure the Ollama app is running!")
            return

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

        try:
            for i, chunk in enumerate(all_chunks):
                embeddings.append(get_embedding(chunk))
    
                if i % 50 == 0:
                    self.log_msg(f"Embedding {i}")
        except Exception as e:
            self.log_msg("❌ Failed to connect to Ollama. Please ensure the Ollama app is running!")
            return

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

        try:
            q_vec = get_embedding(query)
        except Exception as e:
            self.log_msg("❌ Failed to connect to Ollama. Please ensure the Ollama app is running!")
            return

        D, I = index.search(np.array([q_vec]), k=3)

        results = [chunks[i] for i in I[0]]

        # Convert distance → similarity
        similarities = [1 / (1 + d) for d in D[0]]

        context = "\n".join(results)

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:{query}"
                }]
            )
        except Exception as e:
            self.log_msg("❌ Failed to connect to Ollama. Please ensure the Ollama app is running!")
            return

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

                try:
                    vec = get_embedding(test)
                except Exception as e:
                    self.log_msg("❌ Failed to connect to Ollama. Please ensure the Ollama app is running!")
                    return

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