# TEJAS: Enterprise-Grade Secure E-Tendering Platform

> **Next-Generation Government Procurement System powered by Advanced DSA, Machine Learning, and Blockchain-inspired Audit Trails.**

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Django](https://img.shields.io/badge/Django-6.0-green) ![ML](https://img.shields.io/badge/AI-TensorFlow%20%7C%20PyTorch-orange) ![Security](https://img.shields.io/badge/Security-Biometric%20Auth-red)

---

## 📋 Executive Summary

**TEJAS** (Transparent E-tendering & Judicial Auction System) is an enterprise-grade web application designed to eliminate corruption, bid rigging, and cartelization in government and private procurement. 

Unlike traditional tender portals that act as simple "Notice Boards", TEJAS is an **active intelligent agent**. It uses a sophisticated ensemble of Machine Learning models to detect financial anomalies in real-time, employs Graph Theory to uncover hidden collusion between bidders, and leverages cryptographically secure data structures to ensure an immutable audit trail.

This project demonstrates the practical application of **Computer Science Fundamentals** (Data Structures & Algorithms) to solve critical real-world problems.

---

## 🚀 Key Features

### 1. 🛡️ Fortress-Level Security
*   **Biometric Authentication**: Integration with **DeepFace** for facial recognition and **SpeechBrain** for voice verification, ensuring that the person bidding is who they claim to be.
*   **Role-Based Access Control (RBAC)**: Strict separation of duties between Tender Officers, Bidders, and Auditors.

### 2. 🧠 AI-Powered Fraud Detection
*   **Anomaly Detection Engine**: Real-time analysis of incoming tenders using an ensemble of **Isolation Forest**, **Local Outlier Factor (LOF)**, **One-Class SVM**, and **Autoencoders**.
*   **Risk Scoring**: Every tender is assigned a risk score (LOW, MEDIUM, HIGH, EXTREME). High-risk tenders are automatically flagged or blocked before publication.
*   **NLP Text Analysis**: Uses **Sentence-Transformers (BERT)** to analyze tender descriptions for vagueness or "tailored" specifications designed to favor specific vendors.

### 3. 🕸️ Collusion & Cartel Detection
*   **Network Analysis**: Uses **Graph Theory** to map relationships between bidders.
*   **IP & Metadata Correlation**: Detects if multiple "independent" bids originate from the same physical location or device.

### 4. ⚡ High-Performance Architecture
*   **Real-Time Communication**: Built on **Django Channels** and **WebSockets** for secure, instant negotiation chat between verified parties.
*   **Async Processing**: Heavy computations (ML inference, report generation) are handled asynchronously to ensure a buttery-smooth UI.

### 5. 📜 Immutable Audit Trails
*   **Blockchain-Inspired Logging**: Every critical action (Bid submission, Status change) is chained using a **Singly Linked List** structure where each node is cryptographically linked to the previous one, preventing history (log) tampering.

---

## 🏗️ Technical Architecture & DSA Integration

TEJAS is not just a CRUD app; it is a showcase of advanced algorithmic efficiency. We have deliberately integrated **8+ Core Data Structures** to optimize performance.

| Data Structure | Implementation | Real-World Application in TEJAS | Performance Benefit |
| :--- | :--- | :--- | :--- |
| **Trie (Prefix Tree)** | `core/dsa/structures.py` | **Smart Search**: Used for the global search bar to provide instant O(L) autocomplete suggestions for tender titles and categories. | Fast, prefix-based retrieval independent of DB size. |
| **Min-Heap (Priority Queue)** | `core/dsa/advanced.py` | **Urgent Tenders**: Powers the "Closing Soon" dashboard widget. It keeps the top $k$ urgent tenders always accessible in O(1) time. | Immediate access to high-priority items without sorting the full list. |
| **Bloom Filter** | `core/dsa/advanced.py` | **Blacklist Check**: Checks if a user is in a database of 1M+ known bad actors before allowing a bid. | **O(1)** memory-efficient check. Zero DB hits for safe users. |
| **Graph (Adjacency List)** | `core/dsa/graphs.py` | **Collusion Detection**: Models Bidders as Nodes and shared attributes (IP, Phone) as Edges. DSU finds connected components (cartels). | Efficient traversal O(V+E) to find hidden relationships. |
| **Stack (LIFO)** | `core/dsa/structures.py` | **Navigation History**: Custom implementation of "Undo/Back" functionality within complex multi-step forms. | Natural mapping to user navigation flow. |
| **Queue (FIFO)** | `core/dsa/structures.py` | **Notification System**: Ensures that system alerts and bid notifications are processed strictly in the order they were received. | Guarantees fairness in first-come-first-serve scenarios. |
| **Linked List** | `core/dsa/structures.py` | **Audit Trail**: Stores bid history. Since logs only grow at the end, a Linked List provides O(1) appending without memory reallocation overhead. | Infinite growth support for logs. |
| **Merge Sort** | `core/dsa/sorting.py` | **Ranking**: Used to sort tenders by "Value for Money". Unlike QuickSort, Merge Sort is **Stable**, preserving the original order of equal-value bids (fairness). | Guaranteed O(n log n) worst-case performance. |

---

## 🧠 Machine Learning Pipeline (MLOps)

The heart of TEJAS is its **Anomaly Detection System** (`tenders/ml/evaluator.py`).

### The Ensemble Approach
We do not rely on a single model. We use a **Voting System** to minimize false positives:
1.  **Isolation Forest**: Best for detecting "global" outliers (e.g., a $1 Billion budget for a simple "pen and paper" tender).
2.  **Local Outlier Factor (LOF)**: Detects "local" anomalies relative to similar tenders (e.g., a construction project in a cheap region costing 10x the regional average).
3.  **One-Class SVM**: Defines a strict boundary of "normality" based on historical training data.
4.  **Autoencoder (PyTorch)**: A Neural Network that tries to compress and reconstruct the tender data. High "Reconstruction Error" = Anomaly.

### Feature Engineering
*   **CPV Code extraction**: Hierarchical parsing of Common Procurement Vocabulary codes.
*   **Text Embedding**: Converting tender descriptions into 768-dimensional vectors using `all-mpnet-base-v2`.
*   **Entropy Analysis**: Detecting "filler text" or copy-pasted nonsense.

---

## 💻 Technology Stack

### Backend
*   **Framework**: Django 6.0
*   **Language**: Python 3.12+
*   **Real-time**: Django Channels (ASGI) with Daphne
*   **Task Queue**: Celery + Redis (Optional for production)

### Data & ML
*   **Database**: PostgreSQL / SQLite (Dev)
*   **ML Libraries**: TensorFlow 2.x, PyTorch 2.6, Scikit-learn, Sentence-Transformers
*   **Computer Vision**: DeepFace, OpenCV

### Frontend
*   **Template Engine**: Django Templates (Jinja2 compatible)
*   **Styling**: Custom CSS (Business Professional Theme)
*   **Interactivity**: Vanilla JS + WebSockets

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10 or higher
*   Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-org/tejas-tender-portal.git
cd tejas-tender-portal
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies
> **Note**: This project has heavy ML dependencies. Installation may take a few minutes.
```bash
pip install -r requirements.txt
```

### Step 4: Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 5: Initialize ML Models
(Optional) Pre-download the transformer models to speed up first launch:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

### Step 6: Run the Server
```bash
# Development Server
python manage.py runserver

# OR via Daphne (for full WebSocket support)
daphne -p 8000 tejas.asgi:application
```

Visit `http://127.0.0.1:8000` to access the portal.

---

## 📖 Usage Guide

### 1. User Registration & Biometrics
*   Navigate to `/accounts/register`.
*   Users must complete the **Biometric Enrollment** (Face capture) to create an account.
*   *Security Note*: The biometric data is hashed and stored locally; it is never sent to third-party APIs.

### 2. Creating a Tender (Authority)
*   Go to **"Create Tender"**.
*   **Manual Entry**: Fill in the form. The "Estimated Value" and "Description" fields are analyzed in real-time.
*   **Upload**: Upload a PDF/Docx. The system uses OCR to extract data and auto-fill the form.
*   **Risk Feedback**: If the ML model flags the tender as **HIGH RISK**, it will be rejected immediately with an explanation (e.g., "Budget mismatch for this category").

### 3. Placing a Bid (Vendor)
*   Browse tenders using the smart search.
*   Click **"Place Bid"**.
*   Bids are checked against the **Bloom Filter** blacklist.
*   Once submitted, the bid is hashed and added to the **Audit Linked List**.

### 4. Live Negotiation
*   Once a bid is shortlisted, the Authority can initiate a **Secure Chat**.
*   This uses WebSockets for end-to-end encrypted-style communication directly within the portal.

---

## 📂 Project Structure

```
TEJAS/
├── accounts/           # User Mgmt, Biometrics, RBAC
├── audits/             # Audit Logging & History
├── chat/               # WebSocket Real-time Messaging
├── core/               # The "Brain" of the system
│   └── dsa/            # Custom Data Structures (Trie, BFS, Graphs...)
├── ml_models/          # Serialized ML Models (.pkl, .pth)
├── templates/          # Global HTML Templates
├── tenders/            # Main Business Logic
│   ├── ml/             # Anomaly Detection Pipeline
│   └── views.py        # Controller Logic
├── manage.py
└── requirements.txt
```

---

## ⚖️ License & Compliance

This project is built for **Academic & Research Purposes**.
© 2026 TEJAS Development Team. All Rights Reserved.
