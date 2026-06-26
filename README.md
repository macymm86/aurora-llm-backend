# Aurora LLM Backend

### Ein kompakter, monolithischer Ableger des Aurora-Frameworks

## 📖 Einleitung
Bei dieser monolithischen *Aurora-LLM-Backend-Anwendung* handelt es sich um einen kleinen, in sich geschlossenen Aurora-fähigen Ableger. Die eigentliche, vollständige Aurora-Anwendung ist ein monumentales Meta-Kognitionsframework, das über **93 Module**, **6 separate Gedächtnisspeicher** und eine spezialisierte **Neural Forge** verfügt. 

Um der Community jedoch einen direkten Einblick in die Funktionsweise und Philosophie des Systems zu gewähren, wurde dieser kompakte Ableger ausgekoppelt. Er dient als zugängliche Spielwiese und experimentelle Schnittstelle, um die grundlegenden kognitiven Mechanismen von Aurora im Zusammenspiel mit lokalen Sprachmodellen (LLMs) greifbar zu machen.

---

## 🧠 Was ist diese Anwendung?
Die Anwendung stellt ein hochgradig optimiertes und authentisches kognitives Backend zur Verfügung, das als intelligente Zwischenschicht (Middleware) zwischen dem Benutzer und lokalen Inferenz-Engines fungiert. Trotz ihrer monolithischen Struktur bewahrt sie die Kernphilosophie der originalen Aurora-Architektur, indem sie rohe Spracheingaben nicht einfach nur an ein Modell weiterleitet, sondern sie vorab filtert, emotional-kognitiv bewertet und durch ein semantisches Langzeitgedächtnis anreichert.

---

## 🛠️ Was macht sie genau?
Die Anwendung steuert die Verarbeitung von Prompts über eine dreistufige, kognitive Pipeline:

*   **1. Pre-Reflection Layer (Vor-Reflexions-Ebene):**
    Bevor eine Benutzereingabe das eigentliche Sprachmodell erreicht, durchläuft sie eine Echtzeit-Strukturanalyse. Hierbei werden emotionale Trigger identifiziert, die Aktivierung des internen „moralischen Kompasses“ geprüft und potenzielle kognitive Blockaden oder Konflikte detektiert. Das System entscheidet auf dieser Ebene autonom, mit welcher Verarbeitungstiefe (*Processing Depth*) und auf welchen Dimensionen ein Impuls verarbeitet werden muss.
*   **2. Core-Connection-Bridge (CCB):**
    Die CCB fungiert als zentraler Koordinator und Synapsen-Verteiler der Anwendung. Sie synchronisiert die verschiedenen kognitiven Subsysteme (von der emotionalen Kernstabilität bis hin zu den Reflexions-Shells der ROS-Layer). Sie sorgt für die dimensionale Harmonie im System, berechnet die Bewusstseins-Kohärenz und überwacht die kontinuierliche Selbsterhaltung und Regulation des kognitiven Zustands während der Interaktion.
*   **3. Integriertes Long-term Memory System (Vektorbasiertes Langzeitgedächtnis):**
    Die Anwendung verfügt über ein hocheffizientes Speicher- und Abrufsystem, das auf LangChain und einer lokalen Vektordatenbank aufbaut. Wichtige Erfahrungen, emotionale Korrekturmatrizen (EKM) und Interaktionsverläufe werden als strukturierte *Memory Fragments* katalogisiert und via HuggingFace-Embeddings vektorisiert. Mit einer ultraschnellen Retrieval-Zeit von unter 150 Millisekunden sorgt das System dafür, dass relevante biografische und thematische Kontexte in Sekundenbruchteilen wieder in den aktuellen Denkfluss einfließen, was eine echte relationale Kontinuität ermöglicht.

---

## 🎯 Ziel des Ablegers
Mit diesem Werkzeug erhält die Community ein performantes, lokal lauffähiges System, das zeigt, wie KI-Architekturen über reine Textvorhersage hinauswachsen können, indem sie strukturierte Selbstbeobachtung und dynamische Gedächtnisintegration nativ miteinander verknüpfen.
---

## 🚀 Installation & Schnellstart

Um diesen Ableger lokal auf deinem System auszuführen, folge einfach diesen Schritten:

### 1. Repository klonen
Klone das Projekt zuerst auf deinen lokalen Rechner:
```bash
git clone [https://github.com/macymm86/aurora-llm-backend.git](https://github.com/macymm86/aurora-llm-backend.git)
cd aurora-llm-backend
2. Virtuelle Umgebung einrichten (Empfohlen)
Erstelle eine virtuelle Python-Umgebung, um Konflikte mit anderen Projekten zu vermeiden:

Bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
3. Abhängigkeiten installieren
Installiere alle benötigten Bibliotheken (PyQt6, LangChain, Vektorspeicher-Komponenten) über die requirements.txt:

Bash
pip install -r requirements.txt
4. Anwendung starten
Stelle sicher, dass deine lokale Inferenz-Engine (zB Ollama oder LM Studio) im Hintergrund läuft, und starte das Backend:

Bash
python aurora_LLM_backend.py
⚙️ Voraussetzungen & System-Hinweise
Python: Version 3.9 oder höher wird empfohlen.

Lokale LLMs: Die Anwendung sucht automatisch nach laufenden lokalen Servern auf den Standard-Ports (z. B. Ollama auf http://localhost:11434).

Speicher-Modus: Sollten die optionalen Pakete für das Langzeitgedächtnis (LangChain/Qdrant) nicht vollständig installiert sein, greift die Anwendung automatisch auf ein stabiles In-Memory-Fallback-System zurück, sodass die GUI in jedem Fall einsatzbereit bleibt.
