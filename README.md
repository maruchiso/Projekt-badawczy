Wariant algorytmu SAHI, który skraca czas analizy zdjęcia, dzięki użyciu metody Filtrowania Krawędzi (FK).

## Instalacja

### 1. Klonowanie repozytorium
```bash
git clone <REPO_URL>
cd SAHI_FK
```

### 2. Create venv
```bash
python -m venv venv
```
Windows:
```bash
venv\Scripts\activate
```
Linux/MacOS:
```bash
Linux - source venv/bin/activate
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install depedencies
```bash
pip install -r requirements.txt
```

### 5. Install local SAHI package
```bash
pip install -e .
```

### 6. Run example
```bash
python FK/test.py
```