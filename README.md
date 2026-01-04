Wariant algorytmu SAHI, który skraca czas analizy zdjęcia, dzięki użyciu metody Filtrowania Krawędzi (FK).

1. Clone repo
    git clone <REPO_URL>
    cd SAHI_FK

2. Create venv
    python -m venv venv
    Windows - venv\Scripts\activate
    Linux - source venv/bin/activate

3. Upgrade pip
    pip install --upgrade pip

4. Install depedencies
    pip install -r requirements.txt

5. Install local SAHI package
    pip install -e .

6. Run example
    python FK/test.py
