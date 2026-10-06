from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

RAW = RAIZ / "data" / "raw"
INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"

FAKEBR = RAW / "Fake.br-Corpus"
FAKETRUE_CSV = RAW / "FakeTrue.Br" / "FakeTrueBr_corpus.csv"
