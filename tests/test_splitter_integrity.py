import pytest

from rag.splitter import RecursiveTextSplitter, clean_document_text


def test_janelas_preservam_texto_limite_overlap_e_cobertura():
    text = " ".join(f"palavra{i}" for i in range(103))
    parts = RecursiveTextSplitter(chunk_size=20, chunk_overlap=5).split_text(text)
    assert all(part in text and len(part.split()) <= 20 for part in parts)
    assert all(a.split()[-5:] == b.split()[:5] for a, b in zip(parts, parts[1:]))
    reconstructed = parts[0].split()
    for part in parts[1:]:
        reconstructed.extend(part.split()[5:])
    assert reconstructed == text.split()


def test_pontuacao_numeros_e_negacao_permanecem_literais():
    text = ("O projeto não foi aprovado. O valor é R$ 52.361,34, sem reajuste! "
            "A votação ocorrerá depois? Nenhuma data foi confirmada.\n\n") * 8
    cleaned = clean_document_text(text)
    parts = RecursiveTextSplitter(chunk_size=30, chunk_overlap=6).split_text(text)
    assert all(part in cleaned for part in parts)
    assert all(len(part.split()) <= 30 for part in parts)
    assert any("R$ 52.361,34," in part for part in parts)
    assert any("não foi aprovado." in part for part in parts)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_parametros_invalidos_nao_travam(size, overlap):
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=size, chunk_overlap=overlap)


def test_caracteres_respeitam_limite():
    text = "abcdefghij" * 10
    parts = RecursiveTextSplitter(chunk_size=20, chunk_overlap=4, unit="characters").split_text(text)
    assert all(len(part) <= 20 and part in text for part in parts)
    assert all(a[-4:] == b[:4] for a, b in zip(parts, parts[1:]))


def test_prefixo_contextual_entra_no_limite():
    parts = RecursiveTextSplitter(chunk_size=20, chunk_overlap=3).split_text(
        " ".join(f"palavra{i}" for i in range(80)), title="Título", inject_context=True)
    assert all(part.startswith("[Documento: Título]") and len(part.split()) <= 20 for part in parts)
