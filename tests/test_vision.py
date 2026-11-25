import sys
from pathlib import Path

from mestre.vision.pages import PageReading, capitular_fixes, table_to_text, visual_texts
from mestre.vision.table import dice_multiset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_mesa import score  # noqa: E402


def _reading(leitura):
    return PageReading("t", 1, None, "m", leitura, 0, 0, 0.0)


def test_capitular_letra_perdida():
    r = _reading({"capitulares": [{"palavra_completa": "Seis"}]})
    assert capitular_fixes(r, "EIS HABILIDADES FORNECEM") == [("EIS", "SEIS")]


def test_capitular_palavra_de_uma_letra():
    r = _reading({"capitulares": [{"palavra_completa": "O"}]})
    assert capitular_fixes(r, "Barulho Da Espada") == [("Barulho", "O Barulho")]


def test_capitular_igual_ao_texto_nao_corrige():
    r = _reading({"capitulares": [{"palavra_completa": "Magia"}]})
    assert capitular_fixes(r, "MAGIA PERMEIA O MUNDO") == []


def test_tabela_vira_frases_coluna_valor():
    t = {"titulo": "Tamanho", "colunas": ["Tamanho", "Espaço"], "linhas": [["Grande", "3 m/3 m"]]}
    assert table_to_text(t) == "Tabela Tamanho. Tamanho: Grande; Espaço: 3 m/3 m."


def test_ilustracao_decorativa_nao_e_indexada():
    r = _reading({"figuras": [{"descricao": "aventureiros na floresta", "relevante_para_regras": False}]})
    assert visual_texts(r) == []


def test_placar_de_dados():
    got = dice_multiset({"dados": [{"tipo": "D20", "valor": 17}, {"tipo": "d6", "valor": 3}]})
    assert got == ["d20:17", "d6:3"]
    s = score(["d20:17", "d6:5"], got)
    assert s["tp"] == 1 and s["tipo_tp"] == 2 and not s["exata"]
