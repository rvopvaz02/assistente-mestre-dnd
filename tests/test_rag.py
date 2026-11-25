from mestre.llm import LLMResult
from mestre.rag.answer import ABSTAIN_MSG, RulesAssistant, audit_citations
from mestre.rag.index import Hit, tokenize
from mestre.rag.ingest import Chunk, _is_heading, _join_lines, _split_sentences


def _chunk(texto, secao="Cobertura", permissao="jogadores"):
    return Chunk(id=secao, texto=texto, fonte="t", titulo_fonte="Teste", versao="1", vigente_desde=None,
                 vigente_ate=None, permissao=permissao, capitulo="", secao=secao, pagina=198, pagina_pdf=196)


def test_hifenizacao():
    assert _join_lines(["empurrá-", "lo para longe"]) == "empurrá-lo para longe"
    assert _join_lines(["conjura-", "ção de magia"]) == "conjuração de magia"


def test_titulos():
    assert _is_heading("COBERTURA")
    assert not _is_heading("198")
    assert not _is_heading("Um alvo com meia-cobertura")


def test_chunk_nao_quebra_frase():
    text = "Primeira frase longa. " * 30
    parts = _split_sentences(text.strip(), 200)
    assert all(p.endswith(".") for p in parts)


def test_tokenize_remove_acentos_e_stopwords():
    assert tokenize("A meia-cobertura dá +2 na Destreza") == ["meia", "cobertura", "+2", "destreza"]


def test_auditoria_de_citacoes():
    hits = [Hit(_chunk("Um alvo com meia-cobertura tem +2 de bônus na CA e nos testes de resistência de Destreza."), 1.0)]
    ok = audit_citations("A meia-cobertura dá +2 de bônus na CA [1].", hits)
    assert ok["taxa_ok"] == 1.0
    bad = audit_citations("Tieflings têm resistência a fogo infernal. Dragões voam alto pelos céus [3].", hits)
    assert bad["sem_citacao"] and bad["citacao_invalida"]


class _FakeIndex:
    def __init__(self, score):
        self.score = score

    def search(self, q, k=5, perfil="jogadores"):
        return [Hit(_chunk("Um alvo com meia-cobertura tem +2 de bônus na CA."), self.score)]


class _FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, prompt, system=None, images=None):
        self.calls += 1
        assert "<trechos>" in prompt and "DADOS" in system
        return LLMResult(self.text, 10, 5)


def test_portao_abstem_sem_chamar_llm():
    llm = _FakeLLM("qualquer")
    a = RulesAssistant(_FakeIndex(0.001), llm, threshold=0.05).ask("Traços do Tiefling?")
    assert a.abstencao and a.texto == ABSTAIN_MSG and llm.calls == 0


def test_llm_nao_sei_vira_abstencao():
    a = RulesAssistant(_FakeIndex(0.9), _FakeLLM("NAO_SEI"), threshold=0.05).ask("x")
    assert a.abstencao and a.motivo == "LLM sem evidência"


def test_resposta_com_citacao():
    a = RulesAssistant(_FakeIndex(0.9), _FakeLLM("A meia-cobertura dá +2 de bônus na CA [1]."), threshold=0.05).ask("x")
    assert not a.abstencao and a.citacoes == [1] and a.auditoria["taxa_ok"] == 1.0
    assert "p. 198" in a.fontes()[0]
