from mestre.asr.vocab import Correction, CorrectionDict, build_prompt
from mestre.asr.wer import normalize, sample_wer


def test_normalize_dados_e_numeros():
    assert normalize("Rola um D-20!") == normalize("rola 1 d20")
    assert normalize("A CA é dezessete.") == "a ca é 17"


def test_wer_zero_para_textos_equivalentes():
    assert sample_wer("O clérigo rola um D20.", "o clérigo rola 1 d 20") == 0.0


def test_prompt_respeita_limite():
    p = build_prompt(["Beholder"] * 500, max_chars=100)
    assert len(p) <= 101 and p.endswith(".")


def test_dicionario_leave_one_out():
    d = CorrectionDict([Correction("be roder", "Beholder", "a01.wav"), Correction("seado lixo", "CA do Lich", "a02.wav")])
    assert d.apply("o be roder ataca") == "o Beholder ataca"
    # avaliando a01, a correção que veio de a01 não pode ser usada
    assert d.apply("o be roder ataca", exclude_origin="a01.wav") == "o be roder ataca"
    assert d.apply("a seado lixo", exclude_origin="a01.wav") == "a CA do Lich"
