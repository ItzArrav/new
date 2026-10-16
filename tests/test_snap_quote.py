import time

from app.agents.cv_checks import snap_quote

CV = """Professional Experience
- Created Jenkins CI/CD pipelines for continuous build and deployment, cutting release time by 40%.
- Collaborated with other developers on designing, developing, testing, and implementing new JSP pages,
  HTML pages, and Servlets.
- Developed different SOAP Web Services using WSDL.
- Led the migration of 12 services to Kubernetes (OpenShift) for the “Atlas” programme.
- Optimised database perfor-
  mance for a high-volume system.
"""


def test_exact_text_is_returned_as_the_cvs_own_wording():
    assert snap_quote("Developed different SOAP Web Services using WSDL", CV) == "Developed different SOAP Web Services using WSDL."


def test_punctuation_case_and_ampersand_differences_are_tolerated():
    got = snap_quote("created jenkins ci_cd pipelines for continuous build & deployment", CV)
    assert got.startswith("Created Jenkins CI/CD pipelines for continuous build and deployment")


def test_a_cut_off_last_word_still_matches():
    assert snap_quote("Created Jenkins CI/CD pipelines for continuous build and deplo", CV).rstrip(",.").endswith("deployment")


def test_line_breaks_inside_a_sentence_are_tolerated():
    got = snap_quote("Collaborated with other developers on designing, developing, testing, and implementing new JSP pages, HTML pages, and Servlets",
                     CV)
    assert got is not None and "\n" not in got and got.endswith("Servlets.")


def test_smart_quotes_and_brackets_do_not_matter():
    assert "Atlas" in snap_quote('Led the migration of 12 services to Kubernetes OpenShift for the "Atlas" programme', CV)


def test_invented_quotes_are_rejected():
    assert snap_quote("Managed a team of fifty engineers across three continents", CV) is None
    assert snap_quote("Developed different GraphQL Services using Apollo", CV) is None  # right start, wrong substance
    assert snap_quote("", CV) is None and snap_quote("   ", CV) is None and snap_quote("///", CV) is None


def test_very_short_quotes_must_match_exactly():
    assert snap_quote("Kubernetes (OpenShift)", CV) == "Kubernetes (OpenShift)"
    assert snap_quote("Kubernetes Docker", CV) is None  # 2 words, one wrong: no fuzzy match for short quotes


def test_a_quote_is_never_a_mix_of_unrelated_places():
    assert snap_quote("Jenkins pipelines for Kubernetes migration of OpenShift services", CV) is None


def test_the_returned_text_always_comes_from_the_cv():
    for q in ("created jenkins ci_cd pipelines for continuous build & deployment", "Developed different SOAP Web Services using WSDL"):
        got = snap_quote(q, CV)
        assert " ".join(got.split()) in " ".join(CV.split())


def test_it_is_fast_on_a_long_cv():
    long_cv = (CV + "\n") * 40  # about 12,000 characters, as sent to the model
    start = time.perf_counter()
    for _ in range(20):
        assert snap_quote("Developed different SOAP Web Services using WSDL", long_cv) is not None
        assert snap_quote("Invented sentence about nothing at all in this CV", long_cv) is None
    assert time.perf_counter() - start < 6
