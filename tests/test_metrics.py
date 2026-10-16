from datetime import datetime, timedelta

from app.metrics import acceptance, aggregate, parse_external_rows, time_saved

NOW = datetime(2026, 10, 3, 12, 0, 0)


def row(user, agent, inp, out, status="success", hours_ago=1, lat=100, cost=0.01):
    return dict(ts=NOW - timedelta(hours=hours_ago), user=user, agent=agent, input_tokens=inp,
                output_tokens=out, latency_ms=lat, status=status, cost_usd=cost)


def test_aggregate_totals_windows_and_health():
    rows = [row("a", "cv", 100, 10), row("a", "cv", 200, 20, hours_ago=48), row("b", "sent", 50, 5, status="error", lat=300),
            row("b", "cv", 10, 1, hours_ago=24 * 40)]
    r = aggregate(rows, NOW)
    assert r["totals"]["input_tokens"] == 360 and r["totals"]["output_tokens"] == 36 and r["totals"]["calls"] == 4
    assert r["by_user"]["a"]["input_tokens"] == 300 and r["by_agent"]["sent"]["calls"] == 1
    assert r["runs"]["cv"] == {"24h": 1, "7d": 2, "30d": 2}
    assert r["agent_health"]["sent"]["error_rate"] == 1.0 and r["agent_health"]["sent"]["avg_latency_ms"] == 300.0


def test_acceptance_rate_ignores_pending():
    recs = [dict(category="DCS", decision="accept"), dict(category="DCS", decision="reject"),
            dict(category="DCS", decision=None), dict(category="PSS", decision="modify")]
    a = acceptance(recs)
    assert a["DCS"]["acceptance_rate"] == 0.5 and a["DCS"]["pending"] == 1
    assert a["PSS"]["acceptance_rate"] == 0.0 and a["ALL"]["recommendations"] == 4


def test_time_saved_rules():
    cfg = dict(manual_min_per_cv=10, review_min_per_cv=2, manual_min_per_feedback=1,
               review_min_per_feedback=0.25, modify_credit=0.5)
    t = time_saved(accepted=3, modified=2, rejected=1, sentiment_ok=4, cfg=cfg)
    # cv: 3*8 + 2*8*0.5 - 1*2 = 30 min; sentiment: 4*0.75 = 3 min
    assert t["cv_hours"] == 0.5 and t["sentiment_hours"] == 0.05 and t["total_hours"] == 0.55


def test_import_jsonl_with_mapping_and_nested_keys():
    content = ('{"timestamp": "2026-10-03T10:00:00Z", "who": "x", "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "ok": "success"}\n'
               'not json\n'
               '{"timestamp": 1790000000, "who": "y", "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "ok": "error"}\n')
    mp = {"ts": "timestamp", "user": "who", "input_tokens": "usage.prompt_tokens",
          "output_tokens": "usage.completion_tokens", "status": "ok"}
    rows, skipped = parse_external_rows(content, "jsonl", mp)
    assert skipped == 1 and len(rows) == 2
    assert rows[0]["input_tokens"] == 7 and rows[0]["user"] == "x" and rows[1]["status"] == "error"


def test_import_csv_default_columns_and_cost_fn():
    content = "ts,user,agent,model,input_tokens,output_tokens,latency_ms,status\n2026-10-03T09:00:00,u,ag,m,1000000,0,50,success\n"
    rows, skipped = parse_external_rows(content, "csv", cost_fn=lambda m, i, o: i / 1_000_000)
    assert skipped == 0 and rows[0]["cost_usd"] == 1.0 and rows[0]["latency_ms"] == 50


def test_month_helpers_and_retry_after():
    from app.metrics import month_start, next_month_start, retry_after_seconds
    assert month_start(datetime(2026, 10, 17, 9, 30)) == datetime(2026, 10, 1)
    assert next_month_start(datetime(2026, 10, 17)) == datetime(2026, 11, 1)
    assert next_month_start(datetime(2026, 12, 31, 23)) == datetime(2027, 1, 1)
    assert retry_after_seconds(NOW - timedelta(seconds=45), NOW) == 15
    assert retry_after_seconds(NOW - timedelta(seconds=59.9), NOW) == 1
    assert retry_after_seconds(NOW - timedelta(seconds=90), NOW) == 1  # never 0 or negative


# ---------- importing logs: automatic column names, pasted text ----------

def test_common_column_names_are_detected_without_a_mapping():
    content = '{"timestamp":"2026-10-03T10:00:00Z","agent_name":"triage","model_name":"m","prompt_tokens":7,"completion_tokens":3,"duration_ms":120,"outcome":"success","user_id":"dave"}\n'
    rows, skipped = parse_external_rows(content, "jsonl")
    assert skipped == 0 and (rows[0]["agent"], rows[0]["user"], rows[0]["input_tokens"], rows[0]["output_tokens"], rows[0]["latency_ms"]) == ("triage", "dave", 7, 3, 120)


def test_an_explicit_mapping_beats_automatic_detection():
    content = '{"timestamp":"2026-10-03T10:00:00Z","when":"2026-01-01T00:00:00Z","prompt_tokens":7,"completion_tokens":3}\n'
    rows, _ = parse_external_rows(content, "jsonl", {"ts": "when"})
    assert rows[0]["ts"].year == 2026 and rows[0]["ts"].month == 1


def test_the_platforms_own_raw_log_can_be_imported():
    line = ('{"ts":"2026-10-05T05:29:19.304778Z","run_id":"r1","user_id":3,"agent":"cv_agent","model":"gpt-4o-mini","status":"error",'
            '"latency_ms":50,"input_tokens":0,"output_tokens":0,"cost_usd":0.0,"raw":null}\n')
    rows, skipped = parse_external_rows(line, "jsonl")
    assert skipped == 0 and rows[0]["user"] == "3" and rows[0]["status"] == "error" and rows[0]["agent"] == "cv_agent"


def test_rows_without_any_timestamp_are_skipped_not_guessed():
    rows, skipped = parse_external_rows('{"agent":"a","input_tokens":1}\n', "jsonl")
    assert rows == [] and skipped == 1


def test_a_single_json_object_is_one_record():
    from app.metrics import sniff_kind
    one = '{"ts":"2026-10-03T10:00:00","input_tokens":5,"output_tokens":2}'
    assert sniff_kind(one) == "json" and len(parse_external_rows(one, "json")[0]) == 1


def test_pasted_text_format_is_guessed():
    from app.metrics import sniff_kind
    assert sniff_kind('[{"ts":"2026-10-03T10:00:00"}]') == "json"
    assert sniff_kind('{"ts":"2026-10-03T10:00:00"}\n{"ts":"2026-10-03T11:00:00"}') == "jsonl"
    assert sniff_kind("ts,input_tokens\n2026-10-03T10:00:00,5") == "csv"
