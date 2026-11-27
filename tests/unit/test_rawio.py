import json

from indicspec.rawio import exists, gz_path, open_append, read_jsonl


def test_plain_gz_and_mixed(tmp_path):
    p = tmp_path / "requests.jsonl"
    assert read_jsonl(p) == [] and not exists(p)
    with open_append(p, compress=False) as fh:
        fh.write(json.dumps({"i": 1}) + "\n")
    # two separate appends to the .gz = two gzip members; both must be read back
    for i in (2, 3):
        with open_append(p, compress=True) as fh:
            fh.write(json.dumps({"i": i}) + "\n")
    assert gz_path(p).name == "requests.jsonl.gz" and exists(p)
    assert [r["i"] for r in read_jsonl(p)] == [1, 2, 3]
    p.unlink()
    assert [r["i"] for r in read_jsonl(p)] == [2, 3]
