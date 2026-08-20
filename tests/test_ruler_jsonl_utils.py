import json

from evaluate.data.ruler.synthetic.jsonl_utils import write_manifest


def test_write_manifest_creates_jsonl_and_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "validation.jsonl"
    records = [
        {"index": 0, "input": "test", "outputs": ["123"], "length": 10},
        {"index": 1, "input": "second", "outputs": ["456"], "length": 11},
    ]

    write_manifest(output_path, records)

    raw = output_path.read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records


def test_write_manifest_matches_nemo_ascii_escaping_default(tmp_path):
    output_path = tmp_path / "validation.jsonl"

    write_manifest(output_path, [{"input": "測試"}])

    assert output_path.read_text(encoding="utf-8") == (
        '{"input": "\\u6e2c\\u8a66"}\n'
    )
