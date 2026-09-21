import pytest
from benchmark_qwen_image21 import main, parser


def test_benchmark_options_and_invalid_repeat_do_not_load_weights(tmp_path):
    args = parser().parse_args(
        [
            "--request",
            "fixture.json",
            "--output",
            str(tmp_path / "bench"),
            "--repeats",
            "3",
            "--cold",
        ]
    )
    assert args.repeats == 3 and args.cold
    with pytest.raises(ValueError, match="repeats"):
        main(["--request", "missing.json", "--output", str(tmp_path / "bench"), "--repeats", "0"])
    assert not (tmp_path / "bench").exists()
