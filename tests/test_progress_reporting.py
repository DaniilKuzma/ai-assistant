from __future__ import annotations

from src.progress import ProgressReporter, format_progress_line


def test_format_progress_line_renders_text_bar_and_metrics() -> None:
    line = format_progress_line(
        "train epoch 1/3",
        current=5,
        total=10,
        metrics={"loss": 1.23456, "examples": 160},
        elapsed_seconds=61.0,
    )

    assert "train epoch 1/3" in line
    assert "50.0%" in line
    assert "5/10" in line
    assert "loss=1.2346" in line
    assert "examples=160" in line
    assert "elapsed=00:01:01" in line
    assert "[" in line and "]" in line


def test_progress_reporter_emits_start_periodic_and_finish_lines() -> None:
    lines: list[str] = []
    clock_values = iter([0.0, 0.0, 5.0, 10.0])
    reporter = ProgressReporter(
        "evaluate",
        total=4,
        sink=lines.append,
        clock=lambda: next(clock_values),
        min_interval_seconds=0.0,
        step_interval=1,
    )

    reporter.start()
    reporter.update(2, {"exact": 1})
    reporter.finish({"exact": 3})

    assert len(lines) == 3
    assert "evaluate" in lines[0]
    assert "0.0%" in lines[0]
    assert "2/4" in lines[1]
    assert "50.0%" in lines[1]
    assert "4/4" in lines[2]
    assert "100.0%" in lines[2]
