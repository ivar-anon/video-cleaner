"""Human-readable summary of an edit decision list."""

from __future__ import annotations

from .timeline import EditDecisionList


def format_seconds(seconds: float) -> str:
    """``83.456`` -> ``"1:23.46"``; hours are added when needed (``"1:02:03.00"``)."""
    centis = round(max(0.0, seconds) * 100)
    hours, rest = divmod(centis, 360_000)
    minutes, rest = divmod(rest, 6_000)
    secs = f"{rest // 100:02d}.{rest % 100:02d}"
    return f"{hours}:{minutes:02d}:{secs}" if hours else f"{minutes}:{secs}"


def render_report(edl: EditDecisionList, *, source: str, output: str | None) -> str:
    total, final, removed = edl.source_duration, edl.output_duration, edl.removed_duration
    share = 100 * removed / total if total > 0 else 0.0
    lines = [
        "video-cleaner report",
        "",
        f"  source             {source}",
        f"  output             {output or '(not rendered)'}",
        f"  original duration  {format_seconds(total)}  ({total:.2f} s)",
        f"  final duration     {format_seconds(final)}  ({final:.2f} s)",
        f"  removed            {removed:.2f} s  ({share:.1f} %)",
        "",
        f"  cuts               {len(edl.cuts)}",
    ]
    summary = edl.summary()
    for reason in sorted(summary, key=lambda r: (r != "silence", r)):
        count, seconds = int(summary[reason]["count"]), summary[reason]["seconds"]
        label = "cut" if count == 1 else "cuts"
        lines.append(f"    {reason:<16} {count:>3} {label:<5} {seconds:7.2f} s")

    retakes = [cut for cut in edl.cuts if cut.reason == "retake"]
    if retakes:
        lines += ["", "  removed takes"]
        for cut in retakes:
            span = f"{format_seconds(cut.start)}-{format_seconds(cut.end)}"
            lines.append(f'    {span}  "{cut.info.get("removed_text", "")}"')
            lines.append(
                f'    {"":{len(span)}}  restarted as "{cut.info.get("matched_text", "")}"'
                f' (similarity {cut.info.get("similarity", 0):.2f})'
            )
    return "\n".join(lines) + "\n"
