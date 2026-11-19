from __future__ import annotations

from localclip.crop import crop_box, crop_x_expr, keyframes_from_track, out_size, smooth_track


def _eval_expr(expr: str, t: float) -> float:
    # Tiny evaluator for the generated subset: if(lt(t,X),A,B) and arithmetic.
    py = expr.replace("if(", "_if(").replace("lt(", "_lt(")
    return eval(py, {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "t": t})


def test_sizes():
    assert out_size((9, 16)) == (1080, 1920)
    assert out_size((1, 1)) == (1920, 1920)
    assert crop_box(1920, 1080, (9, 16)) == (606, 1080)
    assert crop_box(1080, 1920, (9, 16)) == (1080, 1920)
    assert crop_box(1080, 2400, (9, 16)) == (1080, 1920)


def test_smooth_track_fills_gaps_and_damps_jitter():
    samples = [(i / 3, None if i in (3, 4) else (0.3 if i % 2 else 0.32)) for i in range(30)]
    tr = smooth_track(samples)
    assert len(tr) == 30
    xs = [x for _, x in tr]
    assert max(xs) - min(xs) < 0.03


def test_no_faces_gives_empty_track():
    assert smooth_track([(0, None), (1, None)]) == []


def test_keyframes_and_expression_follow_a_pan():
    # Face moves from 25% to 75% of a 1920 px frame over 10 s.
    track = [(t / 3, 0.25 + 0.5 * (t / 30)) for t in range(31)]
    keys = keyframes_from_track(track, 1920, 606)
    assert keys[0][0] == 0.0 and keys[-1][0] == 10.0
    xs = [x for _, x in keys]
    assert xs == sorted(xs)
    assert all(0 <= x <= 1920 - 606 for x in xs)
    expr = crop_x_expr(keys)
    assert abs(_eval_expr(expr, 0.0) - keys[0][1]) < 1
    assert abs(_eval_expr(expr, 10.0) - keys[-1][1]) < 1
    mid = _eval_expr(expr, 5.0)
    assert keys[0][1] < mid < keys[-1][1]


def test_static_face_collapses_to_constant():
    track = [(t / 3, 0.5) for t in range(30)]
    keys = keyframes_from_track(track, 1920, 606)
    assert crop_x_expr(keys) == str((1920 - 606) // 2)


def test_crop_clamped_at_edges():
    track = [(0.0, 0.0), (1.0, 1.0)]
    keys = keyframes_from_track(track, 1920, 606)
    assert keys[0][1] == 0 and keys[-1][1] == 1920 - 606
