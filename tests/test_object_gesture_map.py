from types import SimpleNamespace

from handarm.modes.object_mode import ObjectGestureMap


def _hand():
    return [SimpleNamespace(x=0.0, y=0.5, z=0.0) for _ in range(21)]


def test_object_gesture_map_keeps_object_actions_separate():
    mapping = ObjectGestureMap()
    right = _hand()
    right[4].x, right[4].y = 0.5, 0.5
    right[8].x, right[8].y = 0.2, 0.2
    right[6].y, right[10].y, right[14].y = 0.4, 0.4, 0.4
    right[8].y, right[12].y, right[16].y = 0.2, 0.2, 0.8
    assert mapping.classify(right, "Right")["gesture"] == "Peace"

    left = _hand()
    for tip in (8, 12, 16):
        left[tip].y = 0.2
    for base in (5, 9, 13):
        left[base].y = 0.5
    assert mapping.classify(left, "Left")["gesture"] == "Open Palm"

    for tip in (8, 12, 16):
        left[tip].y = 0.8
    assert mapping.classify(left, "Left")["gesture"] == "Wrist Move"
