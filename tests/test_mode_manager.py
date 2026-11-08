from handarm.modes.manager import ModeManager


class FakeTracker:
    def __init__(self):
        self.settings = None

    def set_classifier(self, classify, stabilize_gestures, include_landmarks):
        self.settings = classify, stabilize_gestures, include_landmarks


class FakeMode:
    def __init__(self, name, stabilize, landmarks):
        self.name = name
        self.stabilize_gestures = stabilize
        self.include_landmarks = landmarks
        self.active = False
        self.updates = 0
        self.inputs = []

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def classify_hand(self, _landmarks, _hand_type):
        return {"gesture": self.name}

    def annotate_frame(self, frame, _results, _state):
        return frame

    def update(self, _dt):
        self.updates += 1

    def input(self, key):
        self.inputs.append(key)


def test_modes_are_lazy_and_only_active_mode_updates():
    created = []

    def factory(name, stabilize, landmarks):
        def create():
            mode = FakeMode(name, stabilize, landmarks)
            created.append(mode)
            return mode
        return create

    manager = ModeManager({
        "pointcloud": factory("pointcloud", True, False),
        "object": factory("object", False, True),
    })
    tracker = FakeTracker()
    manager.set_tracker(tracker)

    manager.start("pointcloud")
    pointcloud = manager.active
    assert len(created) == 1
    assert tracker.settings[1:] == (True, False)

    manager.update(0.016)
    manager.input("1")
    assert pointcloud.updates == 1
    assert pointcloud.inputs == ["1"]

    manager.input("m")
    obj = manager.active
    assert len(created) == 2
    assert not pointcloud.active and obj.active
    assert pointcloud.updates == 1
    assert tracker.settings[1:] == (False, True)

    manager.update(0.016)
    assert obj.updates == 1
    assert manager.classify_hand([], "Right") == {"gesture": "object"}

    manager.input("m")
    assert manager.active is pointcloud
    assert len(created) == 2
