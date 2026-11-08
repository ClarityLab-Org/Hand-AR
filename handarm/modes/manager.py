"""Switching and dispatch for the single Ursina viewer."""


class ModeManager:
    def __init__(self, factories, tracker=None):
        self.factories = factories
        self.instances = {}
        self.active_name = None
        self.tracker = tracker

    @property
    def active(self):
        return self.instances.get(self.active_name)

    def start(self, name):
        self.switch(name)

    def set_tracker(self, tracker):
        self.tracker = tracker
        for mode in self.instances.values():
            mode.tracker = tracker
        if self.active is not None:
            tracker.set_classifier(
                self.classify_hand,
                stabilize_gestures=self.active.stabilize_gestures,
                include_landmarks=self.active.include_landmarks,
            )

    def switch(self, name=None):
        if name is None:
            name = "object" if self.active_name == "pointcloud" else "pointcloud"
        if name not in self.factories:
            raise ValueError(f"Unknown viewer mode: {name}")
        if name == self.active_name:
            return
        instance = self.instances.get(name)
        if instance is None:
            instance = self.factories[name]()
        if self.active is not None:
            self.active.deactivate()
        self.active_name = name
        self.instances[name] = instance
        self.active.activate()
        print(f"Viewer mode: {name}")
        if self.tracker is not None:
            self.tracker.set_classifier(
                self.classify_hand,
                stabilize_gestures=self.active.stabilize_gestures,
                include_landmarks=self.active.include_landmarks,
            )

    def classify_hand(self, landmarks, hand_type):
        if self.active is None:
            return {"gesture": "None"}
        return self.active.classify_hand(landmarks, hand_type)

    def annotate_frame(self, frame_rgb, results, state):
        if self.active is None:
            return frame_rgb
        return self.active.annotate_frame(frame_rgb, results, state)

    def update(self, dt):
        if self.active is not None:
            self.active.update(dt)

    def input(self, key):
        if key == "m":
            self.switch()
        elif self.active is not None:
            self.active.input(key)
