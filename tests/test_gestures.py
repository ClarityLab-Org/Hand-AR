import pytest
import math
from handarm.tracking.gestures import (
    is_open_palm_relaxed,
    is_peace,
    pinch_distance,
    is_pinch,
    compute_explore_features,
    compute_index_up_distance,
    compute_index_down,
    compute_thumbs_up,
    classify_both_hands,
    classify_right_explore,
    classify_left_explore_trial2,
    classify_left_explore_standalone
)

class MockLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def create_hand(pose_type):
    """Creates a 21-element list of MockLandmarks representing different hand poses."""
    lm = [MockLandmark() for _ in range(21)]
    # Set base positions
    for i in [5, 9, 13, 17]:
        lm[i].y = 0.5
    for i in [6, 10, 14, 18]:
        lm[i].y = 0.4
    
    # Wrist at origin
    lm[0].x = 0.0
    lm[0].y = 1.0

    if pose_type == "open_palm":
        # All fingers extended up (y is smaller)
        lm[8].y = 0.2
        lm[12].y = 0.2
        lm[16].y = 0.2
        lm[20].y = 0.2
        # Thumb extended
        lm[4].x = 0.5
        lm[4].y = 0.5
    elif pose_type == "fist":
        # All fingers folded down (y is larger than base)
        lm[8].y = 0.8
        lm[12].y = 0.8
        lm[16].y = 0.8
        lm[20].y = 0.8
        lm[4].x = 0.1
        lm[4].y = 0.8
    elif pose_type == "peace":
        # Index and middle up, ring and pinky down
        lm[8].y = 0.2
        lm[12].y = 0.2
        lm[16].y = 0.8
        lm[20].y = 0.8
    elif pose_type == "partial_palm":
        # Index and middle up, ring down
        lm[8].y = 0.2
        lm[12].y = 0.2
        lm[16].y = 0.8
    elif pose_type == "pinch_close":
        lm[4].x = 0.5
        lm[4].y = 0.5
        lm[8].x = 0.52
        lm[8].y = 0.52
    elif pose_type == "pinch_far":
        lm[4].x = 0.5
        lm[4].y = 0.5
        lm[8].x = 0.6
        lm[8].y = 0.6
    elif pose_type == "pinch_boundary":
        lm[4].x = 0.5
        lm[4].y = 0.5
        lm[8].x = 0.5 + 0.03
        lm[8].y = 0.5 + 0.04
    elif pose_type == "index_up":
        # Only index up
        lm[8].y = 0.2
        lm[8].x = 0.0
        # Others down and close to wrist
        for i in [12, 16, 20]:
            lm[i].y = 0.9
            lm[i].x = 0.0
    elif pose_type == "index_down":
        lm[8].y = 1.8
        lm[8].x = 0.0
        for i in [12, 16, 20]:
            lm[i].y = 0.9
            lm[i].x = 0.0
    elif pose_type == "thumbs_up_08":
        # Thumb far
        lm[4].x = 1.0
        lm[4].y = 1.0
        # Others close
        for i in [8, 12, 16, 20]:
            lm[i].x = 0.0
            lm[i].y = 1.0
    elif pose_type == "thumbs_up_05":
        # Thumb slightly far
        lm[4].x = 0.06
        lm[4].y = 1.0
        # Others close
        for i in [8, 12, 16, 20]:
            lm[i].x = 0.0
            lm[i].y = 1.0
    elif pose_type == "palm_down":
        for i in [8, 12, 16, 20]:
            lm[i].y = 1.5
    
    return lm

def test_is_open_palm_relaxed_with_open_palm():
    """Test relaxed open palm with actual open palm."""
    lm = create_hand("open_palm")
    assert is_open_palm_relaxed(lm) is True

def test_is_open_palm_relaxed_with_fist():
    """Test relaxed open palm with closed fist."""
    lm = create_hand("fist")
    assert is_open_palm_relaxed(lm) is False

def test_is_open_palm_relaxed_with_partial():
    """Test relaxed open palm with exactly 2 fingers up."""
    lm = create_hand("partial_palm")
    assert is_open_palm_relaxed(lm) is True
    
    # 1 finger up
    lm[12].y = 0.8
    assert is_open_palm_relaxed(lm) is False

def test_is_peace_with_peace_sign():
    """Test peace sign."""
    lm = create_hand("peace")
    assert is_peace(lm) is True

def test_is_peace_with_all_fingers_down():
    """Test peace sign with fist."""
    lm = create_hand("fist")
    assert is_peace(lm) is False

def test_is_peace_with_all_fingers_up():
    """Test peace sign with open palm."""
    lm = create_hand("open_palm")
    assert is_peace(lm) is False

def test_pinch_distance():
    """Test exact pinch distance calculation."""
    lm = create_hand("pinch_close")
    dist = pinch_distance(lm)
    assert math.isclose(dist, math.dist((0.5, 0.5), (0.52, 0.52)))

def test_is_pinch_with_close_fingers():
    """Test pinch detection with fingers close."""
    lm = create_hand("pinch_close")
    assert is_pinch(lm) is True

def test_is_pinch_with_far_fingers():
    """Test pinch detection with fingers far."""
    lm = create_hand("pinch_far")
    assert is_pinch(lm) is False

def test_is_pinch_at_boundary():
    """Test pinch detection exactly at boundary (distance = 0.05)."""
    lm = create_hand("pinch_boundary")
    assert is_pinch(lm) is False

def test_compute_explore_features():
    """Test explore features extraction."""
    lm = create_hand("open_palm")
    feats = compute_explore_features(lm)
    assert 'peace' in feats
    assert 'open_palm' in feats
    assert feats['open_palm'] is True
    assert feats['fist'] is False
    assert 'thumb_dist' in feats
    assert 'landmarks' in feats

def test_compute_index_up_distance_far():
    """Test index up distance with index extended upwards."""
    lm = create_hand("index_up")
    feats = compute_explore_features(lm)
    assert compute_index_up_distance(feats) is True

def test_compute_index_up_distance_close():
    """Test index up distance when close to wrist."""
    lm = create_hand("fist")
    feats = compute_explore_features(lm)
    assert compute_index_up_distance(feats) is False

def test_compute_index_down():
    """Test index extended downwards."""
    lm = create_hand("index_down")
    feats = compute_explore_features(lm)
    assert compute_index_down(feats) is True

def test_compute_thumbs_up():
    """Test thumbs up with different margins."""
    lm_08 = create_hand("thumbs_up_08")
    feats_08 = compute_explore_features(lm_08)
    assert compute_thumbs_up(feats_08, margin=0.08) is True
    
    lm_05 = create_hand("thumbs_up_05")
    feats_05 = compute_explore_features(lm_05)
    assert compute_thumbs_up(feats_05, margin=0.08) is False
    assert compute_thumbs_up(feats_05, margin=0.05) is True

def test_classify_both_hands_vertical_navigation():
    right_palm = {'open_palm': True}
    left_palm = {'open_palm': True, 'thumbs_up': False}
    left_thumbs_up = {'open_palm': False, 'thumbs_up': True}

    assert classify_both_hands(left_palm, right_palm) == 'Both hand Move Down'
    assert classify_both_hands(left_thumbs_up, right_palm) == 'Both hand Move Up'
    assert classify_both_hands(left_palm, {'open_palm': False}) == 'None'

def test_classify_right_explore():
    """Test right hand explore classification."""
    assert classify_right_explore(compute_explore_features(create_hand("open_palm"))) == 'Steering'
    assert classify_right_explore(compute_explore_features(create_hand("fist"))) == 'View Locked'
    assert classify_right_explore(compute_explore_features(create_hand("peace"))) == 'None'

def test_classify_left_explore_differences():
    """Test that trial2 and standalone classifiers produce different results for priority conflict."""
    # Create a pose that is both index_down AND index_up_simple to trigger priority difference
    lm = create_hand("index_down")
    # Make index_up_simple true by making index_up true but others not up
    # Wait, index_down requires index far, index_up_simple requires index.y < base.y
    # This is physically impossible for the same finger to be up and down.
    # Let's check priority difference: Backward vs Down vs Up
    # standalone: Down > Up
    # trial2: Up > Down
    # If both index_up and index_down are somehow true...
    
    lm = create_hand("fist")
    # Tweak to satisfy both compute_index_up_distance (trial2) and compute_index_down (both)
    # Both require index far from others.
    # index_up_distance also requires index.y < wrist.y
    lm[0].y = 1.0  # wrist
    lm[8].y = 0.5  # above wrist
    lm[8].x = 1.0  # far away
    
    for i in [12, 16, 20]:
        lm[i].x = 0.0
        lm[i].y = 1.0 # close to wrist
        lm[i-2].y = 0.9 # base
        
    # To satisfy simple_index_up for standalone, index_up must be True:
    lm[6].y = 0.8
    # others not up
    
    feats = compute_explore_features(lm)
    # This setup satisfies:
    # compute_index_up_distance(feats) -> True
    # compute_index_down(feats) -> True
    # index_up_simple -> True
    
    res_trial2 = classify_left_explore_trial2(feats)
    res_standalone = classify_left_explore_standalone(feats)
    
    assert res_trial2 == 'Up'
    assert res_standalone == 'Down'
    assert res_trial2 != res_standalone
