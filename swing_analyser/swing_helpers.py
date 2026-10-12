import math


def smooth(values, window=5):
    """Replace each number with the average of itself and its neighbours.
    This removes the small jitter from the pose points."""
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        nearby = values[max(0, i - half): i + half + 1]
        smoothed.append(sum(nearby) / len(nearby))
    return smoothed


def smooth_points(points, window=5):
    """Same as smooth(), but for a list of (x, y) points."""
    xs = smooth([p[0] for p in points], window)
    ys = smooth([p[1] for p in points], window)
    return list(zip(xs, ys))


def torso_length(pixels):
    """Distance from the middle of the shoulders to the middle of the hips.
    We use this as a 'ruler' so results don't depend on how far the camera was."""
    shoulder_mid = ((pixels[11][0] + pixels[12][0]) / 2, (pixels[11][1] + pixels[12][1]) / 2)
    hip_mid = ((pixels[23][0] + pixels[24][0]) / 2, (pixels[23][1] + pixels[24][1]) / 2)
    return math.dist(shoulder_mid, hip_mid)


def average_torso_length(all_frames):
    """One steady ruler for the whole video (average over every frame)."""
    lengths = [torso_length(frame) for frame in all_frames]
    return sum(lengths) / len(lengths)


def normalise(distance, ruler):
    """Turn a distance in pixels into 'torso lengths'."""
    return distance / ruler