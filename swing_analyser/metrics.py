import math
from swing_helpers import smooth, smooth_points, average_torso_length, normalise

# MediaPipe point numbers we use
SHOULDERS = (11, 12)
HIPS = (23, 24)
WRIST = {"right": 16, "left": 15}


# ---------------------------------------------------------------- small tools
def angle(a, b, c):
    """The angle (in degrees) at point b, formed by the lines b->a and b->c.
    For a knee: a = hip, b = knee, c = ankle. 180 = straight leg, smaller = more bent."""
    angle_a = math.atan2(a[1] - b[1], a[0] - b[0])
    angle_c = math.atan2(c[1] - b[1], c[0] - b[0])
    degrees = abs(math.degrees(angle_a - angle_c))
    if degrees > 180:
        degrees = 360 - degrees
    return degrees


def point_speeds(points, fps, ruler):
    """How fast a moving point travels each frame, in torso-lengths per second."""
    speeds = [0]
    for i in range(1, len(points)):
        distance = math.dist(points[i], points[i - 1])
        speeds.append(normalise(distance, ruler) * fps)
    return speeds


def separation_speeds(all_pixels, point_a, point_b, fps, ruler):
    """How fast the gap between two points is changing (e.g. left shoulder to right shoulder).
    When the body turns, this gap shrinks or grows, so it is a simple sign of rotation."""
    gaps = smooth([math.dist(f[point_a], f[point_b]) for f in all_pixels], 7)
    speeds = [0]
    for i in range(1, len(gaps)):
        speeds.append(normalise(abs(gaps[i] - gaps[i - 1]), ruler) * fps)
    return speeds


def peak_frame(values, start, end):
    """The frame number where values are highest, searching from start to end."""
    window = values[start:end + 1]
    return start + window.index(max(window))


# ------------------------------------------------------------------- contact
def find_contact(all_pixels, fps, ruler, hand="right"):
    """We can't see the ball, so we say contact = the frame where the racket-hand wrist is fastest."""
    wrist_path = smooth_points([f[WRIST[hand]] for f in all_pixels])
    wrist_speeds = point_speeds(wrist_path, fps, ruler)
    contact = wrist_speeds.index(max(wrist_speeds))
    return contact, wrist_speeds


# ------------------------------------------------------------------ metric 1
def leg_drive(all_pixels, fps, ruler):
    """How much the legs bend, then how much the hips rise afterwards."""
    # Knee angle every frame (hip -> knee -> ankle), smoothed to remove jitter
    right_knee = smooth([angle(f[24], f[26], f[28]) for f in all_pixels])
    left_knee = smooth([angle(f[23], f[25], f[27]) for f in all_pixels])

    # Hip height every frame (remember: in images, bigger y = lower down)
    hip_height = smooth([(f[23][1] + f[24][1]) / 2 for f in all_pixels])

    # Find the lowest the hips got, then how high they rose afterwards
    lowest_frame = hip_height.index(max(hip_height))
    highest_after = min(hip_height[lowest_frame:])
    hip_rise = normalise(hip_height[lowest_frame] - highest_after, ruler)

    return {
        "right_knee_min_angle": round(min(right_knee)),
        "left_knee_min_angle": round(min(left_knee)),
        "hip_rise_torso_lengths": round(hip_rise, 2),
    }


# ------------------------------------------------------------------ metric 2
def head_stability(all_pixels, fps, ruler, contact):
    """How much the head wanders in the 0.3 seconds either side of contact.
    Smaller number = steadier head."""
    head = smooth_points([((f[7][0] + f[8][0]) / 2, (f[7][1] + f[8][1]) / 2) for f in all_pixels])

    half_window = int(0.3 * fps)
    start = max(0, contact - half_window)
    end = min(len(head) - 1, contact + half_window)

    distance_moved = sum(math.dist(head[i], head[i - 1]) for i in range(start + 1, end + 1))
    return {"head_movement_torso_lengths": round(normalise(distance_moved, ruler), 2)}


# ------------------------------------------------------------------ metric 3
def kinetic_chain(all_pixels, fps, ruler, contact, wrist_speeds):
    """Power should travel hips -> shoulders -> wrist, each peaking just after the last.
    We find when each one is fastest and check the order."""
    hip_speeds = separation_speeds(all_pixels, HIPS[0], HIPS[1], fps, ruler)
    shoulder_speeds = separation_speeds(all_pixels, SHOULDERS[0], SHOULDERS[1], fps, ruler)

    # Look in the 1 second leading up to contact
    start = max(0, contact - int(1.0 * fps))
    hip_peak = peak_frame(hip_speeds, start, contact)
    shoulder_peak = peak_frame(shoulder_speeds, start, contact)
    wrist_peak = contact

    def ms_before_contact(frame):
        return round((contact - frame) / fps * 1000)

    return {
        "hips_peak_ms_before_contact": ms_before_contact(hip_peak),
        "shoulders_peak_ms_before_contact": ms_before_contact(shoulder_peak),
        "correct_order": hip_peak < shoulder_peak < wrist_peak,
    }


# ------------------------------------------------------------ run everything
def analyse_swing(all_pixels, fps, hand="right"):
    """Runs all three metrics on the swing and returns the results in one dictionary."""
    ruler = average_torso_length(all_pixels)
    contact, wrist_speeds = find_contact(all_pixels, fps, ruler, hand)

    return {
        "contact_time_s": round(contact / fps, 2),
        "leg_drive": leg_drive(all_pixels, fps, ruler),
        "head_stability": head_stability(all_pixels, fps, ruler, contact),
        "kinetic_chain": kinetic_chain(all_pixels, fps, ruler, contact, wrist_speeds),
    }


# ----------------------------------------------------------- printing results
def format_results(results):
    """Turns the results dictionary into text that is easy to read in the terminal."""
    leg = results["leg_drive"]
    head = results["head_stability"]
    chain = results["kinetic_chain"]
    order = "yes" if chain["correct_order"] else "no"

    return "\n".join([
        "",
        "===== SWING RESULTS =====",
        f"Contact at: {results['contact_time_s']} s",
        "",
        "Leg drive",
        f"  Right knee, deepest bend:  {leg['right_knee_min_angle']} degrees",
        f"  Left knee, deepest bend:   {leg['left_knee_min_angle']} degrees",
        f"  Hip rise after loading:    {leg['hip_rise_torso_lengths']} torso lengths",
        "",
        "Head stability (lower = steadier)",
        f"  Head movement around contact: {head['head_movement_torso_lengths']} torso lengths",
        "",
        "Kinetic chain",
        f"  Hips peaked:       {chain['hips_peak_ms_before_contact']} ms before contact",
        f"  Shoulders peaked:  {chain['shoulders_peak_ms_before_contact']} ms before contact",
        f"  Correct order (hips, shoulders, wrist): {order}",
        "=========================",
    ])