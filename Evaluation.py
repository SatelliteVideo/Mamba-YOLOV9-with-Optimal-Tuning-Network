import numpy as np
import cv2
import time


def compute_custom_reward(target, pred):
    """
    Compute a custom similarity-based reward using raw frame difference.
    """
    # Convert to grayscale
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    # Normalize pixel difference
    diff = np.abs(target_gray.astype(np.float32) - pred_gray.astype(np.float32))
    normalized_diff = diff / 255.0

    # Reward: 1 - average normalized difference (range: 0 to 1)
    reward = 1.0 - np.mean(normalized_diff)
    return reward


def Evaluation_Tracking(target_frames, pred_frames):
    """
    Evaluate predicted video frames using custom frame-based reward (no image quality metrics).

    Parameters:
    - target_frames: list of original frames (BGR format)
    - pred_frames: list of predicted frames (BGR format)

    Returns:
    - NumPy array with 8 evaluation metrics.
    """
    assert len(target_frames) == len(pred_frames), "Frame count mismatch!"

    start_time = time.time()
    rewards = []
    success_count = 0
    total_steps = 0
    opt_action_taken = 0

    for target, pred in zip(target_frames, pred_frames):
        reward = compute_custom_reward(target, pred)
        rewards.append(reward)

        # Define success and optimum action thresholds
        if reward > 0.85:
            success_count += 1
        if reward > 0.95:
            opt_action_taken += 1

        # Step range (inversely proportional to reward)
        step_range = int(max(1, (1.0 - reward) * 100))
        total_steps += step_range

    training_time = time.time() - start_time
    rewards = np.array(rewards)

    mean_reward = rewards.mean()
    std_reward = rewards.std()
    min_reward = rewards.min()
    max_reward = rewards.max()
    avg_success_rate = (success_count / len(rewards)) * 100
    avg_step_range = total_steps / len(rewards)

    return np.array([training_time, mean_reward, std_reward, min_reward, max_reward, opt_action_taken, avg_success_rate,
                     avg_step_range])
