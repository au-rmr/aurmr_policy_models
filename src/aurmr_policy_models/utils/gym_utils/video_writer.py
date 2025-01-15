import os
import cv2
import numpy as np
from gym.vector import AsyncVectorEnv

class AsyncVectorEnvVideoWriter:
    def __init__(self, venv: AsyncVectorEnv, output_dir: str, fps: int = 30):
        """
        Initialize the GymVideoWriter.

        Args:
            venv (AsyncVectorEnv): The vectorized environment.
            output_dir (str): Directory where videos will be saved.
            fps (int): Frames per second for the video.
        """
        self.venv = venv
        self.output_dir = output_dir
        self.fps = fps
        self.video_writers = []

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.episode = 0

        # Initialize video writers for each environment
        for i in range(venv.num_envs):
            video_path = os.path.join(output_dir, f"env_{i}_{self.episode}.mp4")
            self.video_writers.append(self._create_video_writer(video_path))

    def _create_video_writer(self, path: str):
        """Create and return a cv2.VideoWriter for saving video."""
        # Use 'mp4v' codec for mp4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = self._get_frame_dimensions()
        return cv2.VideoWriter(path, fourcc, self.fps, (width, height))

    def _get_frame_dimensions(self):
        """Get the dimensions of a rendered frame."""
        # Render a frame from one environment to determine its dimensions
        # frame = self.venv.call_sync('render', mode='rgb_array')[0]
        self.venv.call_async('render', mode='rgb_array')
        results = self.venv.call_wait()
        frame = results[0]
        return frame.shape[1], frame.shape[0]

    def render_and_write_frames(self):
        """
        Render frames from the environment and write them to the videos.
        """
        self.venv.call_async('render', mode='rgb_array')
        results = self.venv.call_wait()
        self.write_frames(results)

    def next_episode(self):
        self.close()
        self.episode += 1
        for i in range(self.venv.num_envs):
            video_path = os.path.join(self.output_dir, f"env_{i}_{self.episode}.mp4")
            self.video_writers.append(self._create_video_writer(video_path))

    def write_frames(self, frames):
        """
        Write pre-rendered frames to the video files.

        Args:
            frames (list[np.ndarray]): List of frames, one per environment.
        """
        for i, frame in enumerate(frames):
            if self.video_writers[i].isOpened():
                # Convert RGB to BGR as OpenCV expects BGR
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writers[i].write(bgr_frame)

    def close(self):
        """
        Release all video writers.
        """
        for writer in self.video_writers:
            writer.release()
        self.video_writers = []