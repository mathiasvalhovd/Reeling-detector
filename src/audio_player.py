"""Audio playback module using pygame."""

from pathlib import Path
import pygame


class AudioPlayer:
    """Plays an audio file when triggered."""

    def __init__(self, audio_path: str, loop: bool = True):
        self.audio_path = Path(audio_path)
        self.loop = loop
        self._playing = False
        self._initialized = False

    def _init_pygame(self):
        """Initialize pygame mixer on first use."""
        if not self._initialized:
            pygame.mixer.init()
            self._initialized = True

    @property
    def is_playing(self) -> bool:
        if self._initialized and self._playing:
            # Check if music is still actually playing
            if not pygame.mixer.music.get_busy():
                self._playing = False
        return self._playing

    def play(self) -> bool:
        """Start playing the audio. Returns False if file doesn't exist."""
        if self._playing:
            return True

        if not self.audio_path.exists():
            print(f"Warning: Audio file not found: {self.audio_path}")
            return False

        self._init_pygame()

        try:
            pygame.mixer.music.load(str(self.audio_path))
            loops = -1 if self.loop else 0  # -1 = infinite loop
            pygame.mixer.music.play(loops=loops)
            self._playing = True
            return True
        except pygame.error as e:
            print(f"Error playing audio: {e}")
            return False

    def stop(self):
        """Stop the audio playback."""
        if not self._playing:
            return

        if self._initialized:
            pygame.mixer.music.stop()
        self._playing = False

    def cleanup(self):
        """Clean up pygame resources."""
        if self._initialized:
            pygame.mixer.quit()
            self._initialized = False


class AlertPlayer:
    """Plays one-shot alert sounds."""

    _initialized = False

    @classmethod
    def _init_pygame(cls):
        """Initialize pygame mixer on first use."""
        if not cls._initialized:
            pygame.mixer.init()
            cls._initialized = True

    @classmethod
    def play_alert(cls, audio_path: str) -> bool:
        """Play a one-shot alert sound. Returns False if file doesn't exist."""
        path = Path(audio_path)
        if not path.exists():
            print(f"Warning: Alert file not found: {path}")
            return False

        cls._init_pygame()

        try:
            sound = pygame.mixer.Sound(str(path))
            sound.play()
            return True
        except pygame.error as e:
            print(f"Error playing alert: {e}")
            return False

    @classmethod
    def cleanup(cls):
        """Clean up pygame resources."""
        if cls._initialized:
            pygame.mixer.quit()
            cls._initialized = False
