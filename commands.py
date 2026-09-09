"""
Driving Command Engine
Maps detected traffic light colors to driving commands with temporal smoothing.
Prevents flickering by requiring consistent detection across multiple frames.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class DrivingCommand:
    """A driving command with metadata."""
    command: str          # "STOP", "GET READY", "MOVE AHEAD", "NO SIGNAL"
    color: Optional[str]  # "Red", "Yellow", "Green", or None
    emoji: str            # 🔴, 🟡, 🟢, ⚪
    bg_color_bgr: Tuple[int, int, int]   # Background color for HUD (BGR)
    text_color_bgr: Tuple[int, int, int]  # Text color for HUD (BGR)
    confidence: float     # Detection confidence
    timestamp: float      # Time when command was issued


# ── Command Definitions ───────────────────────────────────────────────
COMMANDS = {
    "Red": DrivingCommand(
        command="STOP",
        color="Red",
        emoji="🔴",
        bg_color_bgr=(0, 0, 200),
        text_color_bgr=(255, 255, 255),
        confidence=0.0,
        timestamp=0.0,
    ),
    "Yellow": DrivingCommand(
        command="GET READY",
        color="Yellow",
        emoji="🟡",
        bg_color_bgr=(0, 200, 230),
        text_color_bgr=(0, 0, 0),
        confidence=0.0,
        timestamp=0.0,
    ),
    "Green": DrivingCommand(
        command="MOVE AHEAD",
        color="Green",
        emoji="🟢",
        bg_color_bgr=(0, 180, 0),
        text_color_bgr=(255, 255, 255),
        confidence=0.0,
        timestamp=0.0,
    ),
    None: DrivingCommand(
        command="NO SIGNAL",
        color=None,
        emoji="⚪",
        bg_color_bgr=(80, 80, 80),
        text_color_bgr=(200, 200, 200),
        confidence=0.0,
        timestamp=0.0,
    ),
}


class CommandEngine:
    """
    Temporal smoothing engine for driving commands.

    Prevents flickering by requiring `stability_frames` consecutive
    detections of the same color before changing the active command.
    """

    def __init__(self, stability_frames: int = 3):
        """
        Args:
            stability_frames: Number of consecutive frames with the same
                            detection required before switching commands
        """
        self.stability_frames = stability_frames
        self.current_command: DrivingCommand = COMMANDS[None]
        self.candidate_color: Optional[str] = None
        self.candidate_count: int = 0
        self.history: List[DrivingCommand] = []
        self.frame_count: int = 0
        self.state_start_time: float = time.time()

    def update(self, detected_color: Optional[str], confidence: float = 0.0) -> DrivingCommand:
        """
        Process a new frame's detection result and return the current command.

        Temporal smoothing: only changes command after `stability_frames`
        consecutive detections of a new color.

        Args:
            detected_color: "Red", "Yellow", "Green", or None
            confidence: Detection confidence percentage

        Returns:
            Current DrivingCommand (may be unchanged if still stabilizing)
        """
        self.frame_count += 1

        # Same as current command? Reset candidate counter
        if detected_color == self.current_command.color:
            self.candidate_color = None
            self.candidate_count = 0
            # Update confidence on current command
            self.current_command.confidence = confidence
            return self.current_command

        # New candidate detected
        if detected_color == self.candidate_color:
            self.candidate_count += 1
        else:
            # Different candidate — reset
            self.candidate_color = detected_color
            self.candidate_count = 1

        # Stable enough to switch?
        if self.candidate_count >= self.stability_frames:
            self._switch_command(detected_color, confidence)
            self.candidate_color = None
            self.candidate_count = 0

        return self.current_command

    def _switch_command(self, color: Optional[str], confidence: float):
        """Switch to a new driving command."""
        now = time.time()

        # Create new command from template
        template = COMMANDS[color]
        new_command = DrivingCommand(
            command=template.command,
            color=template.color,
            emoji=template.emoji,
            bg_color_bgr=template.bg_color_bgr,
            text_color_bgr=template.text_color_bgr,
            confidence=confidence,
            timestamp=now,
        )

        # Log previous command
        if self.current_command.color is not None or len(self.history) > 0:
            self.history.append(self.current_command)

        self.current_command = new_command
        self.state_start_time = now

    def get_state_duration(self) -> float:
        """Get seconds since last command change."""
        return time.time() - self.state_start_time

    def get_history_summary(self) -> List[dict]:
        """Get command history as list of dicts."""
        return [
            {
                "command": cmd.command,
                "color": cmd.color or "None",
                "confidence": cmd.confidence,
                "timestamp": cmd.timestamp,
            }
            for cmd in self.history
        ]

    def reset(self):
        """Reset engine state."""
        self.current_command = COMMANDS[None]
        self.candidate_color = None
        self.candidate_count = 0
        self.history = []
        self.frame_count = 0
        self.state_start_time = time.time()
