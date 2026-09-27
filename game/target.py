"""
Dynamic Game Target with Diverse Kinematic Trajectories.
Supports Smooth Pursuit (Lissajous, Circular), Kinematic Bouncing Ball,
and Saccadic Step-Jump stimulus paradigms.
"""

import math
import random
import time
from typing import Tuple, List
import pygame


class DynamicTarget:
    """
    Animated game stimulus object with multiple dynamic motion modes.
    Tracks instantaneous position, velocity, and history trail.
    """
    MODES = ["lissajous", "circular", "bouncing", "saccadic_step"]

    def __init__(self, screen_width: int = 1280, screen_height: int = 720, radius: float = 35.0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radius = radius

        # Motion mode
        self.mode_idx = 0
        self.mode = self.MODES[self.mode_idx]

        # Kinematic state
        self.x = screen_width / 2.0
        self.y = screen_height / 2.0
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0

        # Motion parameters
        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time

        # Lissajous params
        self.amp_x = (screen_width / 2.0) * 0.72
        self.amp_y = (screen_height / 2.0) * 0.65
        self.freq_x = 0.35  # Hz
        self.freq_y = 0.50  # Hz
        self.phase_delta = math.pi / 3.0

        # Circular params
        self.circle_radius = min(screen_width, screen_height) * 0.32
        self.circle_speed = 0.40  # revolutions per second

        # Bouncing ball params
        self.bounce_speed = 380.0  # px/sec
        angle = random.uniform(0.3, 1.2)
        self.bounce_vx = self.bounce_speed * math.cos(angle)
        self.bounce_vy = self.bounce_speed * math.sin(angle)

        # Saccadic step params
        self.saccade_dwell_time = 1.8  # seconds per location
        self.saccade_last_jump = self.start_time
        self.saccade_target_x = self.x
        self.saccade_target_y = self.y

        # Visuals & Trails
        self.trail: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self.trail_duration = 0.7  # seconds
        self.pulse_phase = 0.0

    def cycle_mode(self):
        """Cycles to the next trajectory mode."""
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        self.set_mode(self.MODES[self.mode_idx])

    def set_mode(self, mode_name: str):
        """Switches active motion trajectory mode."""
        if mode_name in self.MODES:
            self.mode = mode_name
            self.mode_idx = self.MODES.index(mode_name)
            self.start_time = time.perf_counter()
            self.trail.clear()
            print(f"[DynamicTarget] Switched mode to: {self.mode}")

            if self.mode == "saccadic_step":
                self.saccade_last_jump = time.perf_counter()
                self._pick_new_saccade_target()
            elif self.mode == "bouncing":
                self.x = self.screen_width / 2.0
                self.y = self.screen_height / 2.0
                angle = random.uniform(0.3, 1.2)
                self.bounce_vx = self.bounce_speed * math.cos(angle)
                self.bounce_vy = self.bounce_speed * math.sin(angle)

    def _pick_new_saccade_target(self):
        """Selects random target position with comfortable screen margins."""
        margin_x = self.screen_width * 0.15
        margin_y = self.screen_height * 0.15
        self.saccade_target_x = random.uniform(margin_x, self.screen_width - margin_x)
        self.saccade_target_y = random.uniform(margin_y, self.screen_height - margin_y)
        self.x = self.saccade_target_x
        self.y = self.saccade_target_y
        self.vx = 0.0
        self.vy = 0.0

    def update(self) -> Tuple[float, float]:
        """Updates kinematics according to current trajectory mode and returns (x, y)."""
        now = time.perf_counter()
        dt = now - self.last_update_time
        if dt <= 1e-5:
            dt = 1.0 / 60.0
        self.last_update_time = now
        t = now - self.start_time

        prev_x, prev_y = self.x, self.y
        center_x = self.screen_width / 2.0
        center_y = self.screen_height / 2.0

        if self.mode == "lissajous":
            # Smooth Lissajous curve
            self.x = center_x + self.amp_x * math.sin(2.0 * math.pi * self.freq_x * t + self.phase_delta)
            self.y = center_y + self.amp_y * math.sin(2.0 * math.pi * self.freq_y * t)
            # Analytical velocities
            self.vx = (2.0 * math.pi * self.freq_x * self.amp_x) * math.cos(2.0 * math.pi * self.freq_x * t + self.phase_delta)
            self.vy = (2.0 * math.pi * self.freq_y * self.amp_y) * math.cos(2.0 * math.pi * self.freq_y * t)

        elif self.mode == "circular":
            # Smooth circular pursuit
            angle = 2.0 * math.pi * self.circle_speed * t
            self.x = center_x + self.circle_radius * math.cos(angle)
            self.y = center_y + self.circle_radius * math.sin(angle)
            self.vx = -self.circle_radius * 2.0 * math.pi * self.circle_speed * math.sin(angle)
            self.vy = self.circle_radius * 2.0 * math.pi * self.circle_speed * math.cos(angle)

        elif self.mode == "bouncing":
            # Bouncing ball kinematics
            self.x += self.bounce_vx * dt
            self.y += self.bounce_vy * dt

            # Screen bounds with cushion
            margin = self.radius + 15
            if self.x <= margin:
                self.x = margin
                self.bounce_vx = abs(self.bounce_vx)
            elif self.x >= self.screen_width - margin:
                self.x = self.screen_width - margin
                self.bounce_vx = -abs(self.bounce_vx)

            if self.y <= margin:
                self.y = margin
                self.bounce_vy = abs(self.bounce_vy)
            elif self.y >= self.screen_height - margin:
                self.y = self.screen_height - margin
                self.bounce_vy = -abs(self.bounce_vy)

            self.vx = self.bounce_vx
            self.vy = self.bounce_vy

        elif self.mode == "saccadic_step":
            # Sudden jumps to random points
            if now - self.saccade_last_jump >= self.saccade_dwell_time:
                self.saccade_last_jump = now
                self._pick_new_saccade_target()
                self.vx = (self.x - prev_x) / dt
                self.vy = (self.y - prev_y) / dt
            else:
                self.vx = 0.0
                self.vy = 0.0

        self.speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)

        # Append to trail
        self.trail.append((self.x, self.y, now))
        self.trail = [pt for pt in self.trail if (now - pt[2]) <= self.trail_duration]

        self.pulse_phase = (self.pulse_phase + dt * 4.0) % (2.0 * math.pi)
        return self.x, self.y

    def draw(self, surface: pygame.Surface, is_tracked: bool = False):
        """Renders the game object with glowing trail, reticle, and lock-on effects."""
        now = time.perf_counter()

        # 1. Draw glowing particle trail
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                pt = self.trail[i]
                age = now - pt[2]
                alpha_factor = max(0.0, 1.0 - (age / self.trail_duration))
                trail_radius = max(2, int(self.radius * 0.4 * alpha_factor))

                trail_color = (0, 255, 200) if not is_tracked else (0, 255, 120)
                # Create small alpha surface for soft glow
                glow_surf = pygame.Surface((trail_radius * 2, trail_radius * 2), pygame.SRCALPHA)
                alpha = int(120 * alpha_factor)
                pygame.draw.circle(glow_surf, (*trail_color, alpha), (trail_radius, trail_radius), trail_radius)
                surface.blit(glow_surf, (int(pt[0] - trail_radius), int(pt[1] - trail_radius)))

        # 2. Outer pulse ring
        pulse_expand = math.sin(self.pulse_phase) * 6.0
        outer_r = int(self.radius + pulse_expand + (8 if is_tracked else 0))
        ring_color = (0, 255, 120) if is_tracked else (0, 200, 255)

        pulse_surf = pygame.Surface((outer_r * 2 + 4, outer_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(pulse_surf, (*ring_color, 80 if not is_tracked else 160), (outer_r + 2, outer_r + 2), outer_r, 2)
        surface.blit(pulse_surf, (int(self.x - outer_r - 2), int(self.y - outer_r - 2)))

        # 3. Main Target Sphere
        core_r = int(self.radius * 0.65)
        core_color = (20, 255, 140) if is_tracked else (0, 220, 255)
        pygame.draw.circle(surface, core_color, (int(self.x), int(self.y)), core_r)

        # Center highlight dot
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), max(3, int(core_r * 0.3)))

        # 4. Crosshair Reticle Lines
        reticle_len = int(self.radius * 0.8)
        reticle_color = (255, 255, 255) if is_tracked else (180, 230, 255)
        pygame.draw.line(surface, reticle_color, (int(self.x - reticle_len), int(self.y)), (int(self.x + reticle_len), int(self.y)), 2)
        pygame.draw.line(surface, reticle_color, (int(self.x), int(self.y - reticle_len)), (int(self.x), int(self.y + reticle_len)), 2)
