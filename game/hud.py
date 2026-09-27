"""
Game HUD Overlay: Renders real-time scientific telemetry, pursuit validation gauges,
eye-state badges, camera PIP, and calibration sequences.
"""

import math
import numpy as np
import cv2
import pygame
from typing import Optional, Tuple
from validation.metrics import ValidationSample


class GameHUD:
    """
    Renders high-tech telemetry HUD, gauges, webcam PIP, and calibration overlay.
    """
    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height

        pygame.font.init()
        self.font_lg = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.font_md = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_sm = pygame.font.SysFont("Consolas", 14)
        self.font_xs = pygame.font.SysFont("Consolas", 12)

        # PIP settings
        self.show_pip = True
        self.pip_width = 240
        self.pip_height = 160
        self.pip_margin = 15

    def draw_gaze_cursor(
        self,
        surface: pygame.Surface,
        gaze_x: float,
        gaze_y: float,
        target_x: float,
        target_y: float,
        eye_state: str,
        is_on_target: bool
    ):
        """Renders calibrated eye gaze position reticle and elastic error connector."""
        gx, gy = int(gaze_x), int(gaze_y)
        tx, ty = int(target_x), int(target_y)

        # State color
        state_colors = {
            "SMOOTH_PURSUIT": (0, 255, 120),
            "SACCADE": (255, 50, 100),
            "FIXATION": (0, 200, 255),
            "BLINK": (255, 160, 0),
            "DRIFT": (160, 160, 180)
        }
        color = state_colors.get(eye_state, (200, 200, 200))

        # Elastic distance vector between eye gaze and target
        dist = math.hypot(gx - tx, gy - ty)
        if dist > 15 and eye_state != "BLINK":
            pygame.draw.line(surface, (*color, 90), (tx, ty), (gx, gy), 1)

        if eye_state == "BLINK":
            # Blink cross
            pygame.draw.line(surface, color, (gx - 10, gy - 10), (gx + 10, gy + 10), 2)
            pygame.draw.line(surface, color, (gx - 10, gy + 10), (gx + 10, gy - 10), 2)
            return

        # Calibrated gaze cursor rings
        r_outer = 16
        r_inner = 5
        gaze_surf = pygame.Surface((r_outer * 2 + 4, r_outer * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(gaze_surf, (*color, 70), (r_outer + 2, r_outer + 2), r_outer, 2)
        pygame.draw.circle(gaze_surf, (*color, 220), (r_outer + 2, r_outer + 2), r_inner)
        surface.blit(gaze_surf, (gx - r_outer - 2, gy - r_outer - 2))

    def draw_telemetry_panel(
        self,
        surface: pygame.Surface,
        sample: ValidationSample,
        mode_name: str,
        source: str,
        is_calibrated: bool,
        fps: float,
        elapsed_s: float
    ):
        """Renders glassmorphic telemetry cards on top left and top right."""
        panel_w = 340
        panel_h = 240
        panel_x = 15
        panel_y = 15

        # Glassmorphic background
        bg_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(bg_surf, (18, 22, 34, 215), (0, 0, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(bg_surf, (45, 60, 95, 180), (0, 0, panel_w, panel_h), 2, border_radius=10)
        surface.blit(bg_surf, (panel_x, panel_y))

        # Title
        title_surf = self.font_md.render("EYE GAZE VALIDATION TELEMETRY", True, (0, 240, 255))
        surface.blit(title_surf, (panel_x + 12, panel_y + 10))

        # Eye State Badge
        state_colors = {
            "SMOOTH_PURSUIT": ((0, 255, 120), "SMOOTH PURSUIT"),
            "SACCADE": ((255, 50, 100), "CATCH-UP SACCADE"),
            "FIXATION": ((0, 200, 255), "FIXATION"),
            "BLINK": ((255, 160, 0), "BLINK OCCURRING"),
            "DRIFT": ((160, 160, 180), "DRIFT / SEARCHING")
        }
        badge_color, badge_text = state_colors.get(sample.eye_state, ((200, 200, 200), sample.eye_state))

        badge_w = 200
        badge_h = 24
        badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pygame.draw.rect(badge_surf, (*badge_color, 40), (0, 0, badge_w, badge_h), border_radius=5)
        pygame.draw.rect(badge_surf, badge_color, (0, 0, badge_w, badge_h), 1, border_radius=5)
        text_badge = self.font_sm.render(badge_text, True, badge_color)
        badge_surf.blit(text_badge, (badge_w // 2 - text_badge.get_width() // 2, 3))
        surface.blit(badge_surf, (panel_x + 12, panel_y + 40))

        # Metrics rows
        y_offset = panel_y + 74
        lines = [
            f"Distance Error:   {sample.distance_error_px:5.1f} px ({sample.distance_error_deg:4.2f}°)",
            f"Target Velocity:  {sample.target_speed:5.1f} px/s",
            f"Eye Gaze Velocity:{sample.gaze_speed:5.1f} px/s",
            f"Pursuit Gain:     {sample.pursuit_gain:5.2f} (Ideal: ~1.0)",
            f"Velocity Corr r:  {sample.cross_correlation:5.2f}",
            f"Physiological Lag:{sample.optimal_lag_ms:5.0f} ms",
            f"Tracking Status:  {'LOCKED ON TARGET' if sample.is_on_target else 'OUTSIDE BOUNDS'}"
        ]

        for i, line in enumerate(lines):
            color = (255, 255, 255) if i < len(lines) - 1 else ((0, 255, 140) if sample.is_on_target else (255, 100, 100))
            txt_surf = self.font_sm.render(line, True, color)
            surface.blit(txt_surf, (panel_x + 14, y_offset + i * 22))

        # Top Right Status Bar
        top_bar_w = 320
        top_bar_h = 80
        top_bar_x = self.screen_width - top_bar_w - 15
        top_bg = pygame.Surface((top_bar_w, top_bar_h), pygame.SRCALPHA)
        pygame.draw.rect(top_bg, (18, 22, 34, 215), (0, 0, top_bar_w, top_bar_h), border_radius=8)
        pygame.draw.rect(top_bg, (45, 60, 95, 180), (0, 0, top_bar_w, top_bar_h), 2, border_radius=8)
        surface.blit(top_bg, (top_bar_x, 15))

        status_lines = [
            f"Mode:    {mode_name.upper()}",
            f"Source:  {source.upper()} | FPS: {fps:4.1f}",
            f"Session: {elapsed_s:5.1f}s | Calib: {'YES' if is_calibrated else 'DEFAULT'}"
        ]
        for idx, s_line in enumerate(status_lines):
            s_color = (0, 255, 200) if idx == 0 else (210, 220, 240)
            st_surf = self.font_sm.render(s_line, True, s_color)
            surface.blit(st_surf, (top_bar_x + 14, 22 + idx * 22))

    def draw_bottom_controls(self, surface: pygame.Surface, is_paused: bool, is_logging: bool):
        """Renders keyboard shortcut help bar at screen bottom."""
        bar_h = 32
        bar_y = self.screen_height - bar_h - 5
        bar_w = self.screen_width - 30
        bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        pygame.draw.rect(bar_surf, (15, 18, 28, 230), (0, 0, bar_w, bar_h), border_radius=6)
        pygame.draw.rect(bar_surf, (40, 50, 80, 160), (0, 0, bar_w, bar_h), 1, border_radius=6)

        rec_indicator = "[● REC]" if is_logging else "[○ IDLE]"
        rec_color = (255, 60, 60) if is_logging else (150, 150, 150)
        rec_surf = self.font_sm.render(rec_indicator, True, rec_color)
        bar_surf.blit(rec_surf, (10, 7))

        shortcuts = "[SPACE] Pause | [1-4] Modes | [C] Calibrate | [S] Sim/Cam | [R] Reset | [P] Plot Report | [TAB] PIP | [ESC] Exit"
        txt_surf = self.font_xs.render(shortcuts, True, (180, 200, 220))
        bar_surf.blit(txt_surf, (85, 9))

        surface.blit(bar_surf, (15, bar_y))

    def draw_webcam_pip(self, surface: pygame.Surface, cv2_frame: Optional[np.ndarray]):
        """Renders OpenCV camera preview with landmarks in the lower right corner."""
        if not self.show_pip or cv2_frame is None:
            return

        pip_x = self.screen_width - self.pip_width - self.pip_margin
        pip_y = self.screen_height - self.pip_height - 45

        # Resize cv2 frame
        resized = cv2.resize(cv2_frame, (self.pip_width, self.pip_height))
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to Pygame Surface
        pip_surf = pygame.surfarray.make_surface(np.rot90(np.flipud(rgb_frame)))

        # Border frame
        border_rect = pygame.Rect(pip_x - 2, pip_y - 2, self.pip_width + 4, self.pip_height + 4)
        pygame.draw.rect(surface, (0, 200, 255), border_rect, 2, border_radius=6)
        surface.blit(pip_surf, (pip_x, pip_y))

        # PIP Label
        tag_surf = self.font_xs.render("CAM PREVIEW (TAB)", True, (0, 240, 255))
        surface.blit(tag_surf, (pip_x + 6, pip_y + 4))

    def draw_calibration_overlay(
        self,
        surface: pygame.Surface,
        calib_x: float,
        calib_y: float,
        progress: float,
        point_idx: int,
        total_points: int
    ):
        """Draws interactive multi-point calibration UI."""
        # Dark overlay
        dim_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        dim_surf.fill((10, 12, 18, 220))
        surface.blit(dim_surf, (0, 0))

        # Instructions
        header = self.font_lg.render(f"CALIBRATION IN PROGRESS: POINT {point_idx + 1} / {total_points}", True, (0, 255, 220))
        surface.blit(header, (self.screen_width // 2 - header.get_width() // 2, 40))

        sub = self.font_md.render("Steadily fixate your eye gaze on the center of the pulsing ring", True, (200, 210, 230))
        surface.blit(sub, (self.screen_width // 2 - sub.get_width() // 2, 75))

        # Pulsing target ring
        cx, cy = int(calib_x), int(calib_y)
        max_r = 35
        # Progress shrink ring
        prog_r = max(6, int(max_r * (1.0 - progress)))

        pygame.draw.circle(surface, (0, 255, 180), (cx, cy), max_r, 2)
        pygame.draw.circle(surface, (255, 80, 80), (cx, cy), prog_r, 3)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 5)
