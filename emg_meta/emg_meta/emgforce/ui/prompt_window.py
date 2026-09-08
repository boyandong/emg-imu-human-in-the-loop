from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field


from PySide6.QtCore import (
    QAbstractAnimation, QEasingCurve, QPointF, QRectF, Qt, QTimer,
    QVariantAnimation, Signal,
)

from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from emgforce.experiment.models import ProtocolConfig
from emgforce.experiment.prompt_engine import PromptState


ACTION_NAMES = {
    "click": "点击", "left_swipe": "向左滑动", "right_swipe": "向右滑动",
    "rest": "保持放松（无动作）", "thumb_up": "拇指向上", "thumb_down": "拇指向下",
    "forward": "手臂向前", "backward": "手臂向后",
    "left": "手臂向左", "right": "手臂向右",
    "up": "手臂向上", "down": "手臂向下",
    "index_pinch": "拇指与食指捏合", "middle_pinch": "拇指与中指捏合（保留）",
    "fist": "主观 7/10 稳定握拳", "open_hand": "自然松手（不要用力撑开）",
    "thumb_tap": "拇指轻点", "thumb_swipe_left": "拇指向左滑",
    "thumb_swipe_right": "拇指向右滑", "thumb_swipe_up": "拇指向上滑",
    "thumb_swipe_down": "拇指向下滑", "index_hold": "食指保持",
    "middle_hold": "中指保持",
    "index_press": "食指按下并保持", "index_release": "食指松开",
    "middle_press": "中指按下并保持", "middle_release": "中指松开",
    "null_finger_snap": "响指（非目标动作）",
    "null_finger_flick": "屈指弹出（非目标动作）",
    "null_typing": "自然连续打字",
}

ARM_STATE_NAMES = {
    "still": "手臂静止",
    "up": "手臂向上摆动",
    "down": "手臂向下摆动",
    "left": "手臂向左摆动",
    "right": "手臂向右摆动",
    "forward": "手臂向前摆动",
    "backward": "手臂向后摆动",
}
HAND_ACTION_NAMES = {
    "index_pinch": "拇指与食指捏合",
    "fist": "主观 7/10 稳定握拳",
    "open_hand": "自然张开/放松",
}
MUSIC_CONTROL_LABELS = {
    f"{arm_state}_{hand_action}"
    for arm_state in ARM_STATE_NAMES
    for hand_action in HAND_ACTION_NAMES
}
ACTION_NAMES.update({
    f"{arm_state}_{hand_action}": f"{arm_name} + {HAND_ACTION_NAMES[hand_action]}"
    for arm_state, arm_name in ARM_STATE_NAMES.items()
    for hand_action in HAND_ACTION_NAMES
})

NAVIGATION_DELTAS = {
    "thumb_swipe_left": (-1, 0), "left_swipe": (-1, 0),
    "thumb_swipe_right": (1, 0), "right_swipe": (1, 0),
    "thumb_swipe_up": (0, -1), "thumb_up": (0, -1),
    "thumb_swipe_down": (0, 1), "thumb_down": (0, 1),
    "forward": (1, -1), "backward": (-1, 1),
    "left": (-1, 0), "right": (1, 0),
    "up": (0, -1), "down": (0, 1),
}
for _arm_state in ("up", "down", "left", "right", "forward", "backward"):
    for _hand_action in HAND_ACTION_NAMES:
        NAVIGATION_DELTAS[f"{_arm_state}_{_hand_action}"] = NAVIGATION_DELTAS[_arm_state]
ACTIVATION_ACTIONS = {
    "thumb_tap", "index_hold", "middle_hold", "index_pinch", "middle_pinch", "rest",
    "fist", "open_hand",
} | {f"still_{hand_action}" for hand_action in HAND_ACTION_NAMES}
DISCRETE_ACTIONS = set(NAVIGATION_DELTAS) | ACTIVATION_ACTIONS
ACTION_SYMBOLS = {
    "thumb_swipe_left": "←", "left_swipe": "←",
    "thumb_swipe_right": "→", "right_swipe": "→",
    "thumb_swipe_up": "↑", "thumb_up": "↑",
    "thumb_swipe_down": "↓", "thumb_down": "↓",
    "forward": "↗", "backward": "↙", "left": "←", "right": "→",
    "up": "↑", "down": "↓",
    "thumb_tap": "●", "index_hold": "●", "middle_hold": "●",
    "index_pinch": "●", "middle_pinch": "●", "rest": "○",
    "fist": "●", "open_hand": "○",
}
ACTION_SYMBOLS.update({
    label: ("○" if label.endswith("_open_hand") else "●")
    for label in MUSIC_CONTROL_LABELS
})
ACTIVATION_COLORS = {
    "thumb_tap": QColor("#9bea55"),
    "index_hold": QColor("#ef1741"),
    "middle_hold": QColor("#65d3ef"),
    "index_press": QColor("#ef1741"), "index_release": QColor("#ef1741"),
    "middle_press": QColor("#7c3aed"), "middle_release": QColor("#7c3aed"),
    "index_pinch": QColor("#ef1741"), "middle_pinch": QColor("#7c3aed"),
    "rest": QColor("#98a2b3"), "open_hand": QColor("#98a2b3"),
    "fist": QColor("#f59e0b"),
}
ACTIVATION_COLORS.update({
    "still_index_pinch": QColor("#ef1741"),
    "still_fist": QColor("#f59e0b"),
    "still_open_hand": QColor("#98a2b3"),
})
FINGER_TEXT_COLORS = {
    "thumb": QColor("#2563eb"),
    "index": QColor("#e11d48"),
    "middle": QColor("#7c3aed"),
    "neutral": QColor("#172033"),
}

def finger_tone(action: str) -> str:
    if action.endswith("_index_pinch"):
        return "index"
    for finger in ("thumb", "index", "middle"):
        if action.startswith(f"{finger}_"):
            return finger
    return "neutral"


@dataclass
class TaskRoute:
    actions: list[str]
    points: list[QPointF] = field(default_factory=lambda: [QPointF(0, 0)])
    activation_at_node: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_actions(cls, actions: list[str]) -> "TaskRoute":
        route = cls(list(actions))
        current = QPointF(0, 0)
        navigation_index = 0
        for action in actions:
            if action in NAVIGATION_DELTAS:
                dx, dy = NAVIGATION_DELTAS[action]
                current = QPointF(current.x() + dx, current.y() + dy)
                route.points.append(current)
                navigation_index += 1
            elif action in ACTIVATION_ACTIONS:
                route.activation_at_node[navigation_index] = action
        return route

    def completed_navigation_steps(self, action_index: int) -> int:
        return sum(action in NAVIGATION_DELTAS for action in self.actions[:action_index])


class NavigationGridCanvas(QWidget):
    """Animated collection task inspired by the discrete-grid reference video."""

    countdown_tick = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__()
        self.setObjectName("navigationCanvas")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(390)
        self.route: TaskRoute | None = None
        self.navigation_index = 0
        self.character_position = QPointF(0, 0)
        self.facing = "right"
        self.mode = "idle"
        self.activation_action = ""
        self.navigation_action = ""
        self.preview_action = ""
        self.scroll_progress = 0.0
        self.feedback_text = ""
        self.null_instruction = ""
        self._rng = random.Random(9255)
        self._particles: list[dict[str, float | QColor]] = []

        self.move_animation = QVariantAnimation(self)
        self.move_animation.setDuration(280)
        self.move_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.move_animation.valueChanged.connect(self._move_value_changed)

        self.scroll_animation = QVariantAnimation(self)
        self.scroll_animation.setStartValue(0.0); self.scroll_animation.setEndValue(1.0)
        self.scroll_animation.setEasingCurve(QEasingCurve.Type.Linear)
        self.scroll_animation.valueChanged.connect(self._scroll_value_changed)

        self.overlay_timer = QTimer(self); self.overlay_timer.setSingleShot(True)
        self.overlay_timer.timeout.connect(self._finish_feedback)
        self.confetti_timer = QTimer(self); self.confetti_timer.setInterval(33)
        self.confetti_timer.timeout.connect(self._advance_confetti)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(100)
        self.countdown_timer.timeout.connect(self._advance_countdown)
        self.countdown_total_sec = 0.0
        self.countdown_remaining_sec = 0.0
        self._countdown_deadline = 0.0

    def start_countdown(self, duration_sec: float) -> None:
        self.countdown_total_sec = max(0.1, float(duration_sec))
        self.countdown_remaining_sec = self.countdown_total_sec
        self._countdown_deadline = time.monotonic() + self.countdown_total_sec
        self.countdown_timer.start()
        self.countdown_tick.emit(self.countdown_remaining_sec, self.countdown_total_sec)
        self.update()

    def pause_countdown(self) -> None:
        if self.countdown_timer.isActive():
            self.countdown_timer.stop()
            self.countdown_remaining_sec = max(0.0, self._countdown_deadline - time.monotonic())

    def resume_countdown(self) -> None:
        if self.countdown_remaining_sec > 0 and self.mode == "continuous_null":
            self._countdown_deadline = time.monotonic() + self.countdown_remaining_sec
            self.countdown_timer.start()

    def _advance_countdown(self) -> None:
        remaining = max(0.0, self._countdown_deadline - time.monotonic())
        self.countdown_remaining_sec = remaining
        if remaining <= 0:
            self.countdown_timer.stop()
        self.countdown_tick.emit(self.countdown_remaining_sec, self.countdown_total_sec)
        self.update()


    def load_route(self, route: TaskRoute, show_feedback: bool = False) -> None:
        self.move_animation.stop(); self.confetti_timer.stop(); self.countdown_timer.stop()
        self._particles.clear()
        self.route = route; self.navigation_index = 0
        self.character_position = QPointF(route.points[0])
        self.activation_action = ""
        self.preview_action = ""
        if show_feedback:
            self.mode = "feedback"
            self.feedback_text = self._rng.choice((
                "做得很好！", "非常出色！", "漂亮完成！", "继续保持！",
            ))
            self.overlay_timer.start(900)
        else:
            self.mode = "rest"
        self.update()

    def set_navigation_index(self, index: int, animate: bool = True) -> None:
        if not self.route:
            return
        target_index = min(max(0, index), len(self.route.points) - 1)
        target = QPointF(self.route.points[target_index])
        delta = target - self.character_position
        if abs(delta.x()) >= abs(delta.y()) and delta.x():
            self.facing = "right" if delta.x() > 0 else "left"
        elif delta.y():
            self.facing = "down" if delta.y() > 0 else "up"
        self.navigation_index = target_index
        if animate and target != self.character_position:
            self.move_animation.stop()
            self.move_animation.setStartValue(QPointF(self.character_position))
            self.move_animation.setEndValue(target)
            self.move_animation.start()
        else:
            self.character_position = target
            self.update()

    def show_grid(self) -> None:
        if self.mode not in {"feedback", "confetti"}:
            self.mode = "rest"; self.activation_action = ""; self.navigation_action = ""; self.update()

    def show_preview(self, action: str) -> None:
        self.preview_action = action
        self.activation_action = action
        self.navigation_action = action
        if self.mode not in {"feedback", "confetti"}:
            self.mode = "preview"; self.update()

    def prepare_scrolling_preview(self, action: str) -> None:
        self.preview_action = action
        self.activation_action = action; self.navigation_action = action
        if (self.scroll_animation.state() == QAbstractAnimation.State.Paused
                and self.mode == "scroll_preview"):
            self.scroll_animation.resume(); return
        self.scroll_animation.stop(); self.scroll_progress = 0.0
        self.mode = "scroll_preview"; self.update()

    def start_scrolling_preview(self, action: str, duration_sec: float) -> None:
        self.preview_action = action
        self.activation_action = action; self.navigation_action = action
        self.scroll_animation.stop(); self.scroll_progress = 0.0
        self.scroll_animation.setDuration(max(1, round(duration_sec * 1000)))
        self.mode = "scroll_preview"; self.scroll_animation.start(); self.update()

    def pause_scrolling_preview(self) -> None:
        if self.scroll_animation.state() == QAbstractAnimation.State.Running:
            self.scroll_animation.pause()

    def show_navigation(self, action: str) -> None:
        if self.mode != "confetti":
            self.scroll_animation.stop()
            self.mode = "navigation"; self.navigation_action = action
            self.activation_action = ""; self.update()

    def show_activation(self, action: str) -> None:
        if self.mode != "confetti":
            self.scroll_animation.stop()
            self.mode = "activation"; self.activation_action = action
            self.navigation_action = ""; self.update()

    def show_null(self, action: str, instruction: str = "", *, continuous: bool = False) -> None:
        self.scroll_animation.stop()
        self.mode = "continuous_null" if continuous else "timed_null"
        self.preview_action = action
        self.null_instruction = instruction or ACTION_NAMES.get(action, action)
        self.update()

    def show_finished(self) -> None:
        self.countdown_timer.stop()
        self.scroll_animation.stop()
        self.mode = "confetti"; self.activation_action = ""
        colors = [QColor(c) for c in ("#ef476f", "#ffd166", "#06d6a0", "#4cc9f0", "#8b5cf6")]
        self._particles = []
        for _ in range(90):
            self._particles.append({
                "x": self._rng.random(), "y": self._rng.uniform(-0.8, 0.2),
                "speed": self._rng.uniform(0.005, 0.015),
                "drift": self._rng.uniform(-0.002, 0.002),
                "angle": self._rng.uniform(0, 360), "color": self._rng.choice(colors),
            })
        self.confetti_timer.start()
        QTimer.singleShot(3000, self._stop_confetti)
        self.update()

    def _stop_confetti(self) -> None:
        self.confetti_timer.stop()
        self._particles.clear()
        self.update()

    def reset(self) -> None:
        self.move_animation.stop(); self.overlay_timer.stop(); self.confetti_timer.stop()
        self.scroll_animation.stop(); self.countdown_timer.stop()
        self.countdown_total_sec = 0.0; self.countdown_remaining_sec = 0.0
        self.route = None; self.navigation_index = 0; self.mode = "idle"
        self.activation_action = ""; self.navigation_action = ""; self.preview_action = ""
        self._particles.clear(); self.update()



    def _move_value_changed(self, value: QPointF) -> None:
        self.character_position = QPointF(value); self.update()

    def _scroll_value_changed(self, value: float) -> None:
        self.scroll_progress = float(value); self.update()

    def _finish_feedback(self) -> None:
        if self.mode == "feedback":
            self.mode = "preview" if self.preview_action else "rest"
            self.update()

    def _advance_confetti(self) -> None:
        for particle in self._particles:
            particle["y"] = float(particle["y"]) + float(particle["speed"])
            particle["x"] = float(particle["x"]) + float(particle["drift"])
            particle["angle"] = float(particle["angle"]) + 8
            if float(particle["y"]) > 1.05:
                particle["y"] = self._rng.uniform(-.35, -.05)
                particle["x"] = self._rng.random()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        area = QRectF(self.rect()).adjusted(36, 20, -36, -20)
        if self.mode == "idle" or not self.route:
            painter.setPen(QColor("#667085")); painter.setFont(QFont("Microsoft YaHei UI", 13, QFont.Weight.DemiBold))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "开始实验后自动生成导航路线")
            return
        if self.mode == "navigation":
            self._draw_time_indicator(painter, area)
            self._draw_navigation(painter, area); return
        if self.mode == "activation":
            self._draw_time_indicator(painter, area)
            self._draw_activation(painter, area); return
        if self.mode == "timed_null":
            self._draw_time_indicator(painter, area)
            self._draw_null_prompt(
                painter, area, ACTION_NAMES.get(self.preview_action, self.preview_action)); return
        if self.mode == "continuous_null":
            self._draw_null_prompt(painter, area, self.null_instruction); return
        if self.mode == "scroll_preview":
            self._draw_time_indicator(painter, area)
            start_x = area.right() - min(120.0, area.width() * .12)
            current_x = start_x + (area.center().x() - start_x) * self.scroll_progress
            shifted = area.translated(current_x - area.center().x(), 0)
            if self.preview_action.startswith("null_"):
                self._draw_null_prompt(
                    painter, shifted, ACTION_NAMES.get(self.preview_action, self.preview_action),
                    muted=True)
            elif self.preview_action in ACTIVATION_ACTIONS:
                self._draw_activation(painter, shifted, muted=True)
            else:
                self._draw_navigation(painter, shifted, muted=True)
            return
        if self.mode == "preview":
            if self.preview_action.startswith("null_"):
                self._draw_null_prompt(
                    painter, area, ACTION_NAMES.get(self.preview_action, self.preview_action),
                    muted=True)
            elif self.preview_action in ACTIVATION_ACTIONS:
                self._draw_activation(painter, area, muted=True)
            else:
                self._draw_navigation(painter, area, muted=True)
            return
        if self.mode == "feedback":
            self._draw_feedback(painter, area); return
        if self.mode == "confetti":
            painter.setPen(QColor("#0f172a"))
            painter.setFont(QFont("Microsoft YaHei UI", 26, QFont.Weight.Bold))
            title_rect = QRectF(area.left(), area.center().y() - 55, area.width(), 55)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "🎉 本轮数据采集已全部完成！")
            painter.setPen(QColor("#059669"))
            painter.setFont(QFont("Microsoft YaHei UI", 15, QFont.Weight.DemiBold))
            sub_rect = QRectF(area.left(), area.center().y() + 10, area.width(), 40)
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, "✓ 原始肌电与标注数据已自动安全保存")
            self._draw_confetti(painter, area)
            return

        if self.mode == "rest":
            painter.setPen(QColor("#667085"))
            painter.setFont(QFont("Microsoft YaHei UI", 16, QFont.Weight.DemiBold))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "准备下一动作")
            return

        painter.setPen(QColor("#667085"))
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "等待下一动作")

    def _draw_activation(self, painter: QPainter, area: QRectF,
                         muted: bool = False) -> None:
        card_width = min(420.0, area.width() * .58); card_height = min(270.0, area.height() * .78)
        card = QRectF(area.center().x() - card_width / 2, area.center().y() - card_height / 2,
                      card_width, card_height)
        action = self.activation_action
        center = QPointF(card.center().x(), card.center().y() - 24)
        color = QColor("#98a2b3") if muted else ACTIVATION_COLORS.get(action, QColor("#65d3ef"))
        if action == "thumb_tap":
            self._draw_click_marker(painter, center, muted=muted)
        elif action in {"index_release", "middle_release"}:
            painter.setPen(QPen(color, 12)); painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 46, 46)
        else:
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(color); painter.drawEllipse(center, 48, 48)
        painter.setPen(QColor("#667085") if muted else FINGER_TEXT_COLORS[finger_tone(action)])
        painter.setFont(QFont("Microsoft YaHei UI", 18, QFont.Weight.DemiBold))
        label_top = center.y() + (82 if action == "thumb_tap" else 62)
        painter.drawText(QRectF(card.left(), label_top, card.width(), 54),
                         Qt.AlignmentFlag.AlignCenter, ACTION_NAMES.get(action, action))

    @staticmethod
    def _draw_time_indicator(painter: QPainter, area: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#d0d5dd"), 3, Qt.PenStyle.DashLine))
        x = area.center().x()
        painter.drawLine(QPointF(x, area.top() + 22), QPointF(x, area.bottom() - 22))
        painter.restore()

    def _draw_navigation(self, painter: QPainter, area: QRectF,
                         muted: bool = False) -> None:
        action = self.navigation_action
        name = ACTION_NAMES.get(action, action)
        arrow_center_y = area.top() + area.height() * .39
        direction = NAVIGATION_DELTAS.get(action)
        if direction is not None:
            arrow = self._arrow_polygon(QPointF(area.center().x(), arrow_center_y), direction)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#98a2b3") if muted else QColor("#2563eb"))
            painter.drawPolygon(arrow)
        elif action == "click":
            center = QPointF(area.center().x(), arrow_center_y)
            self._draw_click_marker(painter, center, muted=muted)
        painter.setPen(QColor("#667085") if muted else FINGER_TEXT_COLORS[finger_tone(action)])
        painter.setFont(QFont("Microsoft YaHei UI", 21, QFont.Weight.DemiBold))
        text_top = area.top() + area.height() * .72
        text_rect = QRectF(area.left(), text_top, area.width(), area.bottom() - text_top)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, name)

    def _draw_null_prompt(self, painter: QPainter, area: QRectF, text: str = "",
                          muted: bool = False) -> None:
        painter.setPen(QColor("#98a2b3") if muted else QColor("#172033"))
        if self.mode == "continuous_null" and self.countdown_total_sec > 0:
            instruction_rect = QRectF(area.left(), area.top() + 10, area.width(), 60)
            painter.setFont(QFont("Microsoft YaHei UI", 18, QFont.Weight.DemiBold))
            painter.drawText(instruction_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)

            center = QPointF(area.center().x(), area.center().y() + 28)
            radius = min(75.0, area.height() * 0.20)

            # 背景环
            painter.setPen(QPen(QColor("#e2e8f0"), 10))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius, radius)

            # 动态进度弧线
            progress = max(0.0, min(1.0, self.countdown_remaining_sec / self.countdown_total_sec))
            span_angle = int(-progress * 360 * 16)
            painter.setPen(QPen(QColor("#2563eb"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            arc_rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            painter.drawArc(arc_rect, 90 * 16, span_angle)

            # 倒计时秒数
            sec = max(0, int(round(self.countdown_remaining_sec)))
            painter.setPen(QColor("#1d4ed8"))
            painter.setFont(QFont("Microsoft YaHei UI", 26, QFont.Weight.Bold))
            sec_text = f"{sec}s"
            painter.drawText(arc_rect, Qt.AlignmentFlag.AlignCenter, sec_text)

            # 底部提示语
            label_rect = QRectF(area.left(), center.y() + radius + 15, area.width(), 35)
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Microsoft YaHei UI", 13, QFont.Weight.Normal))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter,
                             f"倒计时剩余 {sec} 秒 · 请保持自然连续打字")
        else:
            painter.setFont(QFont("Microsoft YaHei UI", 22, QFont.Weight.DemiBold))
            painter.drawText(area.adjusted(30, 30, -30, -30),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             text)


    @staticmethod
    def _draw_click_marker(painter: QPainter, center: QPointF,
                           muted: bool = False) -> None:
        """Shared marker for four_gesture click and meta_discrete thumb tap."""
        painter.setBrush(Qt.BrushStyle.NoBrush)
        outer = QColor("#d0d5dd") if muted else QColor("#93c5fd")
        inner = QColor("#98a2b3") if muted else QColor("#2563eb")
        painter.setPen(QPen(outer, 8))
        painter.drawEllipse(center, 72, 72)
        painter.setPen(QPen(inner, 10))
        painter.drawEllipse(center, 45, 45)
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(inner)
        painter.drawEllipse(center, 17, 17)

    @staticmethod
    def _arrow_polygon(center: QPointF, direction: tuple[int, int]) -> QPolygonF:
        """Return an arrow whose tip follows the supplied grid direction."""
        dx, dy = direction
        # Perpendicular vector converts local arrow width into screen coordinates.
        px, py = -dy, dx
        local_points = (
            (-92, -22), (20, -22), (20, -58), (104, 0),
            (20, 58), (20, 22), (-92, 22),
        )
        return QPolygonF([
            QPointF(center.x() + along * dx + across * px,
                    center.y() + along * dy + across * py)
            for along, across in local_points
        ])

    def _draw_feedback(self, painter: QPainter, area: QRectF) -> None:
        painter.setPen(QColor("#172033"))
        painter.setFont(QFont("Microsoft YaHei UI", 21, QFont.Weight.DemiBold))
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self.feedback_text)

    def _draw_confetti(self, painter: QPainter, area: QRectF) -> None:
        painter.save()
        for particle in self._particles:
            x = area.left() + float(particle["x"]) * area.width()
            y = area.top() + float(particle["y"]) * area.height()
            painter.translate(x, y); painter.rotate(float(particle["angle"]))
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(particle["color"])
            painter.drawRect(QRectF(-4, -7, 8, 14)); painter.resetTransform()
        painter.restore()


class ParticipantPromptWindow(QWidget):
    ARROWS = ACTION_SYMBOLS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("embeddedPrompt")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("参与者动作提示")
        self.setMinimumHeight(520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.routes: list[TaskRoute] = []
        self.actions_per_route = 1
        self.current_task_trial = 0
        self.current_state = PromptState.IDLE
        self.trial_locations: list[tuple[int, int]] = []
        self.null_kinds: dict[str, str] = {}
        self.null_instructions: dict[str, str] = {}

        layout = QVBoxLayout(self); layout.setContentsMargins(18, 16, 18, 16); layout.setSpacing(10)
        header = QGridLayout(); header.setContentsMargins(0, 0, 0, 0)
        for column in range(3): header.setColumnStretch(column, 1)
        self.status_label = QLabel("● 准备"); self.status_label.setObjectName("taskStatusIdle")
        self.state_label = QLabel("离散网格导航"); self.state_label.setObjectName("promptState")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label = QLabel("Trial 0 / 0"); self.progress_label.setObjectName("promptProgress")
        header.addWidget(self.status_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        header.addWidget(self.state_label, 0, 1, Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.progress_label, 0, 2, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header)
        self.posture_text = QLabel("")
        self.posture_text.setObjectName("postureInstruction")
        self.posture_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.posture_text.setWordWrap(True)
        self.posture_text.hide()
        layout.addWidget(self.posture_text)
        self.canvas = NavigationGridCanvas(); layout.addWidget(self.canvas, 1)
        self.canvas.countdown_tick.connect(self._on_countdown_tick)
        self.instruction_text = QLabel("等待开始")
        self.instruction_text.setObjectName("taskCollectionPrompt")
        self.instruction_text.setProperty("fingerTone", "neutral")

        self.instruction_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.instruction_text)

        # Compatibility alias used by older integrations.
        self.label = self.instruction_text

    def configure_task(self, sequence: list[str], labels: list[str],
                       protocol: ProtocolConfig | None = None) -> None:
        self.null_kinds = {}
        self.null_instructions = {}
        if protocol is not None:
            self.null_kinds = {
                label: kind for label in sequence
                if (kind := protocol.null_kind(label)) is not None
            }
            self.null_instructions = {
                label: protocol.null_instruction(label) for label in self.null_kinds
            }
            if protocol.posture_name:
                instruction = protocol.posture_instruction or protocol.posture_name
                self.posture_text.setText(
                    f"当前姿态：{protocol.posture_name} · {instruction}")
                self.posture_text.show()
            else:
                self.posture_text.clear(); self.posture_text.hide()
        label_set = set(labels)
        if label_set == {
            "thumb_tap", "thumb_swipe_left", "thumb_swipe_right",
            "thumb_swipe_up", "thumb_swipe_down", "index_hold", "middle_hold",
        }:
            self.actions_per_route = 7
        elif label_set == MUSIC_CONTROL_LABELS:
            # Music gestures are independent states rather than steps in one
            # thumb-navigation route, so each cue gets its own visual trial.
            self.actions_per_route = 1
        else:
            self.actions_per_route = max(1, len(labels))
        first_null = next(
            (index for index, label in enumerate(sequence) if label in self.null_kinds),
            len(sequence),
        )
        target_sequence = sequence[:first_null]
        self.routes = []
        self.trial_locations = []
        for index in range(0, len(target_sequence), self.actions_per_route):
            actions = target_sequence[index:index + self.actions_per_route]
            route_index = len(self.routes)
            self.routes.append(TaskRoute.from_actions(actions))
            self.trial_locations.extend((route_index, action_index)
                                        for action_index in range(len(actions)))
        for label in sequence[first_null:]:
            route_index = len(self.routes)
            self.routes.append(TaskRoute.from_actions([label]))
            self.trial_locations.append((route_index, 0))
        self.current_task_trial = 0
        self.progress_label.setText(f"Trial 0 / {len(self.routes)}")
        if self.routes: self.canvas.load_route(self.routes[0])

    def reset_task(self) -> None:
        self.routes = []; self.trial_locations = []; self.null_kinds = {}
        self.null_instructions = {}; self.current_task_trial = 0
        self.posture_text.clear(); self.posture_text.hide()
        self.progress_label.setText("Trial 0 / 0"); self.state_label.setText("离散网格导航")
        self.instruction_text.setText("等待开始"); self._set_instruction_tone("")
        self._set_status(False, False); self.canvas.reset()

    def update_prompt(self, state: PromptState, label: str, trial: int, total: int) -> None:
        previous_state = self.current_state
        self.current_state = state
        if not self.routes and total:
            # Defensive fallback for callers that have not supplied the full sequence.
            self.actions_per_route = 1
            self.routes = [TaskRoute.from_actions([label]) for _ in range(total)]

        if state == PromptState.FINISHED:
            task_index = len(self.routes) - 1 if self.routes else 0
            action_index = 0
            task_trial = len(self.routes)
        else:
            action_position = max(0, trial - 1)
            if self.trial_locations and action_position < len(self.trial_locations):
                task_index, action_index = self.trial_locations[action_position]
            elif self.trial_locations:
                task_index, action_index = self.trial_locations[-1]
            else:
                task_index = min(action_position // self.actions_per_route,
                                 max(0, len(self.routes) - 1))
                action_index = action_position % self.actions_per_route
            task_trial = task_index + 1 if trial > 0 and self.routes else 0
        self.progress_label.setText(f"Trial {task_trial} / {len(self.routes)}")

        if self.routes and task_trial and task_trial != self.current_task_trial and state != PromptState.FINISHED:
            show_feedback = self.current_task_trial > 0
            self.canvas.load_route(self.routes[task_index], show_feedback=show_feedback)
            self.current_task_trial = task_trial

        route = self.routes[task_index] if self.routes else None

        if route:
            completed = route.completed_navigation_steps(action_index)
            self.canvas.set_navigation_index(completed, animate=state == PromptState.REST)

        if state == PromptState.COUNTDOWN:
            self.state_label.setText("准备"); self.instruction_text.setText("3 · 请保持手部放松")
            self._set_instruction_tone("")
            self._set_status(True, False)
        elif state == PromptState.REST:
            null_kind = self.null_kinds.get(label)
            self.state_label.setText("准备 null 行为" if null_kind else "准备下一步")
            self.instruction_text.setText(
                f"下一行为：{ACTION_NAMES.get(label, label)} · 请准备"
                if null_kind else f"下一动作：{ACTION_NAMES.get(label, label)} · 请准备")
            self._set_instruction_tone("")
            if null_kind == "continuous":
                self.canvas.show_preview(label)
            else:
                self.canvas.prepare_scrolling_preview(label)
            self._set_status(True, False)
        elif state == PromptState.PROMPT:
            null_kind = self.null_kinds.get(label)
            if null_kind:
                continuous = null_kind == "continuous"
                self.state_label.setText("连续自然行为（null）" if continuous else "定时非目标动作（null）")
                instruction = self.null_instructions.get(label, ACTION_NAMES.get(label, label))
                self.instruction_text.setText(
                    f"持续采集：{instruction}" if continuous else f"现在执行：{instruction}")
                self._set_instruction_tone("")
                self.canvas.show_null(label, instruction, continuous=continuous)
                if continuous and previous_state == PromptState.PAUSED:
                    self.canvas.resume_countdown()
                self._set_status(True, True)
                return
            is_activation = label in ACTIVATION_ACTIONS
            if label in MUSIC_CONTROL_LABELS:
                arm_state, hand_action = label.split("_", 1)
                phase_name = "静止手部动作" if arm_state == "still" else "手臂与手部组合动作"
                self.state_label.setText(f"{phase_name} · {HAND_ACTION_NAMES[hand_action]}")
            elif label in {"rest", "open_hand"}:
                self.state_label.setText("放松基线")
            elif label == "fist":
                self.state_label.setText("7/10 握拳参考")
            elif label in {"index_pinch", "middle_pinch"}:
                self.state_label.setText("捏合动作")
            else:
                self.state_label.setText("方向动作")
            self.instruction_text.setText(f"正在采集：{ACTION_NAMES.get(label, label)}")
            self._set_instruction_tone(label)
            if is_activation: self.canvas.show_activation(label)
            else: self.canvas.show_navigation(label)
            self._set_status(True, True)
        elif state == PromptState.HOLD:
            action = "index_press" if label == "index_hold" else "middle_press"
            self.state_label.setText("按下并保持")
            self.instruction_text.setText(f"正在采集：{ACTION_NAMES[action]}")
            self._set_instruction_tone(action)
            self.canvas.show_activation(action); self._set_status(True, True)
        elif state == PromptState.RELEASE:
            action = "index_release" if label == "index_hold" else "middle_release"
            self.state_label.setText("松开")
            self.instruction_text.setText(f"正在采集：{ACTION_NAMES[action]}")
            self._set_instruction_tone(action)
            self.canvas.show_activation(action); self._set_status(True, True)
        elif state == PromptState.PAUSED:
            self.state_label.setText("已暂停"); self.instruction_text.setText("实验已暂停")
            self._set_instruction_tone("")
            self.canvas.pause_scrolling_preview()
            self.canvas.pause_countdown()
            self._set_status(True, False)
        elif state == PromptState.FINISHED:
            if route: self.canvas.set_navigation_index(len(route.points) - 1, animate=True)
            self.state_label.setText("采集完成")
            self.instruction_text.setText("🎉 本次 Session 数据采集已全部完成！数据已自动安全保存。")
            self._set_instruction_tone("")
            self.canvas.show_finished(); self._set_status(False, False, finished=True)
        else:
            self.state_label.setText("离散网格导航"); self.instruction_text.setText("等待开始")
            self._set_instruction_tone("")
            self._set_status(False, False)

    def update_phase_duration(self, state: PromptState, seconds: float) -> None:
        if (state == PromptState.REST and self.current_state == PromptState.REST
                and self.null_kinds.get(self.canvas.preview_action) != "continuous"):
            self.canvas.start_scrolling_preview(self.canvas.preview_action, seconds)
        elif state == PromptState.PROMPT and self.current_state == PromptState.PROMPT:
            if self.null_kinds.get(self.canvas.preview_action) == "continuous" or self.canvas.mode == "continuous_null":
                self.canvas.start_countdown(seconds)

    def _on_countdown_tick(self, remaining: float, total: float) -> None:
        if self.current_state == PromptState.PROMPT and self.canvas.mode == "continuous_null":
            label = self.canvas.preview_action
            instruction = self.null_instructions.get(label, ACTION_NAMES.get(label, label))
            sec = max(0, int(round(remaining)))
            self.instruction_text.setText(f"持续采集：{instruction} · 剩余 {sec} 秒")

    def _set_instruction_tone(self, action: str) -> None:
        self.instruction_text.setProperty("fingerTone", finger_tone(action))
        self.instruction_text.style().unpolish(self.instruction_text)
        self.instruction_text.style().polish(self.instruction_text)

    def _set_status(self, running: bool, collecting: bool, finished: bool = False) -> None:
        if finished: text, object_name = "● 采集完成", "taskStatusCollecting"
        elif collecting: text, object_name = "● 正在采集", "taskStatusCollecting"
        elif running: text, object_name = "● 运行中", "taskStatusRunning"
        else: text, object_name = "● 准备", "taskStatusIdle"
        self.status_label.setText(text); self.status_label.setObjectName(object_name)
        self.status_label.style().unpolish(self.status_label); self.status_label.style().polish(self.status_label)
