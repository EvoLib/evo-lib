# SPDX-License-Identifier: MIT

import pygame


def draw_text_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    *,
    x: int = 18,
    y: int = 18,
    line_height: int = 24,
    color: tuple[int, int, int] = (240, 240, 240),
) -> None:
    """Draw a simple text overlay with one line per entry."""

    y_offset = y

    for line in lines:
        text = font.render(line, True, color)
        screen.blit(text, (x, y_offset))
        y_offset += line_height
