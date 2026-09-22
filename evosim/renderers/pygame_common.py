# SPDX-License-Identifier: MIT
"""Shared Pygame helpers for interactive EvoSim visualizations."""

import pygame

DEFAULT_INFO_PANEL_WIDTH = 300


class PygameWindow:
    """Manage a Pygame simulation window and common interactive events."""

    def __init__(
        self,
        world_size: tuple[int, int],
        *,
        caption: str,
        fps: int,
        panel_width: int = DEFAULT_INFO_PANEL_WIDTH,
        font_size: int = 24,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        if panel_width <= 0:
            raise ValueError("panel_width must be greater than zero.")
        if font_size <= 0:
            raise ValueError("font_size must be greater than zero.")

        self.world_size = world_size
        self.panel_width = int(panel_width)
        self.fps = int(fps)

        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.screen = pygame.display.set_mode(
            simulation_display_size(world_size, self.panel_width)
        )
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, font_size)
        self.running = True

    def process_events(self) -> set[int]:
        """Process common events and return keys pressed in this frame."""
        pressed: set[int] = set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                else:
                    pressed.add(event.key)

        return pressed

    def update(self) -> None:
        """Present the current frame and limit the frame rate."""
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """Close the Pygame window."""
        pygame.quit()
        self.running = False


def simulation_display_size(
    world_size: tuple[int, int],
    panel_width: int = DEFAULT_INFO_PANEL_WIDTH,
) -> tuple[int, int]:
    """Return full window size for a world plus information panel."""
    width, height = world_size
    if width <= 0 or height <= 0:
        raise ValueError("world dimensions must be greater than zero.")
    if panel_width <= 0:
        raise ValueError("panel_width must be greater than zero.")
    return width + panel_width, height


def split_simulation_screen(
    screen: pygame.Surface,
    world_size: tuple[int, int],
    panel_width: int = DEFAULT_INFO_PANEL_WIDTH,
) -> tuple[pygame.Surface, pygame.Surface]:
    """Split a window into world and information surfaces."""
    width, height = world_size
    expected_size = simulation_display_size(world_size, panel_width)
    if screen.get_size() != expected_size:
        actual_size = screen.get_size()
        raise ValueError(
            f"simulation screen must have size {expected_size}, got {actual_size}."
        )

    world_screen = screen.subsurface(pygame.Rect(0, 0, width, height))
    info_screen = screen.subsurface(pygame.Rect(width, 0, panel_width, height))
    return world_screen, info_screen


def draw_text_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    *,
    background: tuple[int, int, int] = (28, 30, 36),
    border: tuple[int, int, int] = (70, 70, 76),
) -> None:
    """Draw one line of text per entry on a dedicated information panel."""
    screen.fill(background)
    pygame.draw.line(screen, border, (0, 0), (0, screen.get_height()), 1)

    y = 18
    for line in lines:
        text = font.render(line, True, (240, 240, 240))
        screen.blit(text, (18, y))
        y += 24
