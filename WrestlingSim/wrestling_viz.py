import pygame
import numpy as np

class WrestlingViz:
    """Visualization class for the wrestling battle royale.
    
    Handles rendering of wrestlers, ring, and stats panel with scrolling capability.
    """
    
    def __init__(self, ring_size=4.0, screen_width=1000, screen_height=600):
        """Initialize visualization with screen dimensions and ring size."""
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Wrestling Battle Royale")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (220, 20, 60)
        self.BLUE = (0, 0, 205)
        self.CYAN = (168, 222, 87)
        self.GREEN = (10, 168, 13)
        self.YELLOW = (255, 255, 0)
        self.SKIN_COLOR = (240, 164, 12)
        
        # Ring dimensions and scaling
        self.ring_size = ring_size
        self.scale = min(screen_width * 0.9, screen_height) / (2 * ring_size * 1.1)
        self.ring_rect = pygame.Rect(
            (screen_width * 0.6 - self.ring_size * 2 * self.scale) / 2,
            (screen_height - self.ring_size * 2 * self.scale) / 2,
            self.ring_size * 2 * self.scale,
            self.ring_size * 2 * self.scale
        )
        
        # Stats tracking
        self.stats = {"current_wrestlers": [], "eliminated": [], "winner": None}
        self.initiator = None  # Currently attacking wrestler
        self.responder = None  # Currently defending wrestler
        
        # Scroll panel variables
        self.scroll_y = 0  # Current scroll position
        self.scroll_speed = 20  # Pixels to scroll per mouse wheel event
        self.panel_width = 300  # Width of stats panel
        self.panel_height = screen_height - 20  # Visible panel height

    def draw_humanoid(self, wrestler, screen_pos, is_left):
        """Draw a wrestler as a humanoid figure at the given screen position.
        
        Args:
            wrestler: The wrestler to draw
            screen_pos: (x,y) screen coordinates
            is_left: Whether this is the left-most wrestler (unused)
        """
        # Color wrestler based on role (attacker/defender/neutral)
        if wrestler == self.initiator:
            color = self.RED
        elif wrestler == self.responder:
            color = self.BLUE
        else:
            color = self.SKIN_COLOR
        
        # Scaling for wrestler size
        player_scale = 1.5
        head_radius = int(15 * player_scale)
        body_length = int(45 * player_scale)
        arm_length = int(20 * player_scale)
        leg_length = int(15 * player_scale)
        leg_pos = int(70 * player_scale)
        line_thickness = int(5 * player_scale)

        # Draw head and body
        pygame.draw.circle(self.screen, color, screen_pos, head_radius)        
        body_top = (screen_pos[0], screen_pos[1] + head_radius)
        body_bottom = (screen_pos[0], screen_pos[1] + body_length)
        pygame.draw.line(self.screen, color, body_top, body_bottom, line_thickness)
        
        # Handle attack animations
        action = wrestler.last_action
        action_time = wrestler.last_action_time
        if action in [0, 1, 3] and pygame.time.get_ticks() - action_time < 500:
            # Animate attacking limbs
            progress = min((pygame.time.get_ticks() - action_time) / 500, 1.0)
            self.draw_attack_limbs(wrestler, screen_pos, body_top, body_bottom, action, progress, color, player_scale)
        else:
            # Draw neutral stance
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - arm_length, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + arm_length, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - leg_length, screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + leg_length, screen_pos[1] + leg_pos), line_thickness)
        
        # Draw health bar above wrestler
        health_ratio = wrestler.health / wrestler.max_health
        bar_width = int(40 * player_scale)
        bar_height = int(5 * player_scale)
        bar_rect = (screen_pos[0] - bar_width // 2, screen_pos[1] - 40 * player_scale, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.BLACK, bar_rect, 2)
        pygame.draw.rect(self.screen, (100, 100, 100), bar_rect)
        pygame.draw.rect(self.screen, self.GREEN if health_ratio > 0.5 else self.YELLOW if health_ratio > 0.25 else self.RED,
                         (screen_pos[0] - bar_width // 2, screen_pos[1] - 40 * player_scale, int(bar_width * health_ratio), bar_height))
        
        # Draw name tag
        name_font = pygame.font.SysFont("Arial", int(16 * player_scale))
        name_text = name_font.render(wrestler.name, True, self.WHITE)
        name_rect = name_text.get_rect(center=(screen_pos[0], screen_pos[1] - 60 * player_scale))
        background_rect = name_rect.inflate(10 * player_scale, 6 * player_scale)
        pygame.draw.rect(self.screen, (50, 50, 50, 200), background_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.WHITE, background_rect, 2, border_radius=5)
        self.screen.blit(name_text, name_rect)

    def draw_attack_limbs(self, wrestler, screen_pos, body_top, body_bottom, action, progress, color, player_scale):
        """Draw limbs in attacking positions based on action type.
        
        Args:
            wrestler: The attacking wrestler
            screen_pos: Position on screen
            body_top: Top of body line
            body_bottom: Bottom of body line
            action: Type of attack (0=punch, 1=kick, 3=signature)
            progress: Animation progress (0-1)
            color: Color to draw with
            player_scale: Size scaling factor
        """
        head_radius = int(15 * player_scale)
        arm_length = int(20 * player_scale)
        leg_length = int(15 * player_scale)
        leg_pos = int(70 * player_scale)
        line_thickness = int(5 * player_scale)
        punch_length = int(40 * player_scale)
        kick_length = int(30 * player_scale)

        if action == 0:  # Punch animation
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + (punch_length if wrestler.id % 2 else -punch_length) * progress, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - (arm_length if wrestler.id % 2 else -arm_length), screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - leg_length, screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + leg_length, screen_pos[1] + leg_pos), line_thickness)
        elif action == 1:  # Kick animation
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + (kick_length if wrestler.id % 2 else -kick_length) * progress, screen_pos[1] + leg_pos + kick_length * progress), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - (leg_length if wrestler.id % 2 else -leg_length), screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - arm_length, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + arm_length, screen_pos[1] + arm_length), line_thickness)
        elif action == 3:  # Signature move animation
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - arm_length, screen_pos[1] - arm_length * progress), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + arm_length, screen_pos[1] - arm_length * progress), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - leg_length, screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + leg_length, screen_pos[1] + leg_pos), line_thickness)
        
        # Draw hit effect if attack connects mid-animation
        if 0.4 < progress < 0.6 and pygame.time.get_ticks() - wrestler.last_hit_time < 100:
            pygame.draw.circle(self.screen, self.YELLOW, (int(screen_pos[0] + (punch_length if wrestler.id % 2 else -punch_length)), int(screen_pos[1] + 30 * player_scale)), 
                               int(15 * player_scale * (1 - abs(progress - 0.5) * 2)), 2)

    def pos_to_screen(self, pos, is_left):
        """Convert world coordinates to screen coordinates."""
        x = self.ring_rect.centerx + (pos[0] * self.scale)
        y = self.ring_rect.centery - (pos[1] * self.scale)
        return (int(x), int(y))

    def draw_ring(self):
        """Draw the wrestling ring and ropes."""
        self.screen.fill((50, 50, 70))  # Dark blue background
        pygame.draw.rect(self.screen, (240, 220, 180), self.ring_rect)  # Light tan ring canvas
        rope_colors = [(200, 200, 200), (180, 180, 180), (160, 160, 160)]  # Top to bottom rope colors
        for i in range(3):  # Draw three ropes
            rope_y = self.ring_rect.top + (i+1)*self.ring_rect.height//4
            pygame.draw.line(self.screen, rope_colors[i], (self.ring_rect.left, rope_y), (self.ring_rect.right, rope_y), 3)

    def draw_stats_panel(self):
        """Draw the scrollable stats panel showing wrestler status."""
        # Calculate total content height
        wrestler_card_height = 110
        eliminated_entry_height = 25
        content_height = (40 + len(self.stats["current_wrestlers"]) * wrestler_card_height +  # Current wrestlers
                         (30 + len(self.stats["eliminated"]) * eliminated_entry_height if self.stats["eliminated"] else 0))  # Eliminated section

        # Create surface for entire content (may be larger than visible area)
        content_surface = pygame.Surface((self.panel_width, max(content_height, self.panel_height)))
        content_surface.fill((40, 40, 50))  # Dark background
        
        # Draw current wrestlers section
        y_pos = 10
        title = self.title_font.render("Current Wrestlers", True, self.WHITE)
        content_surface.blit(title, (10, y_pos))
        y_pos += 40
        
        for wrestler in self.stats["current_wrestlers"]:
            # Color card based on role (attacker/defender/neutral)
            card_color = (220, 20, 60) if wrestler == self.initiator else (0, 0, 205) if wrestler == self.responder else (60, 60, 70)
            pygame.draw.rect(content_surface, card_color, (10, y_pos, self.panel_width-20, 110))
            
            # Draw name with outline effect
            name = self.title_font.render(wrestler.name, True, self.BLACK)
            content_surface.blit(name, (17, y_pos+7))
            name = self.title_font.render(wrestler.name, True, self.WHITE)
            content_surface.blit(name, (15, y_pos+5))
            
            # Health display
            health_ratio = wrestler.health / wrestler.max_health
            health_text = self.font.render(f"HP: {wrestler.health:.1f}/{wrestler.max_health}", True, self.BLACK)
            content_surface.blit(health_text, (17, y_pos+32))
            health_text = self.font.render(f"HP: {wrestler.health:.1f}/{wrestler.max_health}", True, self.WHITE)
            content_surface.blit(health_text, (15, y_pos+30))
            
            # Health bar
            health_bar_rect = (15, y_pos+50, self.panel_width-40, 10)
            pygame.draw.rect(content_surface, self.BLACK, health_bar_rect, 3)
            pygame.draw.rect(content_surface, (100, 100, 100), (15, y_pos+50, self.panel_width-40, 10))
            pygame.draw.rect(content_surface, self.GREEN, (15, y_pos+50, int((self.panel_width-40)*health_ratio), 10))
            
            # Stamina display
            stamina_ratio = wrestler.stamina / wrestler.max_stamina
            stamina_text = self.font.render(f"STA: {wrestler.stamina:.0f}/{wrestler.max_stamina}", True, self.BLACK)
            content_surface.blit(stamina_text, (17, y_pos+72))
            stamina_text = self.font.render(f"STA: {wrestler.stamina:.0f}/{wrestler.max_stamina}", True, self.WHITE)
            content_surface.blit(stamina_text, (15, y_pos+70))

            # Stamina bar
            stamina_bar_rect = (15, y_pos+90, self.panel_width-40, 10)
            pygame.draw.rect(content_surface, self.BLACK, stamina_bar_rect, 3)
            pygame.draw.rect(content_surface, (100, 100, 100), (15, y_pos+90, self.panel_width-40, 10))
            pygame.draw.rect(content_surface, self.CYAN, (15, y_pos+90, int((self.panel_width-40)*stamina_ratio), 10))
            
            y_pos += 120
        
        # Draw eliminated section if any wrestlers have been eliminated
        if self.stats["eliminated"]:
            title = self.title_font.render("Eliminated", True, self.WHITE)
            content_surface.blit(title, (10, y_pos))
            y_pos += 30
            for wrestler in self.stats["eliminated"]:
                text = self.font.render(wrestler.name, True, (200, 200, 200))
                content_surface.blit(text, (15, y_pos))
                y_pos += 25

        # Create visible panel surface
        panel = pygame.Surface((self.panel_width, self.panel_height))
        panel.fill((40, 40, 50))

        # Blit content surface onto panel with scroll offset
        panel.blit(content_surface, (0, -self.scroll_y))

        # Draw panel on screen
        self.screen.blit(panel, (self.screen.get_width() - self.panel_width - 10, 10))

        # Draw scrollbar if content is larger than panel
        if content_height > self.panel_height:
            scrollbar_height = max(20, (self.panel_height / content_height) * self.panel_height)
            scrollbar_pos = (self.panel_height - scrollbar_height) * (self.scroll_y / (content_height - self.panel_height))
            pygame.draw.rect(self.screen, (150, 150, 150),
                             (self.screen.get_width() - 20, 10 + scrollbar_pos, 10, scrollbar_height))

    def handle_events(self):
        """Handle pygame events including scrolling."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEWHEEL:
                # Calculate total content height for scrolling
                content_height = (40 + len(self.stats["current_wrestlers"]) * 110 +
                                 (30 + len(self.stats["eliminated"]) * 25 if self.stats["eliminated"] else 0))
                if content_height > self.panel_height:
                    self.scroll_y -= event.y * self.scroll_speed  # event.y is 1 for scroll up, -1 for scroll down
                    self.scroll_y = max(0, min(self.scroll_y, content_height - self.panel_height))  # Clamp scroll position
        return True

    def render(self, wrestlers, initiator=None, responder=None):
        """Render the current state of the match.
        
        Args:
            wrestlers: List of active wrestlers
            initiator: Wrestler initiating action (None if none)
            responder: Wrestler responding to action (None if none)
            
        Returns:
            bool: False if user requested quit, True otherwise
        """
        # Update stats and handle events
        self.initiator = initiator
        self.responder = responder
        self.stats["current_wrestlers"] = wrestlers
        if not self.handle_events():
            return False

        # Draw all components
        self.draw_ring()
        for i, wrestler in enumerate(wrestlers):
            screen_pos = self.pos_to_screen(wrestler.get_qpos(), i == 0)
            self.draw_humanoid(wrestler, screen_pos, i == 0)
        self.draw_stats_panel()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        return True

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()