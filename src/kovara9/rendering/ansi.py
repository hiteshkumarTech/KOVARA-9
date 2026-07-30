"""Portable text renderer for terminals and CI diagnostics."""

from kovara9.core.types import Position, WorldSnapshot


class AnsiRenderer:
    """Render a world snapshot as deterministic plain text."""

    def render(self, snapshot: WorldSnapshot) -> str:
        """Return one character per grid cell plus episode metadata."""

        agents = {position: agent for agent, position in snapshot.agent_positions.items()}
        lines: list[str] = []
        for row in range(snapshot.height):
            cells: list[str] = []
            for col in range(snapshot.width):
                position = Position(row, col)
                if snapshot.obstacles[row, col]:
                    symbol = "#"
                elif position in agents:
                    symbol = agents[position].removeprefix("agent_")[-1]
                elif position in snapshot.recovered_targets:
                    symbol = "x"
                elif position in snapshot.targets:
                    symbol = "T"
                else:
                    symbol = "."
                cells.append(symbol)
            lines.append("".join(cells))
        recovered = len(snapshot.recovered_targets)
        lines.append(f"step={snapshot.step_count} recovered={recovered}/{len(snapshot.targets)}")
        return "\n".join(lines)
