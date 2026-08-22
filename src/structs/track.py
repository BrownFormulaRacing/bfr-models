from dataclasses import dataclass, field


@dataclass
class AutocrossTrack:
    lengths: list[float] = field(default_factory=list)
    curvatures: list[float] = field(default_factory=list)

    def load(self, path: str) -> None:
        self.lengths = []
        self.curvatures = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    self.lengths.append(float(parts[0]))
                    self.curvatures.append(float(parts[1]))

@dataclass
class EnduranceTrack:
    lengths: list[float] = field(default_factory=list)
    curvatures: list[float] = field(default_factory=list)


@dataclass
class SKidpadTrack:
    inner_radius: float = 7.625
    outer_radius: float = 9.125