"""Bounded sampling for filters that allow rejected groups as a fallback."""

from collections import deque


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise ValueError("expected a nonnegative integer")
    return number


class Resampling:
    def __init__(self, args):
        self.target = args.rollout_batch_size
        self.max_extra_rounds = getattr(args, "max_extra_sampling_rounds", 3)
        self.draws = 0
        self.extra_groups = 0
        self.fallback_fills = 0
        self.reserve = deque()

    def on_draw(self, count):
        if self.draws:
            self.extra_groups += count
        self.draws += 1

    def defer(self, output, group):
        if not output.keep and output.keep_when_insufficient and len(self.reserve) < self.target:
            self.reserve.append(group)
            return True
        return False

    def fill(self, remaining):
        # Wait for in-flight candidates before committing to fallback groups.
        if self.draws <= self.max_extra_rounds:
            return []
        groups = [self.reserve.popleft() for _ in range(min(max(0, self.target - remaining), len(self.reserve)))]
        self.fallback_fills += len(groups)
        return groups

    def metrics(self):
        return {
            "rollout/resampling_extra_groups": self.extra_groups,
            "rollout/resampling_extra_rounds": max(0, self.draws - 1),
            "rollout/resampling_fallback_fills": self.fallback_fills,
        }
