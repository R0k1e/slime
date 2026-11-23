from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None
    # Keep a rejected group when dropping it would leave too few candidates to
    # fill the rollout batch. This avoids launching another oversampling round.
    keep_when_insufficient: bool = False
    # Sum and count of trajectory rewards before the hook runs. Carry only
    # these scalars back from workers, including when rejected data is released.
    raw_reward_stats: tuple[float, int] | None = None
    environment_metrics: tuple[tuple[str, float, int], ...] = ()


def should_drop_dynamic_filter_output(
    output: DynamicFilterOutput,
    *,
    remaining_batch_size: int,
    target_data_size: int,
) -> bool:
    if output.keep:
        return False
    if output.keep_when_insufficient and remaining_batch_size <= target_data_size:
        return False
    return True


def call_dynamic_filter(fn, args, samples, **kwargs):
    trajectories = [sample[-1] if isinstance(sample, list) else sample for sample in samples]
    environment_metrics = tuple(trajectory_metrics(trajectories))
    reward_stats = None
    if fn is not None or getattr(args, "rollout_sample_filter_path", None) is not None:
        # Context-window segments share one outcome; count the terminal reward
        # once per trajectory, before a custom filter can mutate it.
        if all(sample.reward is not None for sample in trajectories):
            rewards = [sample.get_reward_value(args) for sample in trajectories]
            reward_stats = (float(sum(rewards)), len(rewards))

    output = fn(args, samples, **kwargs) if fn is not None else True

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return replace(output, raw_reward_stats=reward_stats, environment_metrics=environment_metrics)


def trajectory_metrics(samples):
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        environment = metadata.get("environment") or {}
        components = environment.get("reward_components") or {}
        if isinstance(components, dict):
            for key, value in components.items():
                if isinstance(key, str) and type(value) in (int, float) and isfinite(value):
                    yield f"rollout/reward_components/{key}", float(value), 1
        errors = metadata.get("environment_errors") or {}
        for key in ("tool", "user_model"):
            value = errors.get(key)
            if type(value) is int and value >= 0:
                yield f"rollout/errors/{key}", float(value), 0


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._raw_reward_sum = 0.0
        self._raw_reward_count = 0
        self._raw_rewards_complete = True
        self._environment_sums = defaultdict(float)
        self._environment_counts = defaultdict(int)

    def on_group_completed(self, output: DynamicFilterOutput):
        for key, value, count in getattr(output, "environment_metrics", ()):
            self._environment_sums[key] += value
            self._environment_counts[key] += count
        if output.raw_reward_stats is None:
            # Older restored results may lack pre-filter observations. Do not
            # label a partial mean as the reward of the whole candidate batch.
            self._raw_rewards_complete = False
            return
        reward_sum, reward_count = output.raw_reward_stats
        self._raw_reward_sum += reward_sum
        self._raw_reward_count += reward_count

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._raw_rewards_complete and self._raw_reward_count:
            metrics["rollout/raw_reward_before_filters"] = self._raw_reward_sum / self._raw_reward_count
        metrics.update({key: value / (self._environment_counts[key] or 1)
                        for key, value in self._environment_sums.items()})
        return metrics
