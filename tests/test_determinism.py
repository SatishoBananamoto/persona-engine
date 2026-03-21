"""
Comprehensive tests for DeterminismManager.

Covers:
- Constructor with/without seed
- set_seed resets state
- Determinism: same seed produces same sequence
- Core random methods and call_count tracking
- add_noise (uniform and gaussian distributions)
- should_trigger (bernoulli trials)
- weighted_choice (weighted selection, all-zero weights, deterministic ordering)
- jitter_value (bounds enforcement, small jitter proximity)
- get_state (dict structure and values)
- reset_call_count
"""

import random

import pytest

from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Constructor
# ============================================================================


class TestConstructor:
    def test_init_with_seed(self):
        dm = DeterminismManager(seed=42)
        assert dm.seed == 42
        assert dm.call_count == 0
        assert dm.rng is not None

    def test_init_without_seed(self):
        dm = DeterminismManager()
        assert dm.seed is None
        assert dm.call_count == 0
        assert dm.rng is not None

    def test_init_with_zero_seed(self):
        dm = DeterminismManager(seed=0)
        assert dm.seed == 0
        assert dm.call_count == 0

    def test_init_with_negative_seed(self):
        dm = DeterminismManager(seed=-1)
        assert dm.seed == -1
        assert dm.call_count == 0


# ============================================================================
# set_seed
# ============================================================================


class TestSetSeed:
    def test_set_seed_updates_seed(self):
        dm = DeterminismManager(seed=1)
        dm.set_seed(99)
        assert dm.seed == 99

    def test_set_seed_resets_call_count(self):
        dm = DeterminismManager(seed=1)
        dm.random()
        dm.random()
        assert dm.call_count == 2
        dm.set_seed(99)
        assert dm.call_count == 0

    def test_set_seed_creates_new_rng(self):
        dm = DeterminismManager(seed=1)
        old_rng = dm.rng
        dm.set_seed(2)
        assert dm.rng is not old_rng

    def test_set_seed_produces_deterministic_sequence(self):
        dm = DeterminismManager(seed=1)
        dm.set_seed(42)
        val1 = dm.random()
        dm.set_seed(42)
        val2 = dm.random()
        assert val1 == val2


# ============================================================================
# Determinism: same seed -> same sequence
# ============================================================================


class TestDeterminism:
    def test_same_seed_same_random_sequence(self):
        dm1 = DeterminismManager(seed=12345)
        dm2 = DeterminismManager(seed=12345)
        for _ in range(20):
            assert dm1.random() == dm2.random()

    def test_same_seed_same_randint_sequence(self):
        dm1 = DeterminismManager(seed=12345)
        dm2 = DeterminismManager(seed=12345)
        for _ in range(20):
            assert dm1.randint(0, 100) == dm2.randint(0, 100)

    def test_same_seed_same_choice_sequence(self):
        dm1 = DeterminismManager(seed=12345)
        dm2 = DeterminismManager(seed=12345)
        items = ["alpha", "beta", "gamma", "delta"]
        for _ in range(20):
            assert dm1.choice(items) == dm2.choice(items)

    def test_same_seed_same_uniform_sequence(self):
        dm1 = DeterminismManager(seed=12345)
        dm2 = DeterminismManager(seed=12345)
        for _ in range(20):
            assert dm1.uniform(0.0, 10.0) == dm2.uniform(0.0, 10.0)

    def test_same_seed_same_gauss_sequence(self):
        dm1 = DeterminismManager(seed=12345)
        dm2 = DeterminismManager(seed=12345)
        for _ in range(20):
            assert dm1.gauss(0.0, 1.0) == dm2.gauss(0.0, 1.0)

    def test_different_seeds_different_sequences(self):
        dm1 = DeterminismManager(seed=1)
        dm2 = DeterminismManager(seed=2)
        vals1 = [dm1.random() for _ in range(10)]
        vals2 = [dm2.random() for _ in range(10)]
        assert vals1 != vals2


# ============================================================================
# Core Random Methods
# ============================================================================


class TestRandom:
    def test_random_returns_float_in_range(self):
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            val = dm.random()
            assert 0.0 <= val < 1.0

    def test_random_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        assert dm.call_count == 0
        dm.random()
        assert dm.call_count == 1
        dm.random()
        assert dm.call_count == 2
        dm.random()
        assert dm.call_count == 3


class TestRandint:
    def test_randint_returns_int_in_range(self):
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            val = dm.randint(5, 15)
            assert isinstance(val, int)
            assert 5 <= val <= 15

    def test_randint_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.randint(1, 10)
        dm.randint(1, 10)
        assert dm.call_count == 2

    def test_randint_equal_bounds(self):
        dm = DeterminismManager(seed=42)
        val = dm.randint(7, 7)
        assert val == 7


class TestChoice:
    def test_choice_returns_element_from_sequence(self):
        dm = DeterminismManager(seed=42)
        items = ["a", "b", "c", "d"]
        for _ in range(50):
            val = dm.choice(items)
            assert val in items

    def test_choice_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.choice([1, 2, 3])
        assert dm.call_count == 1

    def test_choice_single_element(self):
        dm = DeterminismManager(seed=42)
        val = dm.choice(["only"])
        assert val == "only"

    def test_choice_with_tuple(self):
        dm = DeterminismManager(seed=42)
        items = (10, 20, 30)
        val = dm.choice(items)
        assert val in items


class TestUniform:
    def test_uniform_returns_float_in_range(self):
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            val = dm.uniform(3.0, 7.0)
            assert 3.0 <= val <= 7.0

    def test_uniform_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.uniform(0.0, 1.0)
        dm.uniform(0.0, 1.0)
        dm.uniform(0.0, 1.0)
        assert dm.call_count == 3

    def test_uniform_negative_range(self):
        dm = DeterminismManager(seed=42)
        for _ in range(50):
            val = dm.uniform(-5.0, -1.0)
            assert -5.0 <= val <= -1.0


class TestGauss:
    def test_gauss_returns_float(self):
        dm = DeterminismManager(seed=42)
        val = dm.gauss(0.0, 1.0)
        assert isinstance(val, float)

    def test_gauss_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.gauss(0.0, 1.0)
        assert dm.call_count == 1

    def test_gauss_mean_approximation(self):
        """With many samples, the mean should approximate mu."""
        dm = DeterminismManager(seed=42)
        samples = [dm.gauss(5.0, 0.5) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        assert abs(mean - 5.0) < 0.1

    def test_gauss_zero_sigma(self):
        dm = DeterminismManager(seed=42)
        val = dm.gauss(3.0, 0.0)
        assert val == 3.0


# ============================================================================
# add_noise
# ============================================================================


class TestAddNoise:
    def test_add_noise_uniform_stays_within_budget(self):
        dm = DeterminismManager(seed=42)
        base_value = 0.5
        budget = 0.1
        for _ in range(200):
            noisy = dm.add_noise(base_value, budget, distribution="uniform")
            assert base_value - budget <= noisy <= base_value + budget

    def test_add_noise_uniform_default_distribution(self):
        """Default distribution should be uniform."""
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)
        val1 = dm1.add_noise(0.5, 0.1)
        val2 = dm2.add_noise(0.5, 0.1, distribution="uniform")
        assert val1 == val2

    def test_add_noise_gaussian(self):
        dm = DeterminismManager(seed=42)
        base_value = 0.5
        budget = 0.1
        noisy = dm.add_noise(base_value, budget, distribution="gaussian")
        assert isinstance(noisy, float)
        # Gaussian noise is not strictly bounded but should be close
        assert abs(noisy - base_value) < 1.0  # very generous bound

    def test_add_noise_gaussian_mean_approximation(self):
        """Gaussian noise centered at 0 should average close to the base value."""
        dm = DeterminismManager(seed=42)
        base = 0.5
        budget = 0.1
        samples = [dm.add_noise(base, budget, distribution="gaussian") for _ in range(5000)]
        mean = sum(samples) / len(samples)
        assert abs(mean - base) < 0.02

    def test_add_noise_zero_budget(self):
        dm = DeterminismManager(seed=42)
        val = dm.add_noise(0.75, 0.0)
        assert val == 0.75

    def test_add_noise_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.add_noise(0.5, 0.1, distribution="uniform")
        # add_noise calls uniform internally, which increments call_count
        assert dm.call_count == 1
        dm.add_noise(0.5, 0.1, distribution="gaussian")
        # add_noise with gaussian calls gauss internally
        assert dm.call_count == 2


# ============================================================================
# should_trigger
# ============================================================================


class TestShouldTrigger:
    def test_should_trigger_probability_zero(self):
        """Probability 0 should never trigger."""
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            assert dm.should_trigger(0.0) is False

    def test_should_trigger_probability_one(self):
        """Probability 1 should always trigger (random() returns [0,1))."""
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            assert dm.should_trigger(1.0) is True

    def test_should_trigger_returns_bool(self):
        dm = DeterminismManager(seed=42)
        result = dm.should_trigger(0.5)
        assert isinstance(result, bool)

    def test_should_trigger_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.should_trigger(0.5)
        # should_trigger calls random() which increments call_count
        assert dm.call_count == 1

    def test_should_trigger_deterministic(self):
        """Same seed should produce same trigger results."""
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)
        for _ in range(50):
            assert dm1.should_trigger(0.5) == dm2.should_trigger(0.5)

    def test_should_trigger_approximate_rate(self):
        """Over many trials the trigger rate should approximate the probability."""
        dm = DeterminismManager(seed=42)
        p = 0.3
        triggered = sum(dm.should_trigger(p) for _ in range(10000))
        rate = triggered / 10000
        assert abs(rate - p) < 0.05


# ============================================================================
# weighted_choice
# ============================================================================


class TestWeightedChoice:
    def test_weighted_choice_returns_valid_option(self):
        dm = DeterminismManager(seed=42)
        options = {"low": 0.1, "mid": 0.3, "high": 0.6}
        for _ in range(50):
            result = dm.weighted_choice(options)
            assert result in options

    def test_weighted_choice_deterministic(self):
        """Same seed produces same weighted choices."""
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)
        options = {"a": 0.2, "b": 0.3, "c": 0.5}
        for _ in range(30):
            assert dm1.weighted_choice(options) == dm2.weighted_choice(options)

    def test_weighted_choice_all_zero_weights(self):
        """All-zero weights should return first option alphabetically."""
        dm = DeterminismManager(seed=42)
        options = {"charlie": 0, "alpha": 0, "bravo": 0}
        result = dm.weighted_choice(options)
        assert result == "alpha"

    def test_weighted_choice_all_zero_no_call_count_increment(self):
        """All-zero weights returns early without calling random()."""
        dm = DeterminismManager(seed=42)
        options = {"b": 0, "a": 0}
        dm.weighted_choice(options)
        # No random call made because we return early
        assert dm.call_count == 0

    def test_weighted_choice_single_option(self):
        dm = DeterminismManager(seed=42)
        options = {"only_option": 1.0}
        result = dm.weighted_choice(options)
        assert result == "only_option"

    def test_weighted_choice_heavy_weight_dominates(self):
        """An option with overwhelming weight should be chosen most of the time."""
        dm = DeterminismManager(seed=42)
        options = {"rare": 0.001, "common": 0.999}
        results = [dm.weighted_choice(options) for _ in range(1000)]
        common_count = results.count("common")
        assert common_count > 950

    def test_weighted_choice_sorted_keys_for_determinism(self):
        """Insertion order should not matter because keys are sorted internally."""
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)
        options1 = {"a": 0.3, "b": 0.3, "c": 0.4}
        options2 = {"c": 0.4, "a": 0.3, "b": 0.3}
        for _ in range(30):
            assert dm1.weighted_choice(options1) == dm2.weighted_choice(options2)

    def test_weighted_choice_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.weighted_choice({"a": 0.5, "b": 0.5})
        assert dm.call_count == 1


# ============================================================================
# jitter_value
# ============================================================================


class TestJitterValue:
    def test_jitter_value_within_bounds(self):
        dm = DeterminismManager(seed=42)
        for _ in range(200):
            result = dm.jitter_value(target=0.5, min_val=0.0, max_val=1.0, jitter_amount=0.5)
            assert 0.0 <= result <= 1.0

    def test_jitter_value_large_jitter_still_clamped(self):
        """Even with extreme jitter the result must stay within [min_val, max_val]."""
        dm = DeterminismManager(seed=42)
        for _ in range(200):
            result = dm.jitter_value(target=0.5, min_val=0.0, max_val=1.0, jitter_amount=10.0)
            assert 0.0 <= result <= 1.0

    def test_jitter_value_small_jitter_stays_close(self):
        dm = DeterminismManager(seed=42)
        target = 0.5
        for _ in range(100):
            result = dm.jitter_value(target=target, min_val=0.0, max_val=1.0, jitter_amount=0.01)
            assert abs(result - target) < 0.02  # 1% of range=1.0, so max deviation 0.01

    def test_jitter_value_zero_jitter(self):
        dm = DeterminismManager(seed=42)
        result = dm.jitter_value(target=0.7, min_val=0.0, max_val=1.0, jitter_amount=0.0)
        assert result == 0.7

    def test_jitter_value_target_at_min(self):
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            result = dm.jitter_value(target=0.0, min_val=0.0, max_val=1.0, jitter_amount=0.1)
            assert 0.0 <= result <= 1.0

    def test_jitter_value_target_at_max(self):
        dm = DeterminismManager(seed=42)
        for _ in range(100):
            result = dm.jitter_value(target=1.0, min_val=0.0, max_val=1.0, jitter_amount=0.1)
            assert 0.0 <= result <= 1.0

    def test_jitter_value_increments_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.jitter_value(target=0.5, min_val=0.0, max_val=1.0)
        # jitter_value calls uniform internally
        assert dm.call_count == 1

    def test_jitter_value_deterministic(self):
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)
        r1 = dm1.jitter_value(0.5, 0.0, 1.0, 0.1)
        r2 = dm2.jitter_value(0.5, 0.0, 1.0, 0.1)
        assert r1 == r2

    def test_jitter_value_default_jitter_amount(self):
        """Default jitter_amount is 0.1."""
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)
        r1 = dm1.jitter_value(0.5, 0.0, 1.0)
        r2 = dm2.jitter_value(0.5, 0.0, 1.0, jitter_amount=0.1)
        assert r1 == r2


# ============================================================================
# get_state
# ============================================================================


class TestGetState:
    def test_get_state_with_seed(self):
        dm = DeterminismManager(seed=42)
        state = dm.get_state()
        assert state == {
            "seed": 42,
            "call_count": 0,
            "deterministic": True,
        }

    def test_get_state_without_seed(self):
        dm = DeterminismManager()
        state = dm.get_state()
        assert state == {
            "seed": None,
            "call_count": 0,
            "deterministic": False,
        }

    def test_get_state_after_calls(self):
        dm = DeterminismManager(seed=7)
        dm.random()
        dm.random()
        dm.random()
        state = dm.get_state()
        assert state["call_count"] == 3
        assert state["seed"] == 7
        assert state["deterministic"] is True

    def test_get_state_keys(self):
        dm = DeterminismManager(seed=1)
        state = dm.get_state()
        assert set(state.keys()) == {"seed", "call_count", "deterministic"}

    def test_get_state_after_set_seed(self):
        dm = DeterminismManager(seed=1)
        dm.random()
        dm.set_seed(99)
        state = dm.get_state()
        assert state["seed"] == 99
        assert state["call_count"] == 0
        assert state["deterministic"] is True


# ============================================================================
# reset_call_count
# ============================================================================


class TestResetCallCount:
    def test_reset_call_count_to_zero(self):
        dm = DeterminismManager(seed=42)
        dm.random()
        dm.random()
        dm.random()
        assert dm.call_count == 3
        dm.reset_call_count()
        assert dm.call_count == 0

    def test_reset_call_count_does_not_affect_sequence(self):
        """Resetting call_count should not change the RNG state / sequence."""
        dm1 = DeterminismManager(seed=42)
        dm2 = DeterminismManager(seed=42)

        # Advance both by 3 calls
        for _ in range(3):
            dm1.random()
            dm2.random()

        # Reset call_count on dm1 only
        dm1.reset_call_count()

        # Next random values should still match
        assert dm1.random() == dm2.random()

    def test_reset_call_count_when_already_zero(self):
        dm = DeterminismManager(seed=42)
        assert dm.call_count == 0
        dm.reset_call_count()
        assert dm.call_count == 0

    def test_reset_call_count_does_not_change_seed(self):
        dm = DeterminismManager(seed=42)
        dm.random()
        dm.reset_call_count()
        assert dm.seed == 42


# ============================================================================
# Mixed call_count tracking across method types
# ============================================================================


class TestCallCountAcrossMethods:
    def test_mixed_method_calls_increment_call_count(self):
        dm = DeterminismManager(seed=42)
        dm.random()         # 1
        dm.randint(1, 10)   # 2
        dm.choice([1, 2])   # 3
        dm.uniform(0, 1)    # 4
        dm.gauss(0, 1)      # 5
        assert dm.call_count == 5

    def test_should_trigger_increments_via_random(self):
        dm = DeterminismManager(seed=42)
        dm.should_trigger(0.5)  # calls random() -> +1
        assert dm.call_count == 1

    def test_add_noise_uniform_increments_via_uniform(self):
        dm = DeterminismManager(seed=42)
        dm.add_noise(0.5, 0.1, "uniform")  # calls uniform() -> +1
        assert dm.call_count == 1

    def test_add_noise_gaussian_increments_via_gauss(self):
        dm = DeterminismManager(seed=42)
        dm.add_noise(0.5, 0.1, "gaussian")  # calls gauss() -> +1
        assert dm.call_count == 1

    def test_weighted_choice_increments_once(self):
        dm = DeterminismManager(seed=42)
        dm.weighted_choice({"a": 0.5, "b": 0.5})  # calls random() -> +1
        assert dm.call_count == 1

    def test_jitter_value_increments_via_uniform(self):
        dm = DeterminismManager(seed=42)
        dm.jitter_value(0.5, 0.0, 1.0)  # calls uniform() -> +1
        assert dm.call_count == 1
