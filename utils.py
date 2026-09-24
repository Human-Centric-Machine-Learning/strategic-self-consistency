"""
Shared functionality for simulating faithful and unfaithful LLM inference 
with auditing.

Supported stopping rules:
    - Adaptive Self-Consistency (ASC), 
    - PPR-1v1,
    - Early-Stopping Self-Consistency (ESC)
    - Adaptive best-of-N
    
Count-based stopping computations are implemented incrementally and cached 
to avoid repeatedly recomputing statistics over entire prefixes.
"""

import numpy as np
import math
from collections import Counter
import hashlib
from functools import lru_cache
from scipy.special import betaincc

MAX_SEQ_LEN = 5000

def stable_seed(path, qid):
    """
    Use deterministic seeding 
    """
    key = f"{path}|{qid}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little")

def alg2_slow(seq, stop_rule):
    """
    Algorithm 2 for an arbitrary stopping rule.
    """
    N = len(seq)
    if N == 0:
        return {"paths": 0}

    for j in range(1, N):
        if stop_rule(seq[:j]):
            return {"paths": 0}

    if N == 1:
        return {"paths": 1}

    paths = [0] * (N + 1)
    paths[1] = 1

    for j in range(2, N + 1):
        prev_paths = 0
        for L in range(0, j - 1):
            possible_multiple_swap = True
            for k in range(1, L + 1):
                # Length of prefix before inserting y_j.
                idx = j - k - 1
                test_seq = (seq[:idx] + [seq[j - 1]])
                if not stop_rule(test_seq):
                    possible_multiple_swap = False
                    break

            if possible_multiple_swap:
                prev_paths += paths[j - L - 1]

        paths[j] = prev_paths

    return {"paths": paths[N]}


def _is_fast_asc_rule(stop_rule):
    """
    Return True for stop rules produced by make_asc_stop_rule().
    """
    return bool(getattr(stop_rule, "_is_asc_stop_rule", False))


def _asc_triggers_from_top_two(stop_rule, a, b, n):
    """
    Evaluate an ASC rule from top-two counts without rebuilding a sequence.
    """
    if n < stop_rule._min_samples:
        return False
    return asc_confidence_from_counts(a, b) >= stop_rule._confidence


def _top_two_from_counter(counts):
    """
    Top-two counts + leader from an already-built Counter.
    """
    if not counts:
        return 0, 0, None, 0

    ranked = counts.most_common(2)
    a = ranked[0][1]
    b = ranked[1][1] if len(ranked) > 1 else 0
    leader = ranked[0][0]
    mult = sum(c == a for c in counts.values())
    return a, b, leader, mult


def _top_two_after_increment(counts, a, b, mult_a, value):
    """
    Return the top-two count values after adding one occurrence of `value`.

    O(1) given the current top-two counts, rather than rebuilding a
    Counter for prefix + [value].
    """
    c = counts.get(value, 0)

    if c == a:
        # Incrementing a current leader.
        if mult_a >= 2:
            # Break a tie at the top: (a, a) -> (a+1, a).
            return a + 1, a
        return a + 1, b

    # The current leader remains at count a.  The incremented class can only
    # affect the runner-up count (possibly creating a tie at a).
    return a, max(b, c + 1)


def _alg2_fast_asc(seq, stop_rule):
    """
    ASC-specialized implementation of Algorithm 2.

    A generic implementation repeatedly calls stop_rule(prefix+[v]).
    Here we maintain prefix counts once and evaluate hypothetical appends 
    directly from the top-two counts, reducing the expensive stopping-rule 
    portion.

    No prefix slice is actually constructed when checking whether an earlier
    prefix has already stopped.
    """
    N = len(seq)

    if N == 0:
        return {"paths": 0}

    # Alg 2 is zero if any proper prefix already stopped.
    # end = N - 1 avoids constructing seq[:-1].
    earlier = find_stopping_step(seq, stop_rule, end=N - 1)
    if earlier["stopped"]:
        return {"paths": 0}

    if N == 1:
        return {"paths": 1}

    values = set(seq)
    trigger_run = {v: [0] * N for v in values}

    counts = Counter()

    # m denotes the current prefix length, so counts represents seq[:m].
    for m in range(1, N):
        counts[seq[m - 1]] += 1
        a, b, _, mult_a = _top_two_from_counter(counts)
        test_n = m + 1

        for v in values:
            aa, bb = _top_two_after_increment(counts, a, b, mult_a, v)
            if _asc_triggers_from_top_two(stop_rule, aa, bb, test_n):
                previous = trigger_run[v][m - 1] if m > 1 else 0
                trigger_run[v][m] = previous + 1

    # Same window-sum recurrence as the generic implementation.
    paths = [0] * (N + 1)
    prefix_sum = [0] * (N + 1)
    paths[1] = 1
    prefix_sum[1] = 1

    for j in range(2, N + 1):
        m0 = j - 2
        yj = seq[j - 1]
        Lmax = trigger_run[yj][m0] if m0 >= 1 else 0

        lower = j - 2 - Lmax
        paths[j] = prefix_sum[j - 1] - prefix_sum[lower]
        prefix_sum[j] = prefix_sum[j - 1] + paths[j]

    return {"paths": paths[N]}


def alg2_fast(seq, stop_rule):
    """
    O(N'K) version of Alg. 2 with speedups for ASC and ESC and the original generic fallback for all other stopping rules.
    """
    if _is_fast_asc_rule(stop_rule):
        return _alg2_fast_asc(seq, stop_rule)

    if _is_fast_esc_rule(stop_rule):
        return _alg2_fast_esc(seq, stop_rule)
    
    N = len(seq)
    if N == 0:
        return {"paths": 0}

    for j in range(1, N):
        if stop_rule(seq[:j]):
            return {"paths": 0}

    if N == 1:
        return {"paths": 1}

    values = set(seq)

    # trigger_run[v][m]: number of consecutive prefixes ending at prefix-length m
    # for which appending value v would trigger tau.
    trigger_run = {v: [0] * N for v in values}

    for m in range(1, N):
        prefix = seq[:m]

        for v in values:
            triggers = stop_rule(prefix + [v])
            if triggers:
                previous = trigger_run[v][m - 1] if m > 1 else 0
                trigger_run[v][m] = previous + 1

    paths = [0] * (N + 1)
    prefix_sum = [0] * (N + 1)
    paths[1] = 1
    prefix_sum[1] = 1

    for j in range(2, N + 1):
        m0 = j - 2
        yj = seq[j - 1]
        Lmax = trigger_run[yj][m0] if m0 >= 1 else 0

        lower = j - 2 - Lmax
        paths[j] = prefix_sum[j - 1] - prefix_sum[lower]
        prefix_sum[j] = prefix_sum[j - 1] + paths[j]

    return {"paths": paths[N]}


def _is_fast_esc_rule(stop_rule):
    """Return True for stop rules produced by make_esc_stop_rule()."""
    return bool(getattr(stop_rule, "_is_esc_stop_rule", False))


def _find_stopping_step_esc(seq, stop_rule, end=None):
    """
    Find the first ESC stopping time in seq[:end] in O(N), without slicing.

    ESC stops once `window` consecutive answers agree. Counts are maintained
    incrementally so that the returned top-two statistics also require no
    prefix copy.
    """
    if end is None:
        end = len(seq)
    else:
        end = max(0, min(int(end), len(seq)))

    if end == 0:
        return {
            "stopped": False,
            "step": 0,
            "d": 0,
            "leader": None,
        }

    window = stop_rule._window
    counts = Counter()

    run_length = 0
    previous = None

    for i in range(end):
        value = seq[i]
        counts[value] += 1

        if i == 0 or value != previous:
            run_length = 1
        else:
            run_length += 1

        previous = value

        if run_length >= window:
            a, b, leader, _ = _top_two_from_counter(counts)
            return {
                "stopped": True,
                "step": i + 1,
                "d": a - b,
                "leader": leader,
            }

    a, b, leader, _ = _top_two_from_counter(counts)

    return {
        "stopped": False,
        "step": end,
        "d": a - b,
        "leader": leader,
    }


def _alg2_fast_esc(seq, stop_rule):
    """
    ESC-specialized implementation of Algorithm 2.

    For an ESC window w, appending value v to a prefix of length m
    triggers iff the prefix ends in at least w - 1 consecutive copies
    of v. This lets us compute the trigger-run quantity directly from
    trailing run lengths instead of testing every prefix/value pair.
    """
    N = len(seq)

    if N == 0:
        return {"paths": 0}

    window = stop_rule._window

    # If any proper prefix already stops, Alg 2 returns 0.
    # Do this directly without repeatedly constructing prefixes.
    run_length = 1

    for i in range(1, N - 1):
        if seq[i] == seq[i - 1]:
            run_length += 1
        else:
            run_length = 1

        if run_length >= window:
            return {"paths": 0}

    if N == 1:
        return {"paths": 1}

    # trailing_run[m] = length of the consecutive run ending at seq[m-1], i.e., at the end of prefix seq[:m].
    trailing_run = [0] * (N + 1)

    for m in range(1, N + 1):
        if m == 1 or seq[m - 1] != seq[m - 2]:
            trailing_run[m] = 1
        else:
            trailing_run[m] = trailing_run[m - 1] + 1

    # Same window-sum recurrence as alg2_fast.
    paths = [0] * (N + 1)
    prefix_sum = [0] * (N + 1)

    paths[1] = 1
    prefix_sum[1] = 1

    for j in range(2, N + 1):
        m0 = j - 2
        yj = seq[j - 1]

        # trigger_run[yj][m0] can be obtained directly.
        # Appending yj triggers ESC iff seq[:m0]
        # ends with at least window-1 copies of yj.
        if (m0 >= 1 and seq[m0 - 1] == yj and trailing_run[m0] >= window - 1):
            Lmax = trailing_run[m0] - window + 2
        else:
            Lmax = 0

        lower = j - 2 - Lmax
        paths[j] = prefix_sum[j - 1] - prefix_sum[lower]
        prefix_sum[j] = prefix_sum[j - 1] + paths[j]

    return {"paths": paths[N]}


def _stop_rule_at_prefix(seq, end, stop_rule):
    """
    Evaluate stop_rule(seq[:end]) without materializing the prefix for
    optimized ASC and ESC stopping rules.

    `end` is exclusive, matching Python slicing.
    """
    end = max(0, min(int(end), len(seq)))

    if end == 0:
        return False

    if _is_fast_esc_rule(stop_rule):
        window = stop_rule._window

        if end < window:
            return False

        last = seq[end - 1]
        for i in range(end - window, end - 1):
            if seq[i] != last:
                return False

        return True

    if _is_fast_asc_rule(stop_rule):
        if end < stop_rule._min_samples:
            return False

        counts = Counter()
        for i in range(end):
            counts[seq[i]] += 1

        a, b, _, _ = _top_two_from_counter(counts)
        return _asc_triggers_from_top_two(stop_rule, a, b, end)

    # Generic fallback: arbitrary rules may depend on the full ordered prefix.
    return stop_rule(seq[:end])


def find_stopping_step(seq, stop_rule, end=None):
    """
    Find the first place where tau triggers in seq[:end].

    `end` is exclusive, matching Python slicing semantics. For ASC and ESC,
    the prefix is processed directly without materializing seq[:end].
    Other stopping rules retain the original generic behavior.
    """
    if end is None:
        end = len(seq)
    else:
        end = max(0, min(int(end), len(seq)))

    if end == 0:
        return {
            "stopped": False,
            "step": 0,
            "d": 0,
            "leader": None,
        }

    if _is_fast_esc_rule(stop_rule):
        return _find_stopping_step_esc(seq, stop_rule, end=end)

    if _is_fast_asc_rule(stop_rule):
        counts = Counter()

        for i in range(end):
            value = seq[i]
            t = i + 1
            counts[value] += 1

            if t < stop_rule._min_samples:
                continue

            a, b, leader, _ = _top_two_from_counter(counts)
            if _asc_triggers_from_top_two(stop_rule, a, b, t):
                return {
                    "stopped": True,
                    "step": t,
                    "d": a - b,
                    "leader": leader,
                }

        a, b, leader, _ = _top_two_from_counter(counts)
        return {
            "stopped": False,
            "step": end,
            "d": a - b,
            "leader": leader,
        }

    # Generic fallback. This path may still construct prefixes because an
    # arbitrary stopping rule can depend on the full ordered prefix.
    for t in range(1, end + 1):
        prefix = seq[:t]

        if stop_rule(prefix):
            a, b, leader, _ = top_two_stats(prefix)
            return {
                "stopped": True,
                "step": t,
                "d": a - b,
                "leader": leader,
            }

    # Only one copy for the generic fallback's final summary when `end`
    # excludes a suffix.
    final_seq = seq if end == len(seq) else seq[:end]
    a, b, leader, _ = top_two_stats(final_seq)

    return {
        "stopped": False,
        "step": end,
        "d": a - b,
        "leader": leader,
    }


def simulate_provider(samples, T_max=128, alpha=0.1, stop_rule=None, audit_probs=None, rng=None):
    """
    Run faithful stopping + one-step-lookahead continuation attack.

    If the faithful procedure requires more than T_max real samples,
    additional samples are drawn with replacement from the empirical
    distribution of the original reservoir.

    MAX_SEQ_LEN caps only the faithful stopping time N. Once a faithful
    stop is found, the continuation is not capped by MAX_SEQ_LEN.

    The continuation loop performs tentative swaps in place so that it does
    not repeatedly allocate O(N)-sized copies such as seq[:-1] or
    seq[:-1] + [new_sample, certificate].
    """

    samples = list(samples)

    cap = min(T_max, len(samples))
    reservoir = samples[:cap]

    if cap == 0:
        return None

    if stop_rule is None:
        raise ValueError("stop_rule must be provided")

    if rng is None:
        rng = np.random.default_rng()

    # Auditor's plug-in estimate of P.
    # IMPORTANT: this is always based only on the real reservoir,
    # not on empirically generated continuation samples.
    if audit_probs is None:
        reservoir_counts = Counter(reservoir)
        audit_probs = {ans: count / cap for ans, count in reservoir_counts.items()}

    faithful_seq = []
    faithful_source_indices = []
    used_empirical_extension = False

    while len(faithful_seq) < MAX_SEQ_LEN:
        t = len(faithful_seq)

        if t < cap:
            # Consume available real generations in their original order.
            source_idx = t
        else:
            # Once the reservoir is exhausted, bootstrap from its empirical
            # distribution.
            used_empirical_extension = True
            source_idx = int(rng.integers(cap))

        new_sample = reservoir[source_idx]
        faithful_seq.append(new_sample)
        faithful_source_indices.append(source_idx)

        # Because we check after every append, this is the first stopping time.
        if stop_rule(faithful_seq):
            break

    faithful_n = len(faithful_seq)

    # Safety cutoff: even empirical continuation did not stop by MAX_SEQ_LEN.
    faithful_stopped = stop_rule(faithful_seq)
    a, b, _, _ = top_two_stats(faithful_seq)
    faithful_d = a - b
    faithful_mode = empirical_mode(faithful_seq)

    if not faithful_stopped:
        return {
            "samples": reservoir,
            "faithful_seq": faithful_seq,
            "faithful_mode": faithful_mode,
            "faithful_d": faithful_d,
            "faithful_n": faithful_n,
            "faithful_stopped": False,
            "adversarial_seq": faithful_seq,
            "adversarial_n": faithful_n,
            "unfaithful_mode": faithful_mode,
            "audit_safe_n": faithful_n,
            "generation_source_indices": faithful_source_indices,
            "faithful_source_indices": faithful_source_indices,
            "hit_cap": faithful_n >= cap,
            "exceeded_cap": faithful_n > cap,
            "used_empirical_extension": used_empirical_extension,
            "adversarial_triggers": False,
        }

    # Unfaithful continuation.
    adversarial_seq = list(faithful_seq)
    generation_source_indices = list(faithful_source_indices)
    audit_safe_n = None

    while True:
        t = len(adversarial_seq)
        if t - faithful_n >= MAX_SEQ_LEN:
            break

        if t < cap:
            # Use the next real generation when one remains.
            source_idx = t
        else:
            # Otherwise sample from the empirical reservoir.
            used_empirical_extension = True
            source_idx = int(rng.integers(cap))

        new_sample = reservoir[source_idx]

        # No prefix before the current final element has already stopped. 
        # Verify it without constructing adversarial_seq[:-1].
        earlier = find_stopping_step(
            adversarial_seq,
            stop_rule,
            end=t - 1,
        )
        if earlier["stopped"]:
            raise AssertionError(
                f"prefix stopped at {earlier['step']} "
                f"but current length is {t}."
            )

        # Current sequence triggers; try to defer its final sample using
        # one-step lookahead.
        if stop_rule(adversarial_seq):
            certificate = adversarial_seq[-1]
            certificate_source = generation_source_indices[-1]

            # Swapping identical values cannot delay stopping.
            if new_sample == certificate:
                break

            adversarial_seq[-1] = new_sample
            adversarial_seq.append(certificate)

            generation_source_indices[-1] = source_idx
            generation_source_indices.append(certificate_source)

            rollback = False

            try:
                # Audit the tentative sequence directly; no `swapped` copy.
                if alpha is not None:
                    paths = alg2_fast(adversarial_seq, stop_rule)["paths"]
                    terminal_value = adversarial_seq[-1]
                    p_terminal = audit_probs.get(terminal_value, 0.0)
                    likelihood_ratio = paths * p_terminal

                    if (
                        audit_safe_n is None
                        and likelihood_ratio >= 1.0 / alpha
                    ):
                        # The last audit-safe sequence had length t.
                        audit_safe_n = t

                # Equivalent to stop_rule(swapped[:-1]), but avoids constructing the prefix 
                if _stop_rule_at_prefix(adversarial_seq, len(adversarial_seq) - 1, stop_rule):
                    rollback = True

            except Exception:
                # Restore the pre-swap state before propagating the exception.
                adversarial_seq.pop()
                adversarial_seq[-1] = certificate
                generation_source_indices.pop()
                generation_source_indices[-1] = certificate_source
                raise

            if rollback:
                # Undo tentative swap in O(1).
                adversarial_seq.pop()
                adversarial_seq[-1] = certificate
                generation_source_indices.pop()
                generation_source_indices[-1] = certificate_source
                break

            # Otherwise the tentative in-place representation is exactly the
            # accepted adversarial sequence, so no further assignment is needed.

        else:
            # Current sequence does not trigger yet: append normally.
            adversarial_seq.append(new_sample)
            generation_source_indices.append(source_idx)

    adversarial_n = len(adversarial_seq)

    if audit_safe_n is None:
        audit_safe_n = adversarial_n

    # Final invariant check without adversarial_seq[:-1].
    earlier = find_stopping_step(adversarial_seq, stop_rule, end=adversarial_n - 1)
    assert not earlier["stopped"]
    assert len(generation_source_indices) == adversarial_n

    return {
        "samples": reservoir,
        "faithful_seq": faithful_seq,
        "faithful_mode": faithful_mode,
        "faithful_d": faithful_d,
        "faithful_n": faithful_n,
        "faithful_stopped": True,
        "adversarial_seq": adversarial_seq,
        "adversarial_n": adversarial_n,
        "unfaithful_mode": empirical_mode(adversarial_seq),
        "audit_safe_n": audit_safe_n,
        # Maps every generated answer back to the real reservoir generation
        # from which it was drawn.
        "generation_source_indices": generation_source_indices,
        "faithful_source_indices": faithful_source_indices,
        "hit_cap": adversarial_n >= cap,
        "exceeded_cap": adversarial_n > cap,
        "used_empirical_extension": used_empirical_extension,
        "adversarial_triggers": stop_rule(adversarial_seq),
    }


def extend_until_faithful_stop(reservoir, stop_rule, rng=None, max_seq_len=MAX_SEQ_LEN):
    """
    Follow the real pre-generated reservoir in order.

    If the stopping rule has not fired by the end of the reservoir,
    continue sampling with replacement from the empirical distribution
    defined by the ORIGINAL reservoir until the rule fires.

    source_indices[j] gives the index in the original reservoir from
    which sample j came. This is also useful for token accounting.
    """
    reservoir = list(reservoir)
    cap = len(reservoir)

    if cap == 0:
        return {
            "stopped": False,
            "seq": [],
            "source_indices": [],
            "used_empirical_extension": False,
        }

    if rng is None:
        rng = np.random.default_rng()

    seq = []
    source_indices = []

    # First consume the real samples in their original order.
    for i, sample in enumerate(reservoir):
        seq.append(sample)
        source_indices.append(i)
        if stop_rule(seq):
            return {
                "stopped": True,
                "seq": seq,
                "source_indices": source_indices,
                "used_empirical_extension": False,
            }

    # No faithful stop within N_max. Continue from the empirical
    # distribution of the ORIGINAL N_max-sample reservoir.
    while len(seq) < max_seq_len:
        source_idx = int(rng.integers(cap))
        seq.append(reservoir[source_idx])
        source_indices.append(source_idx)

        if stop_rule(seq):
            return {
                "stopped": True,
                "seq": seq,
                "source_indices": source_indices,
                "used_empirical_extension": True,
            }

    # Safety cutoff only.
    return {
        "stopped": False,
        "seq": seq,
        "source_indices": source_indices,
        "used_empirical_extension": True,
    }



"""
Generic count utilities
"""
def top_two_stats(seq):
    """
    Returns:
        a       count of empirical leader
        b       count of empirical runner-up
        leader  empirical leader value
        mult    number of classes tied for first
    """
    counts = Counter(seq)

    if not counts:
        return 0, 0, None, 0

    ranked = counts.most_common()

    a = ranked[0][1]
    b = ranked[1][1] if len(ranked) >= 2 else 0
    leader = ranked[0][0]

    mult = sum(c == a for c in counts.values())
    return a, b, leader, mult


def empirical_mode(seq):
    if len(seq) == 0:
        return None
    return Counter(seq).most_common(1)[0][0]


"""
PPR-1v1 stopping rule
"""
def _log_beta_pdf(x, a, b):
    """
    log PDF at x for Beta(a+1, b+1),  evaluated in log-space so that it's safe 
    for the large counts (N ~ 900) that appear in the simulations; the binomial 
    term overflows float otherwise.
    """
    if not (0.0 < x < 1.0):
        return -np.inf

    return a * math.log(x) + b * math.log(1.0 - x) + math.lgamma(a + b + 2) - math.lgamma(a + 1) - math.lgamma(b + 1)


def ppr_stop_condition(a, b, delta, K, epsilon=0.0):
    if K < 2:
        return False

    threshold = 0.5 - epsilon
    log_pdf = _log_beta_pdf(threshold, a, b)
    return log_pdf <= math.log(delta / (K - 1))


def make_ppr_stop_rule(K, delta, epsilon=0.0, min_samples=4):
    """
    Return a callable tau(seq) for PPR-1v.
    """
    def stop_rule(seq):
        if len(seq) < min_samples:
            return False
        a, b, _, _ = top_two_stats(seq)
        return ppr_stop_condition(a, b, delta, K, epsilon)

    return stop_rule


"""
Adaptive-Consistency (ASC) methods
"""
@lru_cache(maxsize=None)
def asc_confidence_from_counts(a, b):
    """
    Posterior probability that the empirical leader has probability > 0.5
    under a Beta(1,1) prior.

    Equivalent to beta.sf(0.5, a + 1, b + 1), but avoids scipy.stats 
    overhead and caches each distinct (a, b) state.
    """
    return float(betaincc(a + 1, b + 1, 0.5))


def asc_confidence(seq):
    if len(seq) == 0:
        return 0.0

    a, b, _, _ = top_two_stats(seq)
    return asc_confidence_from_counts(a, b)


def make_asc_stop_rule(confidence=0.95, min_samples=1):
    """
    Beta Adaptive-Consistency stopping rule.

    The returned object is an ordinary callable tau(seq) -> bool, but small metadata attributes 
    let find_stopping_step() and alg2_fast() use incremental-count ASC fast paths.
    """
    confidence = float(confidence)
    min_samples = int(min_samples)

    def stop_rule(seq):
        if len(seq) < min_samples:
            return False

        a, b, _, _ = top_two_stats(seq)
        return asc_confidence_from_counts(a, b) >= confidence

    # Markers/parameters used internally for speedup. 
    # Callers can continue treating stop_rule exactly like a normal function.
    stop_rule._is_asc_stop_rule = True
    stop_rule._confidence = confidence
    stop_rule._min_samples = min_samples

    return stop_rule


def make_esc_stop_rule(window=5):
    """
    Early-Stopping Self-Consistency (ESC).

    Stop once the most recent `window` answers all agree.

    Info on the returned callable allows faster
    implementations of find_stopping_step() and alg2_fast().
    """
    window = int(window)

    def stop_rule(seq):
        if len(seq) < window:
            return False

        last = seq[-1]

        # Avoid constructing set(seq[-window:]).
        for i in range(2, window + 1):
            if seq[-i] != last:
                return False

        return True

    stop_rule._is_esc_stop_rule = True
    stop_rule._window = window

    return stop_rule

def make_threshold_bon_stop_rule(reward_threshold, min_samples=1):
    """
    Adaptive best-of-N stopping rule
    """
    # TODO - use avg reward
    def stop_rule(reward_seq):
        if len(reward_seq) < min_samples:
            return False

        return max(reward_seq) >= reward_threshold

    return stop_rule


def simulate_adaptive_bon_provider(rewards, audit_trigger_prob, T_max=128, alpha=0.1, stop_rule=None, rng=None):
    """
    Run faithful adaptive best-of-N stopping + continuation (mirrors simulate_provider).
    """

    rewards = list(rewards)
    cap = min(T_max, len(rewards))
    reservoir = rewards[:cap]


    if rng is None:
        rng = np.random.default_rng()

    if stop_rule is None:
        raise ValueError("stop_rule must be provided")

    stop = find_stopping_step(reservoir, stop_rule)

    if not stop["stopped"]:
        return {
            "samples": reservoir,
            "faithful_seq": reservoir,
            "faithful_n": cap,
            "faithful_stopped": False,
            "adversarial_seq": reservoir,
            "adversarial_n": cap,
            "audit_safe_n": cap,
            "generation_source_indices": list(range(cap)),
            "hit_cap": True,
            "exceeded_cap": False,
            "used_empirical_extension": False
        }

    faithful_n = stop["step"]
    faithful_seq = list(reservoir[:faithful_n])

    # Plug-in probability used by the auditor
    # For threshold BoN: q = P(R >= r*)
    if alpha is not None and audit_trigger_prob is None:
        raise ValueError("audit_trigger_prob must be provided when alpha is not None")

    adversarial_seq = list(faithful_seq)
    generation_source_indices = list(range(faithful_n))

    audit_safe_n = None
    used_empirical_extension = False

    while True:
        t = len(adversarial_seq)
        if t - faithful_n >= MAX_SEQ_LEN:
            break
           

        earlier = find_stopping_step(adversarial_seq[:-1], stop_rule)
        if earlier["stopped"]:
            raise AssertionError(f"prefix stopped at {earlier['step']} but current length is {t}.")

        if t < cap:
            source_idx = t
        else:             # Sample from empirical reward distribution.
            used_empirical_extension = True
            source_idx = int(rng.integers(cap))

        new_reward = reservoir[source_idx]
        if stop_rule(adversarial_seq):
            certificate = adversarial_seq[-1]
            candidate_prefix = adversarial_seq[:-1] + [new_reward]

            if stop_rule(candidate_prefix):
                break

            swapped = adversarial_seq[:-1] + [new_reward, certificate]

            # Audit check BEFORE accepting the swap.
            if alpha is not None:
                candidate_n = len(swapped)

                # For threshold adaptive BoN: LR = N * q
                likelihood_ratio = candidate_n * audit_trigger_prob

                if (audit_safe_n is None and likelihood_ratio >= 1.0 / alpha):
                    # Algorithm 1 would stop before accepting this swap.
                    audit_safe_n = t

            if stop_rule(swapped[:-1]):
                break

            adversarial_seq = swapped
            generation_source_indices.append(source_idx)
        else:          
            adversarial_seq = adversarial_seq + [new_reward]
            generation_source_indices.append(source_idx)

    adversarial_n = len(adversarial_seq)

    if audit_safe_n is None:
        audit_safe_n = adversarial_n

    earlier = find_stopping_step(adversarial_seq[:-1], stop_rule)

    assert not earlier["stopped"]
    assert len(generation_source_indices) == adversarial_n

    return {
        "samples": reservoir,
        "faithful_seq": faithful_seq,
        "faithful_n": faithful_n,
        "faithful_stopped": True,
        "adversarial_seq": adversarial_seq,
        "adversarial_n": adversarial_n,
        "audit_safe_n": audit_safe_n,
        "generation_source_indices": generation_source_indices,
        "hit_cap": adversarial_n >= cap,
        "exceeded_cap": adversarial_n > cap,
        "used_empirical_extension": used_empirical_extension,
        "adversarial_triggers": stop_rule(adversarial_seq)
    }



RESULT_FIELDS = [
    "dataset",
    "model",
    "config",
    "path",
    "temperature",
    "num_samples",
    "max_tokens",
    "alpha",
    "qid",
    "K",
    "n_available",
    "reward_threshold",
    "audit_trigger_prob",

    "faithful_stopped",
    "faithful_stop",
    "unfaithful_stop",
    "audit_safe_stop",
    "extra_samples_reg",
    "extra_samples_audit",
    "relative_overcharge_reg",
    "relative_overcharge_audit",

    # Real pre-generated reservoir
    "real_sample_headroom",
    "hit_cap",
    "exceeded_cap",
    "used_empirical_extension",

    "p1",
    "p2",
    "gap",

    # Token-level billing
    "faithful_output_tokens",
    "unfaithful_output_tokens",
    "audit_safe_output_tokens",
    "extra_output_tokens_reg",
    "extra_output_tokens_audit",
    "relative_token_overcharge_reg",
    "relative_token_overcharge_audit",
    "percent_token_overcharge_reg",
    "percent_token_overcharge_audit",
]
