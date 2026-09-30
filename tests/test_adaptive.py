from ofdm_link.adaptive import choose_profile


def test_adaptive_policy_is_monotonic() -> None:
    orders = [choose_profile(snr).order for snr in [-5, 5, 12, 25]]
    assert orders == sorted(orders)
    assert orders[0] == 2
    assert orders[-1] == 64
