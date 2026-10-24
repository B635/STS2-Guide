"""Version identities emitted by the shipped deterministic policy code."""

POLICY_BUNDLE_VERSION = "extended-decision-policy-v9"
# The previous card-reward policy: an evaluation baseline only, never shipped.
CARD_REWARD_POLICY_VERSION = "card_reward:contextual_mechanics_v5"
DECK_FITNESS_POLICY_VERSION = "card_reward:deck_fitness_3"
# The version the shipped host answers card rewards with.
ACTIVE_CARD_REWARD_POLICY_VERSION = DECK_FITNESS_POLICY_VERSION
ROUTE_POLICY_VERSION = "route:mode_aware_risk_v4"
MERCHANT_POLICY_VERSION = "merchant_structured_v6"
CAMPFIRE_POLICY_VERSION = "campfire_structured_v5"
NEOW_POLICY_VERSION = "neow_structured_v5"
EVENT_POLICY_VERSION = "event_structured_allowlist_v5"
DECK_EDIT_POLICY_VERSION = "deck_edit:structured_targets_v4"
