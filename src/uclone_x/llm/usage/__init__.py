"""The user's limits on paid-model tokens: one system-wide total, checked before each call.

See the token-gateway design. `store` records usage, `limits` holds the limits
and the pure `check`, and `gate` wraps paid connectors in the factory.
"""
