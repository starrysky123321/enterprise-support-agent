# Connection Timeout SOP

Page 1

1. Confirm product version and capture the exact error code.
2. Compare the active configuration with the release notes for that version.
3. Measure upstream response duration and connection pool idle timeout.
4. Apply the change to one canary, observe metrics, then roll out gradually.

Page 2

If the release notes, active configuration, or request timing are unavailable, stop and request those artifacts instead of claiming a root cause.
