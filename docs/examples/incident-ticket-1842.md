# Incident Ticket INC-1842

Page 1

After upgrading Nebula Gateway from 3.1 to 3.2, the payments workspace observed intermittent `E_CONN_TIMEOUT` errors. The root cause was the new 30-second idle timeout combined with a 45-second upstream response.

Page 2

Resolution: set `upstream.pool_idle_timeout=60s`, restart one canary instance, verify the timeout metric for 15 minutes, and then roll out to the remaining instances.
