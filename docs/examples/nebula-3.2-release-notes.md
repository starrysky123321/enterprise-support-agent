# Nebula Gateway 3.2 Release Notes

Page 1

Version 3.2 changes the default upstream connection pool idle timeout from 120 seconds to 30 seconds. Existing deployments with long-running requests should set `upstream.pool_idle_timeout` explicitly.

Page 2

The legacy configuration key `connection.keepalive` is deprecated. Upgrade validation emits warning code `CFG-3207` when the legacy key is present.
