# Security Policy

## Supported versions

The latest `0.2.x` revision on the default branch receives security fixes. Older snapshots are not maintained.

## Reporting a vulnerability

Please use GitHub's private security-advisory feature for this repository. Do not include credentials, private documents, access tokens, or exploit details in a public issue. If private reporting is unavailable, contact the repository maintainer through their GitHub profile before disclosing details publicly.

Include the affected revision, deployment mode, reproduction steps, impact, and any proposed mitigation. Maintainers should acknowledge a report within seven days and coordinate disclosure after a fix is available.

## Deployment requirements

- Replace every example password and JWT secret. With `ENVIRONMENT=production`, startup rejects placeholder or undersized JWT secrets and requires Secure cookies.
- Keep PostgreSQL, Redis, pgAdmin, and the Chroma data directory on trusted networks. The included Docker Compose file is for local development only.
- Store credentials in a secret manager or deployment environment. Never commit `.env`, uploaded documents, generated indexes, cookies, or production evaluation data.
- Terminate TLS at a trusted reverse proxy, restrict CORS and trusted hosts for the deployment, and apply authentication and authorization before exposing the service.

## Dependency audit note

Chroma `1.5.9` currently has published server-mode advisories without a fixed release (`PYSEC-2026-311`, `PYSEC-2026-3813`, `PYSEC-2026-3814`, and `PYSEC-2026-3815`). This project uses the embedded `PersistentClient`; it does not start or expose a Chroma HTTP server. Do not add `chroma run` or expose Chroma over the network until upstream publishes fixes and these exceptions are removed from `scripts/audit_dependencies.sh`.

Run the dependency audit with:

```bash
bash scripts/audit_dependencies.sh
```
