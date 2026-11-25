# Security

IndicSpec is research code that launches local inference servers bound to `127.0.0.1`. It does not handle
credentials, and no token is ever committed (`.env` is gitignored).

If you find a security issue (for example a script that exposes a server beyond localhost, or a way the harness
could execute untrusted content from a downloaded artefact), please report it privately via
[GitHub private vulnerability reporting](https://github.com/pratx1906/indicspec/security/advisories/new)
instead of opening a public issue.
