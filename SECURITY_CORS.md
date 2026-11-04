# CORS hardening (SEC-20261103-01)

Default behavior: public cross-origin reads remain available with wildcard `Access-Control-Allow-Origin`, **without credentialed requests**. This keeps anonymous portfolio/demo usage working and removes the unsafe `allow_origins=["*"]` + `allow_credentials=True` combination.

For legitimate cookie-based browser frontends, configure exact allowed origins via environment:

```
KAVI_CORS_ORIGINS=https://your-frontend.example,https://staging.your-frontend.example
```

Never set wildcard as a credentialed trusted origin. If `KAVI_CORS_ORIGINS` is unset, CORS does not authorize credentialed cross-origin browser access. **CORS is not authentication or authorization.**

Before merging this PR: (1) confirm live frontend origins, (2) run the repo test suite and lint, (3) smoke-test unauthenticated browser access plus credentialed browser access where used, and (4) document production environment settings. No production deploy or endpoint policy override is performed by this PR.
