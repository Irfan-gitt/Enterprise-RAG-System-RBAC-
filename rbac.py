"""Authorization constants shared by the request guard and retrieval layer."""

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "employee": {"general"},
    "finance": {"financial", "general"},
    "hr": {"hr", "general"},
    "engineering": {"engineering", "general"},
    "marketing": {"marketing", "general"},
    "admin": {"financial", "hr", "engineering", "marketing", "general"},
}
