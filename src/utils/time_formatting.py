from datetime import timedelta


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    months, days = divmod(delta.days, 30)
    hours, remaining = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remaining, 60)

    parts = []
    if months > 0:
        parts.append(f"{months}m")
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)
