"""Turn the numbers into a short, plain-language briefing for a supply chain officer.

Uses Claude when credentials are available; otherwise falls back to a deterministic
template so the demo always works offline. The model is only asked to *explain*
numbers computed above - it never invents figures.
"""
import anthropic

MODEL = "claude-opus-5-5"

SYSTEM = (
    "You write short decision briefings for public-health supply chain officers in Africa. "
    "Use only the numbers provided; never invent data. Plain English, no jargon. "
    "Structure: one-line risk verdict, why (2-3 bullets), recommended action (1-2 bullets), "
    "and one caveat about model limits. Under 150 words."
)


def _facts(ctx: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in ctx.items())


def template_briefing(ctx: dict) -> str:
    verdict = (
        "ACT NOW" if ctx["stock-out probability"] >= 0.3
        else "WATCH" if ctx["stock-out probability"] >= 0.1
        else "ON TRACK"
    )
    action = (
        f"Arrange an emergency top-up or redistribution of about {ctx['top-up for 90% cover (units)']:,} units "
        "from a nearby facility with surplus."
        if verdict != "ON TRACK"
        else "No action needed; re-check if the shipment date moves."
    )
    return (
        f"**{verdict}: {ctx['stock-out probability']:.0%} chance of a stock-out before resupply.**\n\n"
        f"- Current stock covers about {ctx['days of cover']:.0f} days; the shipment is due in "
        f"{ctx['days to scheduled arrival']} days.\n"
        f"- The shipment is in the **{ctx['delay risk band']}** delay band "
        f"({ctx['probability >7 days late']:.0%} chance of >7 days late); in 9 of 10 scenarios it arrives by "
        f"day {ctx['90th percentile arrival day']:.0f}.\n\n"
        f"**Action:** {action}\n\n"
        "_Caveat: delay estimates come from 2006-2015 USAID shipment history and are only moderately "
        "predictive (ROC-AUC ~0.7). Treat this as a prioritisation aid, not a forecast._"
    )


def briefing(ctx: dict) -> tuple[str, str]:
    """Return (text, source) where source is 'claude' or 'template'."""
    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Write the briefing from these facts:\n{_facts(ctx)}"}],
        )
    except anthropic.AnthropicError:
        # No credentials, network down, rate limited, etc. - the demo must still work offline.
        return template_briefing(ctx), "template"
    if response.stop_reason == "refusal":
        return template_briefing(ctx), "template"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text or template_briefing(ctx)), ("claude" if text else "template")
