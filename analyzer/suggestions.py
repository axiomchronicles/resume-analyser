def generate_suggestions(analysis: dict, weak_phrases, has_jd: bool, model=None, mlb=None, threshold=0.5):
    suggestions = []

    missing = [s for s, p in analysis["section_found"].items() if not p]
    if missing:
        suggestions.append(f"Missing important sections: {', '.join(missing)}")

    if has_jd and analysis["keyword_score"] < 50:
        suggestions.append("Low keyword match — tailor resume more closely to the job description.")

    if analysis["action_score"] < 60:
        suggestions.append("More bullet points should start with action verbs.")

    if analysis["metric_score"] < 40:
        suggestions.append("Add more measurable achievements (%, $, numbers).")

    if analysis["length_score"] < 60:
        if analysis["word_count"] < 200:
            suggestions.append("Resume is too short — add more detail.")
        elif analysis["word_count"] > 1200:
            suggestions.append("Resume too long — reduce irrelevant content.")

    if weak_phrases:
        wp = sorted(set(w["phrase"] for w in weak_phrases))
        suggestions.append(f"Weak phrases detected: {', '.join(wp)}")

    if model is None or mlb is None:
        return suggestions

    results = []
    proba = model.predict_proba(suggestions)

    for text, probs in zip(suggestions, proba):
        mask = probs >= threshold
        labels = mlb.classes_[mask]

        results.append({
            "suggestion": text,
            "categories": list(labels),
            "scores": {cls: float(prob) for cls, prob in zip(mlb.classes_, probs)}
        })

    return results
