import logging

LOG = logging.getLogger("polyagent.calibration")

GLOBAL_SLICE = "__global__"


def brier(probability, outcome_value):
    return (probability - outcome_value) ** 2


def score_pending(db):
    pending = db.unscored_forecasts()
    scored = 0
    for row in pending:
        outcome = float(row["resolved_value"])
        forecast_probability = float(row["calibrated_probability"])
        market_probability = float(row["market_price"])
        db.insert_calibration_score({
            "forecast_id": row["forecast_id"],
            "condition_id": row["condition_id"],
            "category": row.get("category") or "uncategorized",
            "forecast_probability": forecast_probability,
            "market_probability": market_probability,
            "outcome_value": outcome,
            "model_brier": brier(forecast_probability, outcome),
            "market_brier": brier(market_probability, outcome),
        })
        scored += 1
    if scored:
        LOG.info("scored %s newly resolved forecasts", scored)
    return scored


def slice_stats(db, category=None):
    if category and category != GLOBAL_SLICE:
        rows = db.query(
            "SELECT forecast_probability, market_probability, outcome_value, "
            "model_brier, market_brier FROM calibration_scores WHERE category = ?",
            (category,),
        )
    else:
        rows = db.query(
            "SELECT forecast_probability, market_probability, outcome_value, "
            "model_brier, market_brier FROM calibration_scores"
        )
    if not rows:
        return None

    count = len(rows)
    model_brier = sum(row["model_brier"] for row in rows) / count
    market_brier = sum(row["market_brier"] for row in rows) / count
    mean_forecast = sum(row["forecast_probability"] for row in rows) / count
    mean_outcome = sum(row["outcome_value"] for row in rows) / count

    return {
        "category": category or GLOBAL_SLICE,
        "samples": count,
        "model_brier": round(model_brier, 5),
        "market_brier": round(market_brier, 5),
        "skill_vs_market": round(market_brier - model_brier, 5),
        "mean_forecast": round(mean_forecast, 5),
        "base_rate": round(mean_outcome, 5),
        "bias": round(mean_forecast - mean_outcome, 5),
    }


def category_report(db):
    categories = [
        row["category"]
        for row in db.query(
            "SELECT category, COUNT(*) AS n FROM calibration_scores "
            "GROUP BY category ORDER BY n DESC"
        )
    ]
    report = []
    overall = slice_stats(db)
    if overall:
        report.append(overall)
    for category in categories:
        stats = slice_stats(db, category)
        if stats:
            report.append(stats)
    return report


def reliability_bins(db, bins=10):
    rows = db.query(
        "SELECT forecast_probability, outcome_value FROM calibration_scores"
    )
    buckets = [{"lower": i / bins, "upper": (i + 1) / bins, "n": 0,
                "sum_forecast": 0.0, "sum_outcome": 0.0} for i in range(bins)]
    for row in rows:
        probability = float(row["forecast_probability"])
        index = min(int(probability * bins), bins - 1)
        buckets[index]["n"] += 1
        buckets[index]["sum_forecast"] += probability
        buckets[index]["sum_outcome"] += float(row["outcome_value"])
    output = []
    for bucket in buckets:
        if bucket["n"] == 0:
            continue
        output.append({
            "range": f"{bucket['lower']:.1f}-{bucket['upper']:.1f}",
            "samples": bucket["n"],
            "mean_forecast": round(bucket["sum_forecast"] / bucket["n"], 4),
            "observed_rate": round(bucket["sum_outcome"] / bucket["n"], 4),
        })
    return output


class Calibrator:
    def __init__(self, db, config):
        self.config = config
        self.cache = {}
        self.global_stats = slice_stats(db)
        for row in db.query(
            "SELECT category, COUNT(*) AS n FROM calibration_scores GROUP BY category"
        ):
            if row["n"] >= config.calibration_min_samples:
                stats = slice_stats(db, row["category"])
                if stats:
                    self.cache[row["category"]] = stats

    def _stats_for(self, category):
        key = category or "uncategorized"
        if key in self.cache:
            return self.cache[key]
        if self.global_stats and self.global_stats["samples"] >= self.config.calibration_min_samples:
            return self.global_stats
        return None

    def apply(self, probability, market_price, category=None):
        shrunk = (
            (1.0 - self.config.calibration_shrinkage) * probability
            + self.config.calibration_shrinkage * market_price
        )

        stats = self._stats_for(category)
        if stats is None:
            return min(max(shrunk, 0.001), 0.999), "shrinkage_only"

        bias = stats["bias"]
        corrected = shrunk - bias * 0.5

        if stats["skill_vs_market"] <= 0:
            corrected = 0.5 * corrected + 0.5 * market_price
            note = f"no measured skill in {stats['category']} (n={stats['samples']}), pulled to market"
        else:
            note = f"bias-corrected on {stats['category']} (n={stats['samples']}, bias={bias:+.3f})"

        return min(max(corrected, 0.001), 0.999), note
